"""Channel-wise dynamic graph layers and factored spatial–temporal stages."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.conditioned_backbone import CrossModalMambaBlock
from models.layers.h36m_backbone import H36MBackbone
from models.layers.graph_utils_h36m17 import build_adjacency_matrix, PAIRS_17


def _conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _conv_init(m)

    def forward(self, x, A=None, alpha=1):
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (
            A.unsqueeze(0).unsqueeze(0) if A is not None else 0
        )
        x1 = torch.einsum("ncuv,nctv->nctu", x1, x3)
        return x1


class CTRGCSpatialBlock(nn.Module):
    def __init__(self, dim, num_joints=17, pairs=None):
        super().__init__()
        A_norm = build_adjacency_matrix(num_joints, pairs)
        self.PA = nn.Parameter(torch.from_numpy(A_norm))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.ctrgc = CTRGC(dim, dim, rel_reduction=8)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.bn.weight, 1e-06)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x_nctv = x.permute(0, 3, 1, 2).contiguous()
        z = self.ctrgc(x_nctv, self.PA, self.alpha)
        z = self.bn(z)
        z = z + x_nctv
        z = self.relu(z)
        return z.permute(0, 2, 3, 1).contiguous()


class FactoredSpatialTemporalStage(nn.Module):
    def __init__(
        self,
        d_model,
        cross_modal_dim=64,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        drop=0.0,
        bimamba_type="v2",
        headdim=None,
        rmsnorm=False,
        share_weights=True,
        num_joints=17,
        graph_pairs=None,
        **kwargs
    ):
        super().__init__()
        self.num_joints = num_joints
        self.spatial = CTRGCSpatialBlock(
            dim=d_model, num_joints=num_joints, pairs=graph_pairs
        )
        self.block1 = CrossModalMambaBlock(
            d_model=d_model,
            cross_modal_dim=cross_modal_dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type=bimamba_type,
            headdim=headdim,
            rmsnorm=rmsnorm,
        )
        if share_weights:
            self.block2 = self.block1
        else:
            self.block2 = CrossModalMambaBlock(
                d_model=d_model,
                cross_modal_dim=cross_modal_dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type=bimamba_type,
                headdim=headdim,
                rmsnorm=rmsnorm,
            )

    def forward(self, skel, ball_cm):
        skel = self.spatial(skel)
        B, T, V, C = skel.shape
        cm_dim = ball_cm.shape[-1]
        x_temp = skel.permute(0, 2, 1, 3).reshape(B * V, T, C).contiguous()
        cm = ball_cm.unsqueeze(1).expand(B, V, T, cm_dim)
        cm = cm.reshape(B * V, T, cm_dim).contiguous()
        x_temp = self.block1(x_temp, cross_modal=cm)
        x_temp = self.block2(x_temp, cross_modal=cm)
        skel = x_temp.view(B, V, T, C).permute(0, 2, 1, 3)
        if not skel.is_contiguous():
            skel = skel.contiguous()
        return skel


class GraphTemporalBackbone(H36MBackbone):
    def __init__(self, num_joints=17, **kwargs):
        v9_cfg = {
            "d_state": kwargs.get("d_state", 128),
            "d_conv": kwargs.get("d_conv", 4),
            "expand": kwargs.get("expand", 1),
            "mlp_ratio": kwargs.get("mlp_ratio", 2.0),
            "drop": kwargs.get("drop", 0.1),
            "bimamba_type": kwargs.get("bimamba_type", "v2"),
            "rmsnorm": kwargs.get("rmsnorm", True),
            "share_weights": kwargs.get("share_weights", True),
            "share_weights_per_stage": kwargs.get("share_weights_per_stage", None),
            "mlp_ratio_per_stage": kwargs.get("mlp_ratio_per_stage", None),
            "expand_per_stage": kwargs.get("expand_per_stage", None),
            "cross_modal_dim": kwargs.get("cross_modal_dim", 64),
            "headdim": kwargs.get("headdim", 48),
            "headdim_256": kwargs.get("headdim_256", 32),
            "headdim_384": kwargs.get("headdim_384", 48),
            "headdim_512": kwargs.get("headdim_512", 64),
        }
        super().__init__(num_joints=num_joints, **kwargs)
        new_stages = nn.ModuleList()
        for i in range(self.num_stages):
            d_model_i = self.stage_dims[i]
            headdim_i = self._get_headdim(
                d_model_i,
                v9_cfg["headdim"],
                v9_cfg["headdim_256"],
                v9_cfg["headdim_384"],
                v9_cfg["headdim_512"],
            )
            mlp_ratio_i = self._get_per_stage(
                v9_cfg["mlp_ratio_per_stage"], i, v9_cfg["mlp_ratio"]
            )
            expand_i = int(
                self._get_per_stage(v9_cfg["expand_per_stage"], i, v9_cfg["expand"])
            )
            share_i = self._get_per_stage(
                v9_cfg["share_weights_per_stage"], i, v9_cfg["share_weights"]
            )
            new_stages.append(
                FactoredSpatialTemporalStage(
                    d_model=d_model_i,
                    cross_modal_dim=v9_cfg["cross_modal_dim"],
                    d_state=v9_cfg["d_state"],
                    d_conv=v9_cfg["d_conv"],
                    expand=expand_i,
                    mlp_ratio=mlp_ratio_i,
                    drop=v9_cfg["drop"],
                    bimamba_type=v9_cfg["bimamba_type"],
                    headdim=headdim_i,
                    rmsnorm=v9_cfg["rmsnorm"],
                    share_weights=bool(share_i),
                    num_joints=num_joints,
                    graph_pairs=PAIRS_17,
                )
            )
        self.stages = new_stages


def GraphTemporalBackbone_(**kwargs):
    return GraphTemporalBackbone(**kwargs)
