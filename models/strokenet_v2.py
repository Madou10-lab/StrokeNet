"""StrokeNet-v2: dynamic graph refinement, per-joint temporal SSD and ball conditioning."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.graph_temporal import (
    GraphTemporalBackbone,
    CTRGC,
    FactoredSpatialTemporalStage,
)
from models.layers.graph_utils_h36m17 import (
    PAIRS_17,
    build_adjacency_matrix,
    compute_graph_distances,
)


class CTRGCHopPrior(CTRGC):
    def __init__(
        self, in_channels, out_channels, rel_reduction=8, dist_matrix=None, max_hop=None
    ):
        super().__init__(in_channels, out_channels, rel_reduction=rel_reduction)
        assert (
            dist_matrix is not None
        ), "CTRGCHopPrior needs a (V, V) BFS distance matrix"
        if max_hop is None:
            max_hop = int(dist_matrix.max())
        self.max_hop = int(max_hop)
        dist_clipped = np.clip(np.asarray(dist_matrix, dtype=np.int64), 0, max_hop)
        self.register_buffer(
            "dist_idx", torch.from_numpy(dist_clipped).long(), persistent=False
        )
        self.bias_hop = nn.Parameter(torch.zeros(max_hop + 1))

    def _hop_bias_matrix(self):
        return self.bias_hop[self.dist_idx]

    def forward(self, x, A=None, alpha=1):
        if A is not None:
            A_eff = A + self._hop_bias_matrix().to(dtype=A.dtype, device=A.device)
        else:
            A_eff = self._hop_bias_matrix()
        return super().forward(x, A_eff, alpha)


class CTRGCHopPriorSpatialBlock(nn.Module):
    def __init__(self, dim, num_joints=17, pairs=None, max_hop=None):
        super().__init__()
        if pairs is None:
            pairs = PAIRS_17
        A_norm = build_adjacency_matrix(num_joints, pairs)
        dist_matrix = compute_graph_distances(num_joints, pairs)
        self.PA = nn.Parameter(torch.from_numpy(A_norm))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.ctrgc = CTRGCHopPrior(
            dim, dim, rel_reduction=8, dist_matrix=dist_matrix, max_hop=max_hop
        )
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


class PerJointTDown(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=2):
        super().__init__()
        self.stride = stride
        pad = (kernel_size - 1) // 2
        self.dw = nn.Conv1d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=in_dim,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm1d(in_dim)
        self.act = nn.GELU()
        self.pw = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        B, T, V, C_in = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * V, C_in, T)
        x = self.dw(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pw(x)
        x = self.bn_pw(x)
        x = self.act(x)
        C_out, T_out = (x.shape[1], x.shape[2])
        x = x.view(B, V, C_out, T_out).permute(0, 3, 1, 2).contiguous()
        return x


class StrokeNetStage(FactoredSpatialTemporalStage):
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
        max_hop=None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            cross_modal_dim=cross_modal_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            drop=drop,
            bimamba_type=bimamba_type,
            headdim=headdim,
            rmsnorm=rmsnorm,
            share_weights=share_weights,
            num_joints=num_joints,
            graph_pairs=graph_pairs,
            **kwargs,
        )
        self.spatial = CTRGCHopPriorSpatialBlock(
            dim=d_model, num_joints=num_joints, pairs=graph_pairs, max_hop=max_hop
        )


class StrokeNet(GraphTemporalBackbone):
    def __init__(self, num_joints=17, **kwargs):
        in_channels = int(kwargs.pop("in_channels", 3))
        if in_channels != 3:
            raise ValueError(
                f"StrokeNet is joint-only (in_channels=3). Got in_channels={in_channels}."
            )
        super().__init__(in_channels=3, num_joints=num_joints, **kwargs)
        dist_matrix = compute_graph_distances(num_joints, PAIRS_17)
        max_hop = int(dist_matrix.max())
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
                StrokeNetStage(
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
                    max_hop=max_hop,
                )
            )
        self.stages = new_stages
        downsample_strides = kwargs.get("downsample_strides", [1, 2, 2, 2])
        new_downsamples = nn.ModuleList()
        for i in range(self.num_stages):
            if i < self.num_stages - 1:
                stride_i = downsample_strides[i] if i < len(downsample_strides) else 2
                new_downsamples.append(
                    PerJointTDown(
                        in_dim=self.stage_dims[i],
                        out_dim=self.stage_dims[i + 1],
                        kernel_size=3,
                        stride=stride_i,
                    )
                )
            else:
                new_downsamples.append(nn.Identity())
        self.downsamples = new_downsamples
        self.ball_post_norm = nn.RMSNorm(self.cross_modal_dim)
        self.ball_stage_gate = nn.Parameter(torch.ones(self.num_stages))

    def forward(
        self,
        x_skel,
        x_ball,
        x_ball_mask=None,
        pos_q=None,
        pos_kv=None,
        return_features=False,
    ):
        from models.layers.conditioned_backbone import sinusoidal_embedding

        if x_ball.dim() != 3:
            raise ValueError(
                f"Expected ball stream with 3 dims, got shape {x_ball.shape}"
            )
        if x_ball.shape[1] == self.ball_input_dim:
            x_ball = x_ball.transpose(1, 2).contiguous()
        elif x_ball.shape[2] != self.ball_input_dim:
            raise ValueError(
                f"Ball stream shape mismatch. Expected channel dim {self.ball_input_dim}, got {x_ball.shape}"
            )
        if x_ball_mask is not None and x_ball_mask.dim() == 2:
            x_ball_mask = x_ball_mask.unsqueeze(-1)
        ball_cm = self.ball_encoder(x_ball, vis_mask=x_ball_mask)
        ball_cm = self.ball_post_norm(ball_cm)
        ball_preds = self.aux_ball_head(ball_cm)
        if len(x_skel.shape) == 5:
            B, C, T, V, M = x_skel.shape
            x_skel = x_skel.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        else:
            B, C, T, V = x_skel.shape
        if self.kendall_transform is not None:
            x_skel = self.kendall_transform(x_skel)
        out_skel = self.pose_encoder(x_skel)
        graph_pe = self.graph_dist_embed()
        out_skel = out_skel + graph_pe.unsqueeze(0).unsqueeze(0)
        T_in = out_skel.shape[1]
        C_skel = out_skel.shape[3]
        pos_t = sinusoidal_embedding(
            T_in, C_skel, device=out_skel.device, dtype=out_skel.dtype
        )
        out_skel = out_skel + pos_t.view(1, T_in, 1, C_skel)
        if self.projection is not None:
            out_skel = self.projection(out_skel)
        for i in range(self.num_stages):
            T_curr = out_skel.shape[1]
            if ball_cm.shape[1] != T_curr:
                ball_cm_stage = F.adaptive_avg_pool1d(
                    ball_cm.transpose(1, 2), T_curr
                ).transpose(1, 2)
            else:
                ball_cm_stage = ball_cm
            ball_cm_stage = ball_cm_stage * self.ball_stage_gate[i]
            out_skel = self.stages[i](out_skel, ball_cm_stage)
            out_skel = self.downsamples[i](out_skel)
        skel_features = out_skel.mean(dim=(1, 2))
        skel_features_norm = self.norm_final(skel_features)
        skel_features_drop = self.head_drop(skel_features_norm)
        logits = self.classifier(skel_features_drop)
        if return_features:
            return (logits, skel_features_norm, skel_features_norm, ball_preds)
        return (logits, ball_preds)


def StrokeNet_(**kwargs):
    return StrokeNet(**kwargs)
