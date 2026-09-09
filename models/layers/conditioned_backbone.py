"""Shared trajectory encoder and ball-conditioned Mamba blocks for skeletal tokens."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mamba_ssm.modules.mamba2_conditioned import Mamba2 as Mamba
from models.layers.graph_utils import (
    GraphConvLayer,
    GraphDistanceEmbedding,
    KendallTransformLayer,
)


def sinusoidal_embedding(length: int, dim: int, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    emb = torch.zeros(length, dim, device=device, dtype=dtype)
    emb[:, 0::2] = torch.sin(position[:, None] * div_term)
    emb[:, 1::2] = torch.cos(position[:, None] * div_term)
    return emb


class SimpleMLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AdaptiveTDownsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=2):
        super().__init__()
        self.joint_mix = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.stride = stride

    def forward(self, x):
        B, T, V, C_in = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.joint_mix(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).reshape(B * V, C_in, T)
        x = self.conv(x)
        x = x.view(B, V, -1, x.shape[-1]).permute(0, 3, 1, 2)
        if x.stride(-1) % 8 != 0:
            x = x.contiguous()
        return x


class TemporalDSConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, drop=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw1 = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.pw1 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.dw2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.bn2 = nn.GroupNorm(1, dim)

    def forward(self, x, mask_bct=None):
        res = x
        if mask_bct is not None:
            res = res * mask_bct.to(dtype=x.dtype)
            x = x * mask_bct.to(dtype=x.dtype)
        x = self.dw1(x)
        x = self.pw1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.dw2(x)
        x = self.pw2(x)
        x = self.bn2(x)
        x = self.act(x)
        if mask_bct is not None:
            x = x * mask_bct.to(dtype=x.dtype)
        return x + res


class GCN_Pose_Encoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, num_joints=25):
        super().__init__()
        self.linear = nn.Linear(in_channels, embed_dim)
        self.gcn = GraphConvLayer(embed_dim, embed_dim, num_joints=num_joints)
        self.norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        x = self.gcn(x)
        x = self.norm(x)
        return x


class CrossModalMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        cross_modal_dim=64,
        mlp_ratio=4.0,
        drop=0.0,
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba_type="v2",
        headdim=None,
        rmsnorm=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type=bimamba_type,
            headdim=headdim,
            rmsnorm=rmsnorm,
            cross_modal_dim=cross_modal_dim,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SimpleMLP(d_model, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x, cross_modal=None):
        res = x
        x_norm = self.norm1(x)
        x_mamba = self.mamba(x_norm, cross_modal=cross_modal)
        x = res + x_mamba
        res2 = x
        x_norm2 = self.norm2(x)
        x_ffn = self.mlp(x_norm2)
        x = res2 + x_ffn
        return x


class CrossModalMambaStage(nn.Module):
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
        num_joints=25,
        **kwargs,
    ):
        super().__init__()
        self.num_joints = num_joints
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
        B, T, V, C = skel.shape
        L = T * V
        x_seq = skel.reshape(B, L, C)
        cm_broadcast = ball_cm.unsqueeze(2).expand(B, T, V, -1).reshape(B, L, -1)
        x_seq = self.block1(x_seq, cross_modal=cm_broadcast)
        x_seq = self.block2(x_seq, cross_modal=cm_broadcast)
        skel_out = x_seq.view(B, T, V, C)
        return skel_out


class BallTCNEncoder(nn.Module):
    def __init__(self, input_dim=7, output_dim=64, depth=2, drop=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        blocks = []
        for i in range(depth):
            dilation = 2**i
            blocks.append(
                TemporalDSConvBlock(
                    output_dim, kernel_size=5, dilation=dilation, drop=drop
                )
            )
        self.tcn = nn.ModuleList(blocks)

    def forward(self, x, vis_mask=None):
        mask = None
        if vis_mask is not None:
            if vis_mask.dim() == 3:
                vis_mask = vis_mask.squeeze(-1)
            mask = vis_mask.to(device=x.device, dtype=x.dtype)
            x = x * mask.unsqueeze(-1)
        x = self.norm(x)
        x = self.proj(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        mask_bct = mask.unsqueeze(1) if mask is not None else None
        for block in self.tcn:
            x = block(x, mask_bct=mask_bct)
        x = x.transpose(1, 2)
        return x


class ConditionedBackbone(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dim=192,
        stage_dims=None,
        num_classes=10,
        d_state=128,
        d_conv=4,
        expand=1,
        mlp_ratio=2.0,
        drop=0.1,
        bimamba_type="v2",
        headdim=48,
        headdim_256=32,
        headdim_384=48,
        headdim_512=64,
        rmsnorm=True,
        downsample_strides=None,
        ball_input_dim=7,
        cross_modal_dim=64,
        ball_tcn_depth=2,
        ball_tcn_drop=0.1,
        num_joints=25,
        share_weights=True,
        share_weights_per_stage=None,
        mlp_ratio_per_stage=None,
        expand_per_stage=None,
        kendall_transform_type="none",
        window_size=64,
        **kwargs,
    ):
        super().__init__()
        if stage_dims is None:
            stage_dims = [192, 256, 384, 512]
        if downsample_strides is None:
            downsample_strides = [1, 2, 2, 2]
        self.num_stages = len(stage_dims)
        self.stage_dims = stage_dims
        self.ball_input_dim = ball_input_dim
        self.cross_modal_dim = cross_modal_dim
        self.downsample_strides = downsample_strides
        if kendall_transform_type in ("rigid", "nonrigid"):
            self.kendall_transform = KendallTransformLayer(
                transform_type=kendall_transform_type,
                num_frames=window_size,
                num_joints=num_joints,
            )
        else:
            self.kendall_transform = None
        self.pose_encoder = GCN_Pose_Encoder(
            in_channels=in_channels, embed_dim=embed_dim, num_joints=num_joints
        )
        self.graph_dist_embed = GraphDistanceEmbedding(
            embed_dim=embed_dim, max_dist=10, root_joint=7, num_joints=num_joints
        )
        self.projection = None
        if embed_dim != stage_dims[0]:
            self.projection = nn.Linear(embed_dim, stage_dims[0])
        self.ball_encoder = BallTCNEncoder(
            input_dim=ball_input_dim,
            output_dim=cross_modal_dim,
            depth=ball_tcn_depth,
            drop=ball_tcn_drop,
        )
        self.aux_ball_head = nn.Linear(cross_modal_dim, 2)
        stages = []
        downsamples = []
        for i in range(self.num_stages):
            d_model_i = stage_dims[i]
            headdim_i = self._get_headdim(
                d_model_i, headdim, headdim_256, headdim_384, headdim_512
            )
            mlp_ratio_i = self._get_per_stage(mlp_ratio_per_stage, i, mlp_ratio)
            expand_i = int(self._get_per_stage(expand_per_stage, i, expand))
            share_i = self._get_per_stage(share_weights_per_stage, i, share_weights)
            stage = CrossModalMambaStage(
                d_model=d_model_i,
                cross_modal_dim=cross_modal_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand_i,
                mlp_ratio=mlp_ratio_i,
                drop=drop,
                bimamba_type=bimamba_type,
                headdim=headdim_i,
                rmsnorm=rmsnorm,
                share_weights=bool(share_i),
                num_joints=num_joints,
            )
            stages.append(stage)
            if i < self.num_stages - 1:
                stride_i = downsample_strides[i] if i < len(downsample_strides) else 2
                ds = AdaptiveTDownsample(
                    in_dim=stage_dims[i],
                    out_dim=stage_dims[i + 1],
                    stride=stride_i,
                    kernel_size=3,
                )
            else:
                ds = nn.Identity()
            downsamples.append(ds)
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        final_dim = stage_dims[-1]
        self.norm_final = nn.RMSNorm(final_dim)
        self.head_drop = nn.Dropout(drop)
        self.classifier = nn.Linear(final_dim, num_classes)
        self.feat_dim = final_dim

    @staticmethod
    def _get_headdim(d_model, hd, hd_256, hd_384, hd_512):
        if d_model == 192:
            return hd
        elif d_model == 256:
            return hd_256
        elif d_model == 384:
            return hd_384
        elif d_model == 512:
            return hd_512
        else:
            return hd

    @staticmethod
    def _get_per_stage(per_stage_list, idx, default):
        if (
            isinstance(per_stage_list, (list, tuple))
            and idx < len(per_stage_list)
            and (per_stage_list[idx] is not None)
        ):
            return per_stage_list[idx]
        return default

    def forward(
        self,
        x_skel,
        x_ball,
        x_ball_mask=None,
        pos_q=None,
        pos_kv=None,
        return_features=False,
    ):
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
            out_skel = self.stages[i](out_skel, ball_cm_stage)
            out_skel = self.downsamples[i](out_skel)
        skel_features = out_skel.mean(dim=(1, 2))
        skel_features_norm = self.norm_final(skel_features)
        skel_features_drop = self.head_drop(skel_features_norm)
        logits = self.classifier(skel_features_drop)
        if return_features:
            return (logits, skel_features_norm, skel_features_norm, ball_preds)
        else:
            return (logits, ball_preds)

    def get_feature_dim(self):
        return self.feat_dim


def ConditionedBackbone_(**kwargs):
    return ConditionedBackbone(**kwargs)
