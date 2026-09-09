"""Hierarchical bidirectional Mamba backbone, trajectory TCN and body-part cross-attention."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even."
        inv_freq = base ** (-torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(
        self, seq_len, device=None, dtype=None, positions: torch.Tensor | None = None
    ):
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        if positions is None:
            t = torch.arange(
                seq_len,
                device=device,
                dtype=dtype if dtype is not None else inv_freq.dtype,
            )
        else:
            t = positions.to(
                device=device, dtype=dtype if dtype is not None else inv_freq.dtype
            )
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return (cos, sin)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def sinusoidal_embedding(length: int, dim: int, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype).float()
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    emb = torch.zeros(length, dim, device=device, dtype=dtype)
    emb[:, 0::2] = torch.sin(position[:, None] * div_term)
    emb[:, 1::2] = torch.cos(position[:, None] * div_term)
    return emb


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.1,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert head_dim % 2 == 0, "RoPE requires head_dim to be even."
        self.head_dim = head_dim
        self.norm_q = nn.RMSNorm(dim)
        self.norm_kv = nn.RMSNorm(dim)
        self.norm_ff = nn.RMSNorm(dim)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x_q,
        x_kv,
        kv_mask=None,
        pos_q: torch.Tensor | None = None,
        pos_kv: torch.Tensor | None = None,
        proximity_mask: torch.Tensor | None = None,
    ):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape
        x_qn = self.norm_q(x_q)
        x_kvn = self.norm_kv(x_kv)
        q = (
            self.wq(x_qn)
            .reshape(B, N_q, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.wk(x_kvn)
            .reshape(B, N_kv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.wv(x_kvn)
            .reshape(B, N_kv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        cos_q, sin_q = self.rope.get_cos_sin(
            N_q, device=q.device, dtype=q.dtype, positions=pos_q
        )
        cos_k, sin_k = self.rope.get_cos_sin(
            N_kv, device=k.device, dtype=k.dtype, positions=pos_kv
        )
        q = apply_rotary_emb(q, cos_q, sin_q)
        k = apply_rotary_emb(k, cos_k, sin_k)
        full_mask = None
        if kv_mask is not None:
            kv_mask = kv_mask.to(q.device).to(torch.bool)
            mask_kv = ~kv_mask.unsqueeze(1).unsqueeze(2)
            full_mask = mask_kv if full_mask is None else full_mask | mask_kv
        if proximity_mask is not None:
            prox = proximity_mask.to(q.device).to(torch.bool)
            if prox.dim() == 2:
                prox = prox.unsqueeze(0).unsqueeze(1)
            elif prox.dim() == 3:
                prox = prox.unsqueeze(1)
            else:
                raise ValueError(
                    f"proximity_mask must have dim 2 or 3, got {prox.dim()}"
                )
            full_mask = prox if full_mask is None else full_mask | prox
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=full_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        x = attn_out.transpose(1, 2).contiguous().view(B, N_q, C)
        x = self.proj_drop(self.proj(x))
        x = x_q + x
        x_ffn = self.ffn(self.norm_ff(x))
        x = x + x_ffn
        return x


class Pose_Encoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, act_layer=nn.GELU):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1),
            act_layer(),
            nn.Conv2d(2 * in_channels, 3 * in_channels, kernel_size=1),
            act_layer(),
            nn.Conv2d(3 * in_channels, embed_dim, kernel_size=1),
        ]
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)


from mamba_ssm import Mamba2 as Mamba


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


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        mlp_ratio=4.0,
        drop=0.0,
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba_type="v2",
        headdim=None,
        rmsnorm=False,
        save_params=False,
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
            save_params=save_params,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SimpleMLP(d_model, mlp_ratio=mlp_ratio, drop=drop)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.RMSNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        res = x
        x_norm = self.norm1(x)
        x_mamba = self.mamba(x_norm)
        x = res + x_mamba
        res2 = x
        x_norm2 = self.norm2(x)
        x_ffn = self.mlp(x_norm2)
        x = res2 + x_ffn
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dilation=1, drop=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.GroupNorm(1, dim)

    def forward(self, x, mask_bct: torch.Tensor | None = None):
        if mask_bct is not None:
            res = x * mask_bct.to(dtype=x.dtype)
            x = res
        else:
            res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        if mask_bct is not None:
            x = x * mask_bct.to(dtype=x.dtype)
        return x + res


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

    def forward(self, x, mask_bct: torch.Tensor | None = None):
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


class BallStreamEncoder(nn.Module):
    def __init__(
        self,
        input_dim=7,
        output_dim=256,
        depth=3,
        drop=0.1,
        use_pos_enc=False,
        dw_separable: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.use_pos_enc = use_pos_enc
        blocks = []
        for i in range(depth):
            dilation = 2**i
            if dw_separable:
                blocks.append(
                    TemporalDSConvBlock(
                        output_dim, kernel_size=5, dilation=dilation, drop=drop
                    )
                )
            else:
                blocks.append(
                    TemporalConvBlock(
                        output_dim, kernel_size=5, dilation=dilation, drop=drop
                    )
                )
        self.tcn = nn.ModuleList(blocks)
        self.final_ln = nn.Identity()

    def forward(self, x, vis_mask: torch.Tensor | None = None):
        mask = None
        if vis_mask is not None:
            if vis_mask.dim() == 3:
                vis_mask = vis_mask.squeeze(-1)
            mask = vis_mask.to(device=x.device, dtype=x.dtype)
            x = x * mask.unsqueeze(-1)
        x = self.norm(x)
        x = self.proj(x)
        if self.use_pos_enc:
            pos = sinusoidal_embedding(
                x.shape[1], x.shape[2], device=x.device, dtype=x.dtype
            )
            x = x + pos.unsqueeze(0)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        mask_bct = None
        if mask is not None:
            mask_bct = mask.unsqueeze(1)
        for block in self.tcn:
            x = block(x, mask_bct=mask_bct)
        x = x.transpose(1, 2)
        x = self.final_ln(x)
        return x


class AdaptiveTDownsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=2):
        super().__init__()
        if kernel_size == 3:
            assert (
                kernel_size - 1
            ) // 2 == 1, "Padding must be 1 for k=3 to preserve center alignment"
        self.joint_mix = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_dim,
                out_channels=out_dim,
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
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * V, C_in, T)
        x = self.conv(x)
        x = x.view(B, V, -1, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        if x.stride(-1) % 8 != 0:
            x = x.contiguous()
        return x


class MambaStage(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        drop=0.0,
        bimamba_type="v2",
        headdim=None,
        rmsnorm=False,
        stage_idx=-1,
        m_matrix_target_layer=None,
        share_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        target_this_stage = (
            m_matrix_target_layer is not None and m_matrix_target_layer[0] == stage_idx
        )
        save_params_block1 = target_this_stage and m_matrix_target_layer[1] == 0
        save_params_block2 = target_this_stage and m_matrix_target_layer[1] == 1
        self.block1 = MambaBlock(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type=bimamba_type,
            headdim=headdim,
            rmsnorm=rmsnorm,
            save_params=save_params_block1,
        )
        if share_weights:
            self.block2 = self.block1
        else:
            self.block2 = MambaBlock(
                d_model=d_model,
                mlp_ratio=mlp_ratio,
                drop=drop,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type=bimamba_type,
                headdim=headdim,
                rmsnorm=rmsnorm,
                save_params=save_params_block2,
            )

    def forward(self, x):
        B, T, V, C = x.shape
        L = T * V
        x_seq = x.reshape(B, L, C)
        x_seq = self.block1(x_seq)
        x_seq = self.block2(x_seq)
        x = x_seq.view(B, T, V, C)
        return x


class PoseHead(nn.Module):
    def __init__(self, d_model, num_classes=60, drop=0.0, return_features=False):
        super().__init__()
        self.dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.linear = (
            nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.return_features = return_features

    def forward(self, x):
        x_mean = x.mean(dim=(1, 2))
        x_mean = self.dropout(x_mean)
        logits = self.linear(x_mean)
        if self.return_features:
            return (logits, x_mean)
        else:
            return logits


class MambaSkeletonActionModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dim=64,
        stage_dims=None,
        num_classes=60,
        d_state=16,
        d_conv=4,
        expand=1,
        mlp_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        bimamba_type="v2",
        headdim=None,
        headdim_256=None,
        headdim_384=None,
        headdim_512=None,
        rmsnorm=False,
        ball_input_dim=7,
        fusion_layers: int = 1,
        attn_heads: int = 8,
        attn_dim: int | None = 128,
        ball_tcn_depth: int = 3,
        ball_tcn_drop: float = 0.1,
        ball_pos_enc: bool = True,
        fusion_local_window: int | None = 8,
        mid_fusion_stage: int | None = 1,
        fusion_bridge_stages=None,
        fusion_bridge_scale: float = 0.5,
        fusion_bridge_drop: float = 0.1,
        use_bridge_fusion: bool = True,
        use_mid_fusion: bool = True,
        use_late_fusion: bool = True,
        share_weights: bool = True,
        ball_dw_separable: bool = True,
        share_weights_per_stage=None,
        mlp_ratio_per_stage=None,
        expand_per_stage=None,
        m_matrix_target_layer=None,
        downsample_strides=None,
        **kwargs,
    ):
        super().__init__()
        self._m_matrix_target_layer_config = m_matrix_target_layer
        self.ball_input_dim = ball_input_dim
        self.fusion_local_window = fusion_local_window
        self.mid_fusion_stage = mid_fusion_stage
        self.fusion_bridge_stages = (
            fusion_bridge_stages if fusion_bridge_stages is not None else []
        )
        if isinstance(self.fusion_bridge_stages, int):
            self.fusion_bridge_stages = [self.fusion_bridge_stages]
        self.use_bridge_fusion = bool(use_bridge_fusion)
        self.use_mid_fusion = bool(use_mid_fusion)
        self.use_late_fusion = bool(use_late_fusion)
        self.downsample_strides = (
            downsample_strides if downsample_strides is not None else [1, 2, 2, 2]
        )
        self.body_parts = {
            "torso": [0, 7, 8, 9, 10],
            "right_arm": [14, 15, 16, 17, 18],
            "left_arm": [11, 12, 13, 19, 20],
            "right_leg": [1, 2, 3, 21, 22],
            "left_leg": [4, 5, 6, 23, 24],
        }
        self.num_parts = len(self.body_parts)
        self.part_names_sorted = sorted(self.body_parts.keys())
        self.initial_encoder = Pose_Encoder(
            in_channels=in_channels, embed_dim=embed_dim
        )
        if stage_dims is None:
            stage_dims = [192, 256, 384, 512]
        self.attn_dim = attn_dim
        self.num_stages = len(stage_dims)
        stages = []
        downsamples = []
        self.projection = None
        self._target_mamba_layer_ref = None
        if embed_dim != stage_dims[0]:
            self.projection = nn.Linear(embed_dim, stage_dims[0])
        default_attn_dim = stage_dims[1]
        attn_dim_eff = attn_dim if attn_dim is not None else default_attn_dim
        if attn_dim_eff % attn_heads != 0:
            raise ValueError(
                f"attn_dim ({attn_dim_eff}) must be divisible by attn_heads ({attn_heads})"
            )
        self.ball_stream = BallStreamEncoder(
            input_dim=ball_input_dim,
            output_dim=attn_dim_eff,
            depth=ball_tcn_depth,
            drop=ball_tcn_drop,
            use_pos_enc=ball_pos_enc,
            dw_separable=ball_dw_separable,
        )
        if self.mid_fusion_stage is not None:
            if not 0 <= self.mid_fusion_stage < len(stage_dims):
                raise ValueError(
                    f"mid_fusion_stage should be in [0, {len(stage_dims) - 1}], got {self.mid_fusion_stage}"
                )
            mid_dim = stage_dims[self.mid_fusion_stage]
            if mid_dim % attn_heads != 0:
                raise ValueError(
                    f"mid_fusion_stage dim {mid_dim} must be divisible by attn_heads {attn_heads}"
                )
            self.ball_to_mid_proj = nn.Linear(attn_dim, mid_dim)
            self.mid_fusion_block = CrossAttentionBlock(
                dim=mid_dim, num_heads=attn_heads, drop=drop, attn_drop=attn_drop
            )
        else:
            self.ball_to_mid_proj = None
            self.mid_fusion_block = None
        self.fusion_bridge_blocks = nn.ModuleDict()
        self.fusion_bridge_proj = nn.ModuleDict()
        self.fusion_bridge_drop = (
            nn.Dropout(fusion_bridge_drop) if fusion_bridge_drop > 0 else nn.Identity()
        )
        self.fusion_bridge_scale = fusion_bridge_scale
        for s_idx in self.fusion_bridge_stages:
            if not 0 <= s_idx < len(stage_dims):
                raise ValueError(
                    f"fusion_bridge_stages entry {s_idx} must be within [0, {len(stage_dims) - 1}]"
                )
            dim_s = stage_dims[s_idx]
            if dim_s % attn_heads != 0:
                raise ValueError(
                    f"Stage {s_idx} dim {dim_s} must be divisible by attn_heads {attn_heads} for fusion bridge"
                )
            self.fusion_bridge_blocks[str(s_idx)] = CrossAttentionBlock(
                dim=dim_s, num_heads=attn_heads, drop=drop, attn_drop=attn_drop
            )
            self.fusion_bridge_proj[str(s_idx)] = nn.Linear(attn_dim, dim_s)
        skeleton_final_dim = stage_dims[-1]
        attn_dim = attn_dim_eff
        self.skel_proj_seq = nn.Linear(skeleton_final_dim, attn_dim)
        self.skel_token_dropout = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.fusion_attention_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=attn_dim, num_heads=attn_heads, drop=0.15, attn_drop=attn_drop
                )
                for _ in range(max(1, fusion_layers))
            ]
        )
        self.aux_ball_head = nn.Linear(attn_dim, 2)
        self.norm_final = nn.RMSNorm(attn_dim)
        self.classifier_head = nn.Linear(attn_dim, num_classes)
        self.fusion_pool = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, max(attn_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(attn_dim // 2, 1), 1),
        )
        self.skeleton_skip_scale = nn.Parameter(torch.tensor(1.0))
        self.pose_head = PoseHead(
            d_model=skeleton_final_dim,
            num_classes=num_classes,
            drop=drop,
            return_features=True,
        )
        self.feat_dim = attn_dim
        for i in range(self.num_stages):
            d_model_i = stage_dims[i]
            if d_model_i == 192:
                headdim_i = headdim
            elif d_model_i == 256:
                headdim_i = headdim_256
            elif d_model_i == 384:
                headdim_i = headdim_384
            elif d_model_i == 512:
                headdim_i = headdim_512
            else:
                raise ValueError(f"Unsupported d_model: {d_model_i}")
            mlp_ratio_i = mlp_ratio
            if (
                isinstance(mlp_ratio_per_stage, (list, tuple))
                and i < len(mlp_ratio_per_stage)
                and (mlp_ratio_per_stage[i] is not None)
            ):
                mlp_ratio_i = float(mlp_ratio_per_stage[i])
            expand_i = expand
            if (
                isinstance(expand_per_stage, (list, tuple))
                and i < len(expand_per_stage)
                and (expand_per_stage[i] is not None)
            ):
                expand_i = int(expand_per_stage[i])
            share_i = share_weights
            if (
                isinstance(share_weights_per_stage, (list, tuple))
                and i < len(share_weights_per_stage)
                and (share_weights_per_stage[i] is not None)
            ):
                share_i = bool(share_weights_per_stage[i])
            stage_i = MambaStage(
                d_model=d_model_i,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand_i,
                mlp_ratio=mlp_ratio_i,
                drop=drop,
                bimamba_type=bimamba_type,
                headdim=headdim_i,
                rmsnorm=rmsnorm,
                stage_idx=i,
                m_matrix_target_layer=self._m_matrix_target_layer_config,
                share_weights=share_i,
            )
            stages.append(stage_i)
            if (
                self._m_matrix_target_layer_config is not None
                and self._m_matrix_target_layer_config[0] == i
            ):
                layer_idx_in_stage = self._m_matrix_target_layer_config[1]
                target_block = None
                if layer_idx_in_stage == 0 and hasattr(stage_i, "block1"):
                    target_block = stage_i.block1
                elif layer_idx_in_stage == 1 and hasattr(stage_i, "block2"):
                    target_block = stage_i.block2
                if target_block and hasattr(target_block, "mamba"):
                    self._target_mamba_layer_ref = target_block.mamba
            if i < self.num_stages - 1:
                curr_stride = (
                    self.downsample_strides[i]
                    if i < len(self.downsample_strides)
                    else 2
                )
                ds = AdaptiveTDownsample(
                    in_dim=stage_dims[i],
                    out_dim=stage_dims[i + 1],
                    stride=curr_stride,
                    kernel_size=3,
                )
            else:
                ds = nn.Identity()
            downsamples.append(ds)
        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)

    def get_computed_m_matrix(self):
        if self._target_mamba_layer_ref:
            if hasattr(self._target_mamba_layer_ref, "debug_M"):
                return self._target_mamba_layer_ref.debug_M
            else:
                return None
        else:
            return None

    def forward(
        self,
        x_skel,
        x_ball,
        x_ball_mask=None,
        pos_q: torch.Tensor | None = None,
        pos_kv: torch.Tensor | None = None,
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
        feats_ball_seq = self.ball_stream(
            x_ball, vis_mask=x_ball_mask if x_ball_mask is not None else None
        )
        ball_preds = self.aux_ball_head(feats_ball_seq)
        ball_time_positions = torch.arange(
            feats_ball_seq.shape[1],
            device=feats_ball_seq.device,
            dtype=feats_ball_seq.dtype,
        )
        attn_mask = None
        if x_ball_mask is not None:
            attn_mask = x_ball_mask.squeeze(-1).to(torch.bool).to(feats_ball_seq.device)
        if len(x_skel.shape) == 5:
            B, C, T, V, M = x_skel.shape
            x_skel = x_skel.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        else:
            B, C, T, V = x_skel.shape
            x_skel = x_skel.permute(0, 1, 2, 3).contiguous().view(B, C, T, -1)
        out_skel = self.initial_encoder(x_skel)
        if self.projection is not None:
            out_skel = self.projection(out_skel.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out_skel = out_skel.permute(0, 2, 3, 1).contiguous()
        T_in = out_skel.shape[1]
        V_total = out_skel.shape[2]
        C_skel = out_skel.shape[3]
        pos_t = sinusoidal_embedding(
            T_in, C_skel, device=out_skel.device, dtype=out_skel.dtype
        ).view(1, T_in, 1, C_skel)
        pos_v = sinusoidal_embedding(
            V_total, C_skel, device=out_skel.device, dtype=out_skel.dtype
        ).view(1, 1, V_total, C_skel)
        out_skel = out_skel + pos_t + pos_v
        current_stride = 1
        for i in range(self.num_stages):
            out_skel = self.stages[i](out_skel)
            if self.use_bridge_fusion and str(i) in self.fusion_bridge_blocks:
                B_s, T_s, V_s, C_s = out_skel.shape
                part_tokens = []
                for name in self.part_names_sorted:
                    indices = self.body_parts[name]
                    part_tokens.append(out_skel[:, :, indices, :].mean(dim=2))
                part_tokens = torch.stack(part_tokens, dim=2)
                q = part_tokens.view(B_s, T_s * self.num_parts, C_s)
                kv = feats_ball_seq
                if kv.shape[-1] != C_s and str(i) in self.fusion_bridge_proj:
                    kv = self.fusion_bridge_proj[str(i)](kv)
                pos_q_stage = torch.arange(
                    T_s, device=out_skel.device, dtype=feats_ball_seq.dtype
                )
                pos_q_stage = (
                    pos_q_stage.repeat_interleave(self.num_parts) * current_stride
                )
                pos_kv_stage = ball_time_positions
                proximity_mask = None
                if (
                    self.fusion_local_window is not None
                    and kv.shape[1] > 0
                    and (q.shape[1] > 0)
                ):
                    dist = torch.abs(pos_q_stage[:, None] - pos_kv_stage[None, :])
                    proximity_mask = dist > float(self.fusion_local_window)
                fused_stage = self.fusion_bridge_blocks[str(i)](
                    x_q=q,
                    x_kv=kv,
                    kv_mask=attn_mask,
                    pos_q=pos_q_stage,
                    pos_kv=pos_kv_stage,
                    proximity_mask=proximity_mask,
                )
                fused_stage = self.fusion_bridge_drop(fused_stage)
                fused_parts = fused_stage.view(B_s, T_s, self.num_parts, C_s)
                skel_update = torch.zeros_like(out_skel)
                for idx, name in enumerate(self.part_names_sorted):
                    indices = self.body_parts[name]
                    part_context = fused_parts[:, :, idx, :].unsqueeze(2)
                    for j_idx in indices:
                        skel_update[:, :, j_idx, :] += part_context.squeeze(2)
                out_skel = out_skel + self.fusion_bridge_scale * skel_update
            if (
                self.use_mid_fusion
                and self.mid_fusion_block is not None
                and (i == self.mid_fusion_stage)
            ):
                B_s, T_s, V_s, C_s = out_skel.shape
                part_tokens = []
                for name in self.part_names_sorted:
                    indices = self.body_parts[name]
                    part_tokens.append(out_skel[:, :, indices, :].mean(dim=2))
                part_tokens = torch.stack(part_tokens, dim=2)
                q = part_tokens.view(B_s, T_s * self.num_parts, C_s)
                kv = feats_ball_seq
                if self.ball_to_mid_proj is not None:
                    kv = self.ball_to_mid_proj(kv)
                if kv.shape[-1] != C_s and str(i) in self.fusion_bridge_proj:
                    kv = self.fusion_bridge_proj[str(i)](kv)
                pos_q_mid = torch.arange(
                    T_s, device=out_skel.device, dtype=feats_ball_seq.dtype
                )
                pos_q_mid = pos_q_mid.repeat_interleave(self.num_parts) * current_stride
                pos_kv_mid = ball_time_positions
                proximity_mask = None
                if (
                    self.fusion_local_window is not None
                    and kv.shape[1] > 0
                    and (q.shape[1] > 0)
                ):
                    dist = torch.abs(pos_q_mid[:, None] - pos_kv_mid[None, :])
                    proximity_mask = dist > float(self.fusion_local_window)
                q = self.mid_fusion_block(
                    x_q=q,
                    x_kv=kv,
                    kv_mask=attn_mask,
                    pos_q=pos_q_mid,
                    pos_kv=pos_kv_mid,
                    proximity_mask=proximity_mask,
                )
                q = self.fusion_bridge_drop(q)
                fused_parts = q.view(B_s, T_s, self.num_parts, C_s)
                skel_update = torch.zeros_like(out_skel)
                for idx, name in enumerate(self.part_names_sorted):
                    indices = self.body_parts[name]
                    part_context = fused_parts[:, :, idx, :].unsqueeze(2)
                    for j_idx in indices:
                        skel_update[:, :, j_idx, :] += part_context.squeeze(2)
                out_skel = out_skel + self.fusion_bridge_scale * skel_update
            out_skel = self.downsamples[i](out_skel)
            ds_stride = getattr(self.downsamples[i], "stride", 1)
            current_stride *= ds_stride
        skeleton_logits, skeleton_features = self.pose_head(out_skel)
        skel_tokens = out_skel.mean(dim=2)
        skel_tokens = self.skel_proj_seq(skel_tokens)
        skel_tokens = self.skel_token_dropout(skel_tokens)
        fused_seq = skel_tokens
        if self.use_late_fusion:
            for layer in self.fusion_attention_layers:
                _pos_q = pos_q
                _pos_kv = pos_kv
                if _pos_q is None:
                    _pos_q = (
                        torch.arange(
                            fused_seq.shape[1],
                            device=fused_seq.device,
                            dtype=feats_ball_seq.dtype,
                        )
                        * current_stride
                    )
                elif (
                    not torch.is_tensor(_pos_q)
                    or _pos_q.ndim != 1
                    or _pos_q.shape[0] != fused_seq.shape[1]
                ):
                    _pos_q = (
                        torch.arange(
                            fused_seq.shape[1],
                            device=fused_seq.device,
                            dtype=feats_ball_seq.dtype,
                        )
                        * current_stride
                    )
                else:
                    _pos_q = _pos_q.to(
                        device=fused_seq.device, dtype=feats_ball_seq.dtype
                    )
                if _pos_kv is None:
                    _pos_kv = ball_time_positions
                elif (
                    not torch.is_tensor(_pos_kv)
                    or _pos_kv.ndim != 1
                    or _pos_kv.shape[0] != feats_ball_seq.shape[1]
                ):
                    _pos_kv = ball_time_positions
                else:
                    _pos_kv = _pos_kv.to(
                        device=feats_ball_seq.device, dtype=feats_ball_seq.dtype
                    )
                proximity_mask = None
                if (
                    self.fusion_local_window is not None
                    and feats_ball_seq.shape[1] > 0
                    and (fused_seq.shape[1] > 0)
                ):
                    dist = torch.abs(_pos_q[:, None] - _pos_kv[None, :])
                    proximity_mask = dist > float(self.fusion_local_window)
                fused_seq = layer(
                    x_q=fused_seq,
                    x_kv=feats_ball_seq,
                    kv_mask=attn_mask,
                    pos_q=_pos_q,
                    pos_kv=_pos_kv,
                    proximity_mask=proximity_mask,
                )
        pool_logits = self.fusion_pool(fused_seq).squeeze(-1)
        pool_weights = torch.softmax(pool_logits, dim=1).unsqueeze(-1)
        fused_vector = torch.sum(fused_seq * pool_weights, dim=1)
        fused_vector_norm = self.norm_final(fused_vector)
        fused_logits = self.classifier_head(fused_vector_norm)
        logits = fused_logits + self.skeleton_skip_scale * skeleton_logits
        if return_features:
            return (logits, fused_vector_norm, skeleton_features, ball_preds)
        else:
            return (logits, ball_preds)

    def get_feature_dim(self):
        return getattr(self, "feat_dim", None)


def MambaSkeletonActionModel_(**kwargs):
    return MambaSkeletonActionModel(**kwargs)
