# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings # To warn user about M computation limitations

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        bimamba_type="none",
        save_params=False, # Keep the user's original flag
        cross_modal_dim: int = 0,  # NEW: 0 = disabled (backward-compatible)
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.save_params = save_params # Keep this flag

        # Cross-modal parameter modulation (MSFMamba-inspired)
        self.cross_modal_dim = cross_modal_dim
        if cross_modal_dim > 0:
            self.cross_B_proj = nn.Linear(cross_modal_dim, self.ngroups * self.d_state, bias=False)
            self.cross_C_proj = nn.Linear(cross_modal_dim, self.ngroups * self.d_state, bias=False)
            self.cross_dt_proj = nn.Linear(cross_modal_dim, self.nheads, bias=False)
            # Zero-init for stable training start (model starts as skeleton-only)
            nn.init.zeros_(self.cross_B_proj.weight)
            nn.init.zeros_(self.cross_C_proj.weight)
            nn.init.zeros_(self.cross_dt_proj.weight)

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)
        else:
            print("Not Using RMSNormGated")

        # Add backward parameters for bidirectionality (v2 approach)
        if bimamba_type == "v2":
            conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
            self.conv1d_b = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if conv_init is not None:
                nn.init.uniform_(self.conv1d_b.weight, -conv_init, conv_init)

            # Backward dt_bias, initialized like dt_bias
            dt_b = torch.exp(
                torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt_b = dt_b + torch.log(-torch.expm1(-dt_b))
            self.dt_bias_b = nn.Parameter(inv_dt_b)
            self.dt_bias_b._no_weight_decay = True

            # Backward A_log, initialized like A_log
            A_b = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            A_b_log = torch.log(A_b).to(dtype=dtype)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            # Backward D, initialized like D
            self.D_b = nn.Parameter(torch.ones(self.d_ssm if D_has_hdim else self.nheads, device=device))
            self.D_b._no_weight_decay = True


        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)
        # --- Initialize debug_M ---
        self.debug_M = None # Initialize attribute

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None,
                cross_modal=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        # --- Reset debug_M if not saving this time (Optional but good practice) ---
        if not self.save_params:
            self.debug_M = None

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        if self.bimamba_type == "v2":
            A_b = -torch.exp(self.A_b_log.float())  # (nheads) or (d_inner, d_state)

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and inference_params is None and cross_modal is None:
            if self.bimamba_type == "v2":
                print("Bidirectionnal mode")
                # Use attributes directly
                A_b = -torch.exp(self.A_b_log.float())

                out_f = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                    rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                    headdim=None if self.D_has_hdim else self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=self.norm_before_gate,
                    dt_limit=self.dt_limit,save_params=self.save_params,
                )

                # --- M Matrix computation Block (Corrected Call) ---
                params = None
                output_tensor = None
                if self.save_params:
                    if isinstance(out_f, tuple) and len(out_f) == 2:
                        output_tensor, params = out_f
                        if not isinstance(params, dict): params = None # Validate params
                    else:
                        warnings.warn("Scan func didn't return params dict when save_params=True.")
                        output_tensor = out_f
                else:
                    output_tensor = out_f

                if self.save_params and params is not None: # Only compute if flag is True and params exist
                    print(f"Computing M matrix for layer {self.layer_idx}...")
                    try:
                        # --- CORRECTED CALL to build function ---
                        M = build_semiseparable_matrix_avg_heads(
                            params=params,
                            nheads=self.nheads,
                            d_state=self.d_state,
                            ngroups=self.ngroups,
                            seq_len=seqlen, # Use seqlen determined at start
                            D_param=self.D, # Pass the layer's D parameter
                            D_has_hdim=self.D_has_hdim,
                            headdim=self.headdim,
                            # Assuming params['dt'] is final, dt_bias etc. not needed here
                            sample_idx=0,
                            device=u.device,
                            dtype=u.dtype,
                        )
                        if M is not None:
                            print("M matrix computation successful.")
                            self.debug_M = M.detach().cpu() # Store in debug_M
                        else:
                            print("Warning: M matrix computation failed.")
                            self.debug_M = None
                    except Exception as e:
                         print(f"ERROR during M matrix computation: {e}")
                         self.debug_M = None
                # --- End M Computation Block ---

                # Use the output tensor for the rest of the logic
                if output_tensor is None: raise RuntimeError("Forward output tensor is None")
                out_f = output_tensor

                out_b = mamba_split_conv1d_scan_combined(
                    zxbcdt.flip(1).contiguous(), # <--- Add .contiguous() here,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.dt_bias_b,
                    A_b,
                    rearrange(self.D_b, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D_b,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx.flip(1) if seq_idx is not None else None,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                    rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                    headdim=None if self.D_has_hdim else self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=self.norm_before_gate,
                    dt_limit=self.dt_limit,save_params=False,
                )

                out = out_f + out_b.flip(1)  # Flip backward output back to forward order
                out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

                if seqlen_og is not None:
                    out = rearrange(out, "b l d -> (b l) d")

                if self.process_group is not None:
                    reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                    out = reduce_fn(out, self.process_group)
            else:
                print("Unidirectionnal mode")
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                    rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                    outproj_weight=self.out_proj.weight,
                    outproj_bias=self.out_proj.bias,
                    headdim=None if self.D_has_hdim else self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=self.norm_before_gate,
                    **dt_limit_kwargs,
                )
                if seqlen_og is not None:
                    out = rearrange(out, "b l d -> (b l) d")
                if self.process_group is not None:
                    reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                    out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2).contiguous(),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            # --- Cross-modal parameter modulation (forward direction) ---
            if cross_modal is not None and self.cross_modal_dim > 0:
                B = B + self.cross_B_proj(cross_modal)
                C = C + self.cross_C_proj(cross_modal)
                dt = dt + self.cross_dt_proj(cross_modal)

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")

            # --- Bidirectional backward scan for non-fused path (v2) ---
            if self.bimamba_type == "v2":
                # Backward conv1d on flipped xBC
                xBC_b_input = torch.split(
                    zxbcdt,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )[3]  # extract xBC portion
                xBC_b_flipped = xBC_b_input.flip(1).contiguous()
                if causal_conv1d_fn is not None and self.activation in ["silu", "swish"]:
                    xBC_b = causal_conv1d_fn(
                        xBC_b_flipped.transpose(1, 2).contiguous(),
                        rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                        bias=self.conv1d_b.bias,
                        activation=self.activation,
                    ).transpose(1, 2)
                else:
                    xBC_b = self.act(
                        self.conv1d_b(xBC_b_flipped.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                    )
                x_b, B_b, C_b = torch.split(
                    xBC_b, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
                )
                # Extract backward dt from flipped in_proj
                dt_b = torch.split(
                    zxbcdt.flip(1).contiguous(),
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )[4].contiguous()  # dt portion

                # Cross-modal modulation on backward direction (flipped cross_modal)
                if cross_modal is not None and self.cross_modal_dim > 0:
                    cm_flip = cross_modal.flip(1)
                    B_b = B_b + self.cross_B_proj(cm_flip)
                    C_b = C_b + self.cross_C_proj(cm_flip)
                    dt_b = dt_b + self.cross_dt_proj(cm_flip)

                A_b = -torch.exp(self.A_b_log.float())
                y_b = mamba_chunk_scan_combined(
                    rearrange(x_b, "b l (h p) -> b l h p", p=self.headdim),
                    dt_b,
                    A_b,
                    rearrange(B_b, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C_b, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D_b, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D_b,
                    z=None,
                    dt_bias=self.dt_bias_b,
                    dt_softplus=True,
                    **dt_limit_kwargs,
                )
                y_b = rearrange(y_b, "b l h p -> b l (h p)")
                y = y + y_b.flip(1)
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    # Add this to your Mamba2 class (from response #18, #24)
    @property
    def kernel_matrix_m(self):
        """Returns the computed approximate M matrix stored in debug_M, otherwise None."""
        return getattr(self, 'debug_M', None)


# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modifications for M matrix computation using compute/store flags

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np # Keep if needed by build function
import warnings

# Assume necessary imports for causal_conv1d, mamba_ssm.ops, etc., are present
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
# ... other necessary imports ...
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
# ... etc ...


# ==============================================================================
# Simplified build_semiseparable_matrix_avg_heads function definition
# Assumes params['dt'] contains the FINAL delta values after bias/softplus/limit
# ==============================================================================
def build_semiseparable_matrix_avg_heads(
    params: dict,
    nheads: int,
    d_state: int,
    ngroups: int,
    seq_len: int,
    D_param: torch.Tensor | None, # Keep D_param to check if it exists, but won't use value
    D_has_hdim: bool,
    headdim: int,
    sample_idx: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """
    Constructs an averaged (L x L) matrix representing an *approximation*
    of the STATE-MEDIATED semiseparable matrix (M') magnitude across heads.
    This version EXCLUDES the D term contribution from the diagonal.

    Approximation: Computes M'_h[t, s] = H_h[t,s] * (avg_scalar_c_ht * avg_scalar_b_hs * delta_s),
                   where H is state transition, avg_scalar_c/b are derived from C/B tensors.

    Args:
        params: Dictionary from scan function ('A', 'B', 'C', 'dt').
        nheads (H), d_state (N), ngroups (G), seq_len (T), headdim (P): Dimensions.
        D_param: Layer's D parameter (Used only to check D_has_hdim if needed, value ignored).
        D_has_hdim: Boolean indicating D parameter shape.
        sample_idx: Batch sample index (requires batch=1).
        device: Torch device.
        dtype: Torch dtype for output matrix.

    Returns:
        M_prime_approx_avg: An averaged (L x L) matrix (torch.Tensor), or None if error.
    """
    # --- 1) Extract and Validate tensors ---
    required_keys = ["A", "B", "C", "dt"]
    if params is None or not all(key in params for key in required_keys):
        missing = [key for key in required_keys if key not in (params or {})]
        print(f"Error: Missing key(s) {missing} in params dictionary.")
        return None
    try:
        A_all = params["A"].detach().clone().to(device=device)
        B_all = params["B"].detach().clone().to(device=device)
        C_all = params["C"].detach().clone().to(device=device)
        dt_param_saved = params["dt"].detach().clone().to(device=device)
        if dt_param_saved.dim() == 4:
            _, _, n_chunks, chunk_size = dt_param_saved.shape
            dt_seq = rearrange(dt_param_saved, 'b h c l -> b (c l) h', c=n_chunks, l=chunk_size)
        elif dt_param_saved.dim() == 3: dt_seq = dt_param_saved
        else: print(f"Error: Unexpected dt shape: {dt_param_saved.shape}"); return None
        if B_all.shape[0] != 1: print(f"Error: Batch size != 1"); return None
    except Exception as e: print(f"Error processing params: {e}"); return None
    # --- 2) Determine length and trim ---
    actual_seq_len = B_all.shape[1]; max_len = min(seq_len, actual_seq_len)
    if max_len <= 0: print("Error: sequence length <= 0."); return None
    B_all, C_all, dt_seq = B_all[:,:max_len], C_all[:,:max_len], dt_seq[:,:max_len]
    # --- 3) Accumulate M'_h_approx ---
    M_accum = torch.zeros((max_len, max_len), device=device, dtype=torch.float32)
    actual_heads_computed = 0
    if ngroups <= 0 or nheads <= 0 or nheads % ngroups != 0: print(f"Error: Invalid ngroups/nheads"); return None
    groups_per_head_inv = ngroups / nheads
    for h in range(nheads):
        try:
            A_val = A_all[h].float(); delta_t_h = dt_seq[sample_idx, :, h].float()
            alpha = delta_t_h * A_val; group_idx = int(h * groups_per_head_inv)
            if group_idx >= B_all.shape[2]: print(f"Warn: Invalid group idx {group_idx}"); continue
            B_scalar_h = torch.abs(B_all[sample_idx, :, group_idx, :]).mean(dim=-1).to(torch.float32)
            C_scalar_h = torch.abs(C_all[sample_idx, :, group_idx, :]).mean(dim=-1).to(torch.float32)
            # D_h_scalar calculation removed/ignored

            alpha_cumsum = torch.cumsum(alpha, dim=0); alpha_cumsum_padded = torch.cat([torch.tensor([0.0],device=device,dtype=torch.float32), alpha_cumsum])
            log_H_matrix = alpha_cumsum_padded[1:][:, None] - alpha_cumsum_padded[:-1][None, :]; H_matrix = torch.exp(log_H_matrix); H_matrix = torch.tril(H_matrix, diagonal=-1)
            M_h_off_diag = C_scalar_h[:, None] * H_matrix * (delta_t_h[None, :] * B_scalar_h[None, :])
            # --- MODIFIED DIAGONAL ---
            # Diagonal term M'_ii = C_i * (delta_i * B_i) approx (NO D TERM)
            M_h_diag_vals = C_scalar_h * delta_t_h * B_scalar_h
            # -------------------------
            M_h = torch.tril(M_h_off_diag) + torch.diag(M_h_diag_vals)
            M_accum += M_h; actual_heads_computed += 1
        except Exception as e: print(f"Warning: Error processing head {h}. Error: {e}"); continue
    if actual_heads_computed == 0: print("Error: No heads processed."); return None
    M_prime_approx_avg = M_accum / actual_heads_computed
    return M_prime_approx_avg.to(dtype=dtype)
