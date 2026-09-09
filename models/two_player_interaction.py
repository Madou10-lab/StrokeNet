"""Shared player encoders and role-aware interaction for two-player stroke recognition."""
import torch
import torch.nn as nn


class TwoPlayerInteractionHead(nn.Module):
    def __init__(
        self,
        feat_dim,
        ball_in_dim,
        num_classes,
        d_inter=256,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2.0,
        drop=0.1,
        use_ball_token=True,
        use_cls=True,
        mode="attn",
    ):
        super().__init__()
        assert mode in ("attn", "concat"), "mode must be 'attn' (C2) or 'concat' (C1)"
        self.mode = mode
        self.use_ball_token = bool(use_ball_token)
        self.use_cls = bool(use_cls) and mode == "attn"
        self.d_inter = int(d_inter)
        self.ball_in_dim = int(ball_in_dim)
        self.player_proj = nn.Linear(feat_dim, d_inter)
        self.role_emb = nn.Embedding(4, d_inter)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        if self.use_ball_token:
            self.ball_enc = nn.Sequential(
                nn.Conv1d(ball_in_dim, d_inter, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(d_inter, d_inter, kernel_size=3, padding=1),
            )
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_inter))
            nn.init.normal_(self.cls_token, std=0.02)
        if mode == "attn":
            layer = nn.TransformerEncoderLayer(
                d_model=d_inter,
                nhead=n_heads,
                dim_feedforward=int(d_inter * mlp_ratio),
                dropout=drop,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.encoder = None
        self.out_dim = 2 * d_inter
        self.norm = nn.LayerNorm(self.out_dim)
        self.head_drop = nn.Dropout(drop)
        self.classifier = nn.Linear(self.out_dim, num_classes)

    def _ball_token(self, x_ball, x_ball_mask):
        if x_ball.shape[1] != self.ball_in_dim and x_ball.shape[-1] == self.ball_in_dim:
            x_ball = x_ball.transpose(1, 2).contiguous()
        bt = self.ball_enc(x_ball)
        if x_ball_mask is not None:
            w = x_ball_mask.to(bt.dtype).unsqueeze(1)
            bt = (bt * w).sum(-1) / w.sum(-1).clamp(min=1e-06)
        else:
            bt = bt.mean(-1)
        return bt

    def forward(self, feats_pM, x_ball=None, x_ball_mask=None):
        B, M, _ = feats_pM.shape
        assert M == 2, f"TwoPlayerInteractionHead expects M=2, got {M}"
        dev = feats_pM.device
        p = self.player_proj(feats_pM)
        p = p + self.role_emb(torch.tensor([0, 1], device=dev)).unsqueeze(0)
        tokens = [p]
        if self.use_ball_token and x_ball is not None:
            bt = self._ball_token(x_ball, x_ball_mask)
            bt = bt + self.role_emb(torch.tensor(2, device=dev))
            tokens.append(bt.unsqueeze(1))
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1) + self.role_emb(
                torch.tensor(3, device=dev)
            ).view(1, 1, -1)
            tokens = [cls] + tokens
        seq = torch.cat(tokens, dim=1)
        if self.encoder is not None:
            seq = self.encoder(seq)
        offset = 1 if self.use_cls else 0
        top = seq[:, offset + 0]
        bot = seq[:, offset + 1]
        feat = self.norm(torch.cat([top, bot], dim=-1))
        logits = self.classifier(self.head_drop(feat))
        return (logits, feat)


class TwoPlayerInteractWrapper(nn.Module):
    def __init__(
        self,
        inner,
        in_channels,
        num_classes,
        use_player_id_embedding=True,
        head_kwargs=None,
    ):
        super().__init__()
        self.inner = inner
        self.in_channels = int(in_channels)
        self.num_players = 2
        self.use_player_id_embedding = bool(use_player_id_embedding)
        self.ball_input_dim = inner.ball_input_dim
        self.cross_modal_dim = getattr(inner, "cross_modal_dim", None)
        head_kwargs = dict(head_kwargs or {})
        self.head = TwoPlayerInteractionHead(
            feat_dim=inner.feat_dim,
            ball_in_dim=self.ball_input_dim,
            num_classes=num_classes,
            **head_kwargs,
        )
        self.feat_dim = self.head.out_dim
        if self.use_player_id_embedding:
            self.player_id_emb = nn.Embedding(self.num_players, self.in_channels)
            nn.init.zeros_(self.player_id_emb.weight)
        else:
            self.player_id_emb = None

    def get_feature_dim(self):
        return self.feat_dim

    @staticmethod
    def _expand_along_M(x_ball, x_ball_mask, pos_q, pos_kv, B, M):
        def exp3(t):
            return (
                None
                if t is None
                else t.unsqueeze(1)
                .expand(-1, M, *[-1] * (t.dim() - 1))
                .reshape(B * M, *t.shape[1:])
            )

        def exp2(t):
            return (
                None
                if t is None
                else t.unsqueeze(1).expand(-1, M, -1).contiguous().reshape(B * M, -1)
            )

        return (exp3(x_ball), exp2(x_ball_mask), exp2(pos_q), exp2(pos_kv))

    def forward(
        self,
        x_skel,
        x_ball,
        x_ball_mask=None,
        pos_q=None,
        pos_kv=None,
        return_features=False,
    ):
        if x_skel.dim() == 4:
            B, C, T, V = x_skel.shape
            M = 1
            x_skel_5d = x_skel.unsqueeze(-1)
        elif x_skel.dim() == 5:
            B, C, T, V, M = x_skel.shape
            x_skel_5d = x_skel
        else:
            raise ValueError(f"Expected x_skel 4D/5D, got {tuple(x_skel.shape)}")
        x_skel_per = (
            x_skel_5d.permute(0, 4, 1, 2, 3).contiguous().reshape(B * M, C, T, V, 1)
        )
        if self.player_id_emb is not None and M == self.num_players:
            ids = (
                torch.arange(M, device=x_skel.device)
                .unsqueeze(0)
                .expand(B, -1)
                .reshape(-1)
            )
            id_vec = self.player_id_emb(ids)
            x_skel_per = x_skel_per + id_vec.view(B * M, C, 1, 1, 1)
        xb, xm, pq, pk = self._expand_along_M(x_ball, x_ball_mask, pos_q, pos_kv, B, M)
        logits_i, feats, feats2, ball_preds = self.inner(
            x_skel_per, xb, x_ball_mask=xm, pos_q=pq, pos_kv=pk, return_features=True
        )
        feats = feats.view(B, M, -1)
        ball_preds = ball_preds.view(B, M, *ball_preds.shape[1:])[:, 0].contiguous()
        if M != self.num_players:
            feats = feats.expand(-1, self.num_players, -1)
        logits, fused = self.head(feats, x_ball, x_ball_mask)
        if return_features:
            return (logits, fused, fused, ball_preds)
        return (logits, ball_preds)
