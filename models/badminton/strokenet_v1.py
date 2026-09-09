import torch
import torch.nn as nn
from models.layers.attention_backbone import MambaSkeletonActionModel

_COCO17_BODY_PARTS = {
    "torso": [0, 1, 2, 3, 4],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
}


class MambaSkeletonActionModelCOCO17(MambaSkeletonActionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body_parts = {k: list(v) for k, v in _COCO17_BODY_PARTS.items()}
        self.num_parts = len(self.body_parts)
        self.part_names_sorted = sorted(self.body_parts.keys())


class StrokeNetV1_M2(nn.Module):
    def __init__(self, num_joints=17, use_player_id_embedding=True, **kwargs):
        super().__init__()
        self.in_channels = int(kwargs.get("in_channels", 2))
        self.use_player_id_embedding = bool(use_player_id_embedding)
        self.num_players = 2
        self.inner = MambaSkeletonActionModelCOCO17(**kwargs)
        self.ball_input_dim = self.inner.ball_input_dim
        self.cross_modal_dim = getattr(self.inner, "cross_modal_dim", None)
        self.feat_dim = self.inner.feat_dim
        if self.use_player_id_embedding:
            self.player_id_emb = nn.Embedding(self.num_players, self.in_channels)
            nn.init.zeros_(self.player_id_emb.weight)
        else:
            self.player_id_emb = None

    def get_feature_dim(self):
        return self.feat_dim

    def _expand_along_M(self, x_ball, x_ball_mask, pos_q, pos_kv, B, M):
        if x_ball is not None:
            xb = x_ball.unsqueeze(1).expand(-1, M, -1, -1).contiguous()
            xb = xb.reshape(B * M, *x_ball.shape[1:])
        else:
            xb = None
        if x_ball_mask is not None:
            xm = x_ball_mask.unsqueeze(1).expand(-1, M, -1).contiguous()
            xm = xm.reshape(B * M, x_ball_mask.shape[-1])
        else:
            xm = None
        if pos_q is not None:
            pq = pos_q.unsqueeze(1).expand(-1, M, -1).contiguous().reshape(B * M, -1)
        else:
            pq = None
        if pos_kv is not None:
            pk = pos_kv.unsqueeze(1).expand(-1, M, -1).contiguous().reshape(B * M, -1)
        else:
            pk = None
        return (xb, xm, pq, pk)

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
            raise ValueError(
                f"Expected x_skel with 4 or 5 dims, got shape {tuple(x_skel.shape)}"
            )
        x_skel_per = (
            x_skel_5d.permute(0, 4, 1, 2, 3).contiguous().reshape(B * M, C, T, V, 1)
        )
        if self.player_id_emb is not None and M == self.num_players:
            ids = torch.arange(M, device=x_skel.device)
            ids = ids.unsqueeze(0).expand(B, -1).reshape(-1)
            id_vec = self.player_id_emb(ids)
            x_skel_per = x_skel_per + id_vec.view(B * M, C, 1, 1, 1)
        xb, xm, pq, pk = self._expand_along_M(x_ball, x_ball_mask, pos_q, pos_kv, B, M)
        out = self.inner(
            x_skel_per,
            xb,
            x_ball_mask=xm,
            pos_q=pq,
            pos_kv=pk,
            return_features=return_features,
        )
        if return_features:
            logits, feats, feats2, ball_preds = out
            logits = logits.view(B, M, -1).mean(dim=1)
            feats = feats.view(B, M, -1).mean(dim=1)
            feats2 = feats2.view(B, M, -1).mean(dim=1)
            ball_preds = ball_preds.view(B, M, *ball_preds.shape[1:])[:, 0].contiguous()
            return (logits, feats, feats2, ball_preds)
        else:
            logits, ball_preds = out
            logits = logits.view(B, M, -1).mean(dim=1)
            ball_preds = ball_preds.view(B, M, *ball_preds.shape[1:])[:, 0].contiguous()
            return (logits, ball_preds)


def StrokeNetV1_M2_(**kwargs):
    return StrokeNetV1_M2(**kwargs)
