import torch.nn as nn
from models.layers.conditioned_backbone import ConditionedBackbone
from models.layers.graph_utils_h36m17 import (
    GraphConvLayer,
    GraphDistanceEmbedding,
    KendallTransformLayer,
)


class GCN_Pose_Encoder_H36M17(nn.Module):
    def __init__(self, in_channels=3, embed_dim=192, num_joints=17):
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


class H36MBackbone(ConditionedBackbone):
    def __init__(
        self,
        in_channels=3,
        embed_dim=192,
        num_joints=17,
        kendall_transform_type="none",
        window_size=64,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_joints=25,
            kendall_transform_type=kendall_transform_type,
            window_size=window_size,
            **kwargs
        )
        self.num_joints = num_joints
        if kendall_transform_type in ("rigid", "nonrigid"):
            self.kendall_transform = KendallTransformLayer(
                transform_type=kendall_transform_type,
                num_frames=window_size,
                num_joints=num_joints,
            )
        else:
            self.kendall_transform = None
        self.pose_encoder = GCN_Pose_Encoder_H36M17(
            in_channels=in_channels, embed_dim=embed_dim, num_joints=num_joints
        )
        self.graph_dist_embed = GraphDistanceEmbedding(
            embed_dim=embed_dim, max_dist=10, root_joint=7, num_joints=num_joints
        )


def H36MBackbone_(**kwargs):
    return H36MBackbone(**kwargs)
