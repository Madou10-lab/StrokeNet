"""StrokeNet-v1 with H36M-17 anatomical groups for pose–ball cross-attention."""
from models.layers.attention_backbone import MambaSkeletonActionModel


class MambaSkeletonActionModelH36M17(MambaSkeletonActionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.body_parts = {
            "torso": [0, 7, 8, 9, 10],
            "right_arm": [14, 15, 16],
            "left_arm": [11, 12, 13],
            "right_leg": [1, 2, 3],
            "left_leg": [4, 5, 6],
        }
        self.num_parts = len(self.body_parts)
        self.part_names_sorted = sorted(self.body_parts.keys())


def MambaSkeletonActionModel_H36M17_(**kwargs):
    return MambaSkeletonActionModelH36M17(**kwargs)
