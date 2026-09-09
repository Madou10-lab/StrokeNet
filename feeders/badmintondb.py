import os
from typing import Dict, List
import numpy as np
from feeders.two_player import Feeder as _ShuttleSetM2Feeder

CLASS_LIST_18: List[str] = [
    "Bottom-Block",
    "Bottom-Clear",
    "Bottom-Drive",
    "Bottom-Dropshot",
    "Bottom-Net-Kill",
    "Bottom-Net-Lift",
    "Bottom-Net-Shot",
    "Bottom-Serve",
    "Bottom-Smash",
    "Top-Block",
    "Top-Clear",
    "Top-Drive",
    "Top-Dropshot",
    "Top-Net-Kill",
    "Top-Net-Lift",
    "Top-Net-Shot",
    "Top-Serve",
    "Top-Smash",
]
assert len(CLASS_LIST_18) == 18
CLASS_TO_IDX_18: Dict[str, int] = {c: i for i, c in enumerate(CLASS_LIST_18)}
CLASS_TO_IDX_18["Bottom-Flick-Serve"] = CLASS_TO_IDX_18["Bottom-Serve"]
CLASS_TO_IDX_18["Top-Flick-Serve"] = CLASS_TO_IDX_18["Top-Serve"]
assert len(CLASS_TO_IDX_18) == 20 and len(set(CLASS_TO_IDX_18.values())) == 18


class Feeder(_ShuttleSetM2Feeder):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        data_type: str = "jb",
        window_size: int = 72,
        balance_oversample: bool = False,
        **kwargs,
    ):
        self.balance_oversample = bool(balance_oversample)
        super().__init__(
            data_path=data_path,
            split=split,
            data_type=data_type,
            window_size=window_size,
            **kwargs,
        )

    def _scan_clips(self):
        split_dir = os.path.join(self.data_path, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.entries: List[Dict] = []
        class_dirs = sorted(os.listdir(split_dir))
        for cls_name in class_dirs:
            if cls_name not in CLASS_TO_IDX_18:
                continue
            cls_idx = CLASS_TO_IDX_18[cls_name]
            cls_dir = os.path.join(split_dir, cls_name)
            stems = sorted(
                {
                    fn.rsplit("_", 1)[0]
                    for fn in os.listdir(cls_dir)
                    if fn.endswith("_joints.npy")
                }
            )
            for stem in stems:
                self.entries.append(
                    {
                        "stem": stem,
                        "class_name": cls_name,
                        "label": cls_idx,
                        "cls_dir": cls_dir,
                    }
                )
        if self.split == "train" and self.balance_oversample:
            self._oversample_entries_balanced()
        self.label = np.array([e["label"] for e in self.entries], dtype=np.int64)
        self.sample_name = [e["stem"] for e in self.entries]
        self.name_to_index = {name: i for i, name in enumerate(self.sample_name)}
        if self.debug:
            n_cls = len(set((e["label"] for e in self.entries)))
            print(
                f"[Feeder BadmintonDB_M2/{self.split}] {len(self.entries)} clips, {n_cls} classes, oversample={self.balance_oversample}"
            )

    def _oversample_entries_balanced(self):
        by_label: Dict[int, List[Dict]] = {}
        for e in self.entries:
            by_label.setdefault(e["label"], []).append(e)
        max_n = max((len(v) for v in by_label.values()))
        balanced: List[Dict] = []
        for label in sorted(by_label.keys()):
            group = by_label[label]
            n = len(group)
            magni = max_n // n
            remain = max_n % n
            balanced.extend(group * magni)
            if remain:
                balanced.extend(group[:remain])
        self.entries = balanced
