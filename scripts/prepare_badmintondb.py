"""Validate the uncollated BST BadmintonDB arrays before training.

Obtain dataset_npy from the official BST repository. Temporal sampling,
18-class mapping, bone construction and trajectory descriptors are performed
by the released feeder; this check does not rewrite or re-split the data.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feeders.badmintondb import CLASS_TO_IDX_18


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--check-counts", action="store_true")
    a = p.parse_args()
    for split, expected in [("train", 6120), ("val", 757), ("test", 781)]:
        root = a.data_root / split
        if not root.is_dir():
            raise FileNotFoundError(root)
        count = 0
        for cls in root.iterdir():
            if not cls.is_dir():
                continue
            if cls.name not in CLASS_TO_IDX_18:
                raise ValueError(f"Unknown class directory: {cls.name}")
            for joint in sorted(cls.glob("*_joints.npy")):
                shuttle = joint.with_name(
                    joint.name.removesuffix("_joints.npy") + "_shuttle.npy"
                )
                j, b = np.load(joint, allow_pickle=False), np.load(
                    shuttle, allow_pickle=False
                )
                if (
                    j.ndim != 4
                    or j.shape[1:] != (2, 17, 2)
                    or b.shape != (j.shape[0], 2)
                    or not len(j)
                ):
                    raise ValueError(
                        f"Invalid pair: {joint.name}, shapes={j.shape}, {b.shape}"
                    )
                if not np.isfinite(j).all() or not np.isfinite(b).all():
                    raise ValueError(f"Non-finite coordinates: {joint.name}")
                count += 1
        print(f"{split}: {count} clips (protocol: {expected})")
        if a.check_counts and count != expected:
            raise ValueError(f"{split}: count differs from protocol")


if __name__ == "__main__":
    main()
