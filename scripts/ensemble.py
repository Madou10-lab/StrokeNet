"""Combine raw joint/bone scores and report Fine-Tennis10 metrics."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import paper_metrics, is_better


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint", required=True)
    p.add_argument("--bone", required=True)
    p.add_argument("--annotation", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--joint-weight", type=float)
    mode.add_argument(
        "--search",
        action="store_true",
        help="Original 0.001-grid search using the supplied labels; Top-1, then Macro-F1, then Top-5 tie breaking.",
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with open(a.joint, "rb") as f:
        joint = pickle.load(f)
    with open(a.bone, "rb") as f:
        bone = pickle.load(f)
    labels = {}
    for line in Path(a.annotation).read_text().splitlines():
        if line.strip():
            name, label = line.rsplit(maxsplit=1)
            name = Path(name).stem
            if name in labels:
                raise ValueError(f"Duplicate sample: {name}")
            labels[name] = int(label)
    if set(labels) != set(joint) or set(labels) != set(bone):
        raise ValueError("Joint, bone and annotation sample keys must match exactly.")
    names = sorted(labels)
    j = np.stack([np.asarray(joint[n], dtype=np.float64) for n in names])
    b = np.stack([np.asarray(bone[n], dtype=np.float64) for n in names])
    y = np.asarray([labels[n] for n in names])
    if (
        j.shape != b.shape
        or j.ndim != 2
        or not np.isfinite(j).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Expected matching finite (clips, classes) score arrays.")
    if not len(y) or y.min() < 0 or y.max() >= j.shape[1]:
        raise ValueError("Invalid labels.")
    if a.joint_weight is not None and not 0 <= a.joint_weight <= 1:
        p.error("joint-weight must lie in [0,1]")
    weights = np.arange(0.0, 1.0001, 0.001) if a.search else [a.joint_weight]
    best = None
    for w in weights:
        metrics = paper_metrics(w * j + (1 - w) * b, y)
        if best is None or is_better(metrics, best):
            best, best_w = metrics, float(w)
    result = {
        "joint_weight": best_w,
        "selection": "label-based grid search" if a.search else "fixed weight",
        "num_clips": len(y),
        "metrics": {
            k: best[k] for k in ("top1", "macro_acc", "macro_f1", "top2", "top5")
        },
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
