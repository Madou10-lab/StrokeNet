"""Report mean and sample standard deviation across per-seed metric JSON files."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.metrics) < 2:
        parser.error("At least two independent seed results are required.")
    records = [json.loads(path.read_text())["metrics"] for path in args.metrics]
    for key in ("top1", "macro_acc", "macro_f1", "top2"):
        values = 100 * np.asarray([record[key] for record in records], dtype=float)
        print(f"{key}: {values.mean():.2f} ± {values.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
