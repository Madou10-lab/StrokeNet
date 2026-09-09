"""Evaluate a saved StrokeNet checkpoint and export per-clip scores."""
from train import main

if __name__ == "__main__":
    main(evaluation=True)
