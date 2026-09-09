"""Launch a dataset-specific StrokeNet trainer with the released configuration."""
import argparse
import importlib
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(evaluation=False):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", required=True)
    cli.add_argument("--seed", type=int, default=1)
    cli.add_argument("--work-dir", required=True)
    cli.add_argument("--weights", required=evaluation)
    cli.add_argument(
        "--check-config",
        action="store_true",
        help="Validate imports and arguments without reading data or using a GPU.",
    )
    args, extra = cli.parse_known_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if config["feeder"] == "feeders.badmintondb.Feeder":
        module = "training.badmintondb"
    elif config["model"].startswith("models.strokenet_v1."):
        module = "training.fine_tennis10_v1"
    else:
        module = "training.fine_tennis10_v2"
    trainer = importlib.import_module(module)
    parser = trainer.get_parser()
    allowed = vars(parser.parse_args([]))
    unknown = set(config) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown trainer config keys: {sorted(unknown)}")
    config.update(
        seed=args.seed,
        work_dir=args.work_dir,
        visual_dir=str(Path(args.work_dir) / "visualizations"),
        phase="test" if evaluation else "train",
    )
    if args.weights:
        config["weights"] = args.weights
    if evaluation:
        config["save_score"] = True
        if module == "training.badmintondb":
            config["test_feeder_args"] = dict(config["test_feeder_args"], split="test")
    parser.set_defaults(**config)
    resolved = parser.parse_args(extra)
    for key in ("model", "feeder"):
        name, attr = getattr(resolved, key).rsplit(".", 1)
        getattr(importlib.import_module(name), attr)
    if args.check_config:
        print(
            f"Valid: {module}, phase={resolved.phase}, selection={resolved.best_metric}, epochs={resolved.num_epoch}, batch={resolved.batch_size}, seed={resolved.seed}"
        )
        return
    trainer.init_seed(resolved.seed)
    trainer.Processor(resolved).start()


if __name__ == "__main__":
    main()
