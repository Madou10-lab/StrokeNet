"""Synthetic loader, metric, loss and native CUDA forward checks."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import yaml
from evaluation.metrics import paper_metrics, is_better

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_metrics_include_absent_classes(self):
        scores = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
        m = paper_metrics(scores, np.array([0, 1]))
        self.assertEqual(m["top1"], 0.5)
        self.assertAlmostEqual(m["macro_acc"], 1 / 3)
        self.assertAlmostEqual(m["macro_f1"], 2 / 9)
        self.assertEqual(m["top2"], 1.0)
        self.assertFalse(is_better(m, m))

    def test_synthetic_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ball = root / "ball"
            ball.mkdir()
            split = root / "test"
            split.mkdir()
            rng = np.random.default_rng(0)
            (split / "example.json").write_text(
                json.dumps(
                    {
                        "file_name": "example",
                        "skeletons": rng.normal(size=(80, 17, 3)).tolist(),
                    }
                )
            )
            (split / "annotation.txt").write_text("example.json 0\n")
            (root / "labels.txt").write_text("Synthetic 0\n")
            (ball / "example_ball.csv").write_text(
                "Frame,Visibility,X,Y\n"
                + "\n".join(f"{i},1,{100+i},{200+i}" for i in range(80))
            )
            for version in ("v1", "v2"):
                c = yaml.safe_load(
                    (ROOT / f"configs/fine_tennis10/{version}_joint.yaml").read_text()
                )
                args = dict(
                    c["test_feeder_args"],
                    data_path=str(split),
                    ball_data_path=str(ball),
                    label_file=str(root / "labels.txt"),
                )
                mod = importlib.import_module("feeders.fine_tennis10_" + version)
                for stream in ("j", "b"):
                    f = mod.Feeder(**dict(args, data_type=stream))
                    sample = f[0]
                    self.assertEqual(sample[0].shape, (3, 64, 17, 1))
                    self.assertEqual(sample[1].shape, (7, 64))
                    self.assertTrue(np.isfinite(sample[0]).all())
            cls = root / "val" / "Bottom-Serve"
            cls.mkdir(parents=True)
            np.save(
                cls / "sample_joints.npy",
                rng.normal(size=(50, 2, 17, 2)).astype("float32"),
            )
            np.save(cls / "sample_shuttle.npy", np.full((50, 2), 0.5, dtype="float32"))
            from feeders.badmintondb import Feeder

            sample = Feeder(
                data_path=str(root), split="val", data_type="jb", window_size=72
            )[0]
            self.assertEqual(sample[0].shape, (4, 72, 17, 2))
            self.assertEqual(sample[1].shape, (7, 72))
            self.assertEqual(int(sample[3]), 7)

    def test_training_losses(self):
        for v in ("v1", "v2"):
            mod = importlib.import_module("training.fine_tennis10_" + v)
            logits = torch.randn(4, 10, requires_grad=True)
            target = torch.tensor([0, 1, 1, 9])
            loss = mod.LabelSmoothingCrossEntropy()(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "Native Mamba kernels require CUDA")
    def test_cuda_forward_and_backward(self):
        for relative in (
            "fine_tennis10/v1_joint.yaml",
            "fine_tennis10/v2_joint.yaml",
            "badmintondb/v1.yaml",
            "badmintondb/v2.yaml",
        ):
            with self.subTest(config=relative):
                c = yaml.safe_load((ROOT / "configs" / relative).read_text())
                name, attr = c["model"].rsplit(".", 1)
                torch.manual_seed(1)
                model = (
                    getattr(importlib.import_module(name), attr)(**c["model_args"])
                    .cuda()
                    .train()
                )
                channels = c["model_args"]["in_channels"]
                players = 2 if relative.startswith("badminton") else 1
                # Small temporal window exercises the same native layers without full training.
                x = torch.randn(2, channels, 32, 17, players, device="cuda")
                b = torch.randn(2, 7, 32, device="cuda")
                mask = torch.ones(2, 32, device="cuda")
                b[:, -1] = mask
                result = model(x, b, mask)
                logits = result[0] if isinstance(result, tuple) else result
                self.assertEqual(
                    tuple(logits.shape), (2, c["model_args"]["num_classes"])
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([0, 1], device="cuda")
                )
                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and torch.is_tensor(result[1])
                ):
                    loss = loss + 0.1 * result[1].square().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                del model, x, b, result, logits, loss, grads
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
