from __future__ import print_function
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.manifold import TSNE
from torchlight import DictAction
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from timm.scheduler.cosine_lr import CosineLRScheduler
import seaborn as sns
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
try:
    import umap
except ImportError:
    umap = None
import torch


def add_legend_with_label_names(scatter, labels, label_mapping=None):
    cmap = scatter.get_cmap()
    norm = plt.Normalize(vmin=np.min(labels), vmax=np.max(labels))
    unique_labels = np.unique(labels)
    legend_patches = []
    for ul in unique_labels:
        color = cmap(norm(ul))
        if label_mapping is not None:
            label_name = label_mapping.get(ul, str(ul))
        else:
            label_name = str(ul)
        patch = mpatches.Patch(color=color, label=label_name)
        legend_patches.append(patch)
    plt.legend(handles=legend_patches, title="Classes")


def visualize_features_tsne(
    features,
    labels,
    output_path="tsne_batch.pdf",
    random_state=42,
    label_mapping=None,
    dpi=150,
    max_points=20000,
    rasterize_threshold=5000,
    title=None,
):
    print("Running t-SNE on features...")
    tsne = TSNE(n_components=2, random_state=random_state)
    features_2d = tsne.fit_transform(features)
    n_samples = features_2d.shape[0]
    if n_samples > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, max_points, replace=False)
        features_2d = features_2d[idx]
        labels_plot = np.array(labels)[idx]
        print(f"Subsampled {n_samples} -> {max_points} points for plotting.")
    else:
        labels_plot = labels
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=labels_plot,
        cmap="viridis",
        alpha=0.8,
        s=25,
        edgecolors="none",
        linewidths=0,
    )
    title_fs = 18
    axis_label_fs = 14
    tick_fs = 12
    legend_fs = 12
    legend_title_fs = 14
    if title is None:
        title = "t-SNE visualization"
    ax.set_title(title, fontsize=title_fs, pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=axis_label_fs)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=axis_label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    add_legend_with_label_names(scatter, labels_plot, label_mapping)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_fs)
        if legend.get_title() is not None:
            legend.get_title().set_fontsize(legend_title_fs)
    plt.tight_layout()
    ext = os.path.splitext(output_path)[1].lower()
    pil_kwargs = {}
    if ext == ".png":
        pil_kwargs = {"optimize": True, "compress_level": 6}
    elif ext in (".jpg", ".jpeg"):
        pil_kwargs = {"quality": 85, "optimize": True}
    elif ext in (".pdf", ".svg"):
        if features_2d.shape[0] > rasterize_threshold:
            scatter.set_rasterized(True)
    try:
        if pil_kwargs:
            plt.savefig(
                output_path, dpi=dpi, bbox_inches="tight", pil_kwargs=pil_kwargs
            )
        else:
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    except TypeError:
        if ext in (".jpg", ".jpeg"):
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight", quality=85)
        else:
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to {output_path}")


def visualize_features_umap(
    features,
    labels,
    output_path="umap_batch.png",
    random_state=42,
    label_mapping=None,
    title=None,
):
    if umap is None:
        print(
            "UMAP is not installed. Please install it using 'pip install umap-learn'."
        )
        return
    print("Running UMAP on features...")
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    features_2d = reducer.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    if title is None:
        title = "UMAP visualization"
    plt.title(title)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    add_legend_with_label_names(scatter, labels, label_mapping)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"UMAP plot saved to {output_path}")


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition(".")
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            "Class %s cannot be found (%s)"
            % (class_str, traceback.format_exception(*sys.exc_info()))
        )


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight: torch.Tensor | None = None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        if self.weight is not None:
            w = self.weight[target]
            loss = loss * w
        return loss.mean()


class ModelEMA:
    def __init__(self, model, decay=0.999, device=None, include_buffers=True):
        import torch.nn as nn

        self.decay = float(decay)
        self.include_buffers = include_buffers
        self.shadow = {}
        self.backup = {}
        m = model.module if isinstance(model, nn.DataParallel) else model
        dev = device
        for name, p in m.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (
                    p.detach().data.clone().to(dev)
                    if dev is not None
                    else p.detach().data.clone()
                )
        if include_buffers:
            for name, b in m.named_buffers():
                self.shadow[name] = (
                    b.detach().data.clone().to(dev)
                    if dev is not None
                    else b.detach().data.clone()
                )

    def update(self, model):
        import torch.nn as nn

        m = model.module if isinstance(model, nn.DataParallel) else model
        d = self.decay
        for name, p in m.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach().data, alpha=1.0 - d)
        if self.include_buffers:
            for name, b in m.named_buffers():
                if name in self.shadow:
                    self.shadow[name].copy_(b.detach().data)

    def store(self, model):
        import torch.nn as nn

        m = model.module if isinstance(model, nn.DataParallel) else model
        self.backup = {}
        for name, p in m.named_parameters():
            self.backup[name] = p.detach().data.clone()
        if self.include_buffers:
            for name, b in m.named_buffers():
                self.backup[name] = b.detach().data.clone()

    def restore(self, model):
        import torch.nn as nn

        m = model.module if isinstance(model, nn.DataParallel) else model
        for name, p in m.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        if self.include_buffers:
            for name, b in m.named_buffers():
                if name in self.backup:
                    b.data.copy_(self.backup[name])
        self.backup = {}

    def copy_to(self, model):
        import torch.nn as nn

        m = model.module if isinstance(model, nn.DataParallel) else model
        for name, p in m.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])
        if self.include_buffers:
            for name, b in m.named_buffers():
                if name in self.shadow:
                    b.data.copy_(self.shadow[name])


class CenterLossEMA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        alpha: float = 0.3,
        normalize_feats: bool = False,
        device: str | int | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.alpha = float(alpha)
        self.normalize_feats = bool(normalize_feats)
        centers = torch.zeros(self.num_classes, self.feat_dim, dtype=torch.float32)
        if device is not None:
            centers = centers.cuda(device)
        self.register_buffer("centers", centers)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.normalize_feats:
            feats = torch.nn.functional.normalize(feats, dim=1)
        centers_batch = self.centers[labels]
        diff = feats - centers_batch
        loss = 0.5 * diff.pow(2).sum(dim=1).mean()
        with torch.no_grad():
            unique_labels = labels.unique()
            for c in unique_labels:
                idx = labels == c
                if idx.any():
                    f_mean = feats[idx].mean(dim=0)
                    self.centers[c] = (1.0 - self.alpha) * self.centers[
                        c
                    ] + self.alpha * f_mean
        return loss


NTU60_CLASS_MAP = {
    0: "drink water",
    1: "eat meal",
    2: "brush teeth",
    3: "brush hair",
    4: "drop",
    5: "pick up",
    6: "throw",
    7: "sit down",
    8: "stand up",
    9: "clapping",
    10: "reading",
    11: "writing",
    12: "tear up paper",
    13: "put on jacket",
    14: "take off jacket",
    15: "put on a shoe",
    16: "take off a shoe",
    17: "put on glasses",
    18: "take off glasses",
    19: "put on a hat/cap",
    20: "take off a hat/cap",
    21: "cheer up",
    22: "hand waving",
    23: "kicking something",
    24: "reach into pocket",
    25: "hopping",
    26: "jump up",
    27: "phone call",
    28: "play with phone/tablet",
    29: "type on a keyboard",
    30: "point to something",
    31: "taking a selfie",
    32: "check time (from watch)",
    33: "rub two hands",
    34: "nod head/bow",
    35: "shake head",
    36: "wipe face",
    37: "salute",
    38: "put palms together",
    39: "cross hands in front",
    40: "sneeze/cough",
    41: "staggering",
    42: "falling down",
    43: "headache",
    44: "chest pain",
    45: "back pain",
    46: "neck pain",
    47: "nausea/vomiting",
    48: "fan self",
    49: "punching/slapping",
    50: "kicking",
    51: "pushing",
    52: "pat on back",
    53: "point finger",
    54: "hugging",
    55: "giving object",
    56: "touch pocket",
    57: "shaking hands",
    58: "walking towards",
    59: "walking apart",
}
NWUCLA_CLASSES = {
    0: "pick_up_with_one_hand",
    1: "pick_up_with_two_hands",
    2: "drop_trash",
    3: "walk_around",
    4: "sit_down",
    5: "stand_up",
    6: "donning",
    7: "doffing",
    8: "throw",
    9: "carry",
}
THETIS_CLASSES = {
    0: "backhand",
    1: "backhand2h",
    2: "bslice",
    3: "bvolley",
    4: "serflat",
    5: "foreflat",
    6: "foreopen",
    7: "fslice",
    8: "fvolley",
    9: "serkick",
    10: "serslice",
    11: "smash",
}
FFT_CLASSES = {0: "Service", 1: "Coup_droit", 2: "Revers", 3: "Reception"}
FFT_10_CLASSES = {
    0: "Service",
    1: "Coup_droit",
    2: "Revers",
    3: "Amortie",
    4: "Demi-Volée",
    5: "Lob",
    6: "Revers_amortie",
    7: "Revers_volée",
    8: "Smash",
    9: "Volée",
}
TENNIS7_CLASSES = {
    0: "backhand_slice",
    1: "backhand_topspin",
    2: "backhand_volley",
    3: "forehand_slice",
    4: "forehand_topspin",
    5: "forehand_volley",
    6: "overhead",
}
DATASET_LABEL_MAPS = {
    "ntu": NTU60_CLASS_MAP,
    "nw_ucla": NWUCLA_CLASSES,
    "thetis": THETIS_CLASSES,
    "fft": FFT_CLASSES,
    "tennis7": TENNIS7_CLASSES,
    "fft10": FFT_10_CLASSES,
}


def get_parser():
    parser = argparse.ArgumentParser(description="Mamba based action recognition")
    parser.add_argument(
        "--class_weight_beta",
        type=float,
        default=0.999,
        help="effective-number class weight beta",
    )
    parser.add_argument(
        "--work-dir",
        default="./work_dirs/run",
        help="the work folder for storing results",
    )
    parser.add_argument(
        "--visual-dir",
        default="./work_dirs/run/visu",
        help="the visual folder for storing results",
    )
    parser.add_argument("--model_saved_name", default="")
    parser.add_argument(
        "--config", default="./config", help="path to the configuration file"
    )
    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument(
        "--save-score",
        type=str2bool,
        default=False,
        help="if ture, the classification score will be stored",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed for pytorch")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="the interval for printing messages (#iteration)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="the interval for storing models (#iteration)",
    )
    parser.add_argument(
        "--save-epoch",
        type=int,
        default=30,
        help="the start epoch to save model (#iteration)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="the interval for evaluating models (#iteration)",
    )
    parser.add_argument(
        "--print-log", type=str2bool, default=True, help="print logging or not"
    )
    parser.add_argument(
        "--show-topk",
        type=int,
        default=[1, 5],
        nargs="+",
        help="which Top K accuracy will be shown",
    )
    parser.add_argument(
        "--feeder", default="feeder.feeder", help="data loader will be used"
    )
    parser.add_argument(
        "--num-worker", type=int, default=4, help="the number of worker for data loader"
    )
    parser.add_argument(
        "--train-feeder-args",
        action=DictAction,
        default=dict(),
        help="the arguments of data loader for training",
    )
    parser.add_argument(
        "--test-feeder-args",
        action=DictAction,
        default=dict(),
        help="the arguments of data loader for test",
    )
    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument(
        "--model-args", action=DictAction, default=dict(), help="the arguments of model"
    )
    parser.add_argument(
        "--weights", default=None, help="the weights for network initialization"
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )
    parser.add_argument(
        "--base-lr", type=float, default=0.01, help="initial learning rate"
    )
    parser.add_argument(
        "--min-lr", type=float, default=0.01, help="initial learning rate"
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=0.01, help="initial learning rate"
    )
    parser.add_argument("--warmup_prefix", type=bool, default=False)
    parser.add_argument("--warm_up_epoch", type=int, default=0)
    parser.add_argument("--grad-clip", type=bool, default=False)
    parser.add_argument("--grad-max", type=float, default=1.0)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        nargs="+",
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument("--optimizer", default="AdamW", help="type of optimizer")
    parser.add_argument(
        "--lr-scheduler", default="cosine", help="type of learning rate scheduler"
    )
    parser.add_argument(
        "--nesterov", type=str2bool, default=False, help="use nesterov or not"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--start-epoch", type=int, default=0, help="start training from which epoch"
    )
    parser.add_argument(
        "--num-epoch", type=int, default=80, help="stop training in which epoch"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0005, help="weight decay for optimizer"
    )
    parser.add_argument(
        "--lr-ratio", type=float, default=0.001, help="decay rate for learning rate"
    )
    parser.add_argument(
        "--lr-decay-rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--loss-type", type=str, default="CE")
    parser.add_argument(
        "--eval_ablate_ball", action="store_true", help="Enable ball ablation at eval"
    )
    parser.add_argument(
        "--eval_interval_epoch",
        type=int,
        default=1,
        help="evaluate every N epochs during training",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="top1",
        choices=["top1", "macro_acc", "macro_f1"],
        help="metric used to select best checkpoint",
    )
    parser.add_argument(
        "--save-optimizer",
        type=str2bool,
        default=False,
        help="save optimizer + scheduler states in best/last checkpoints",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=1,
        help="number of best checkpoints to keep (1 keeps only best)",
    )
    parser.add_argument(
        "--train_use_sampler",
        type=str2bool,
        default=False,
        help="use weighted random sampler for training",
    )
    parser.add_argument(
        "--ema", type=str2bool, default=True, help="enable EMA of model weights"
    )
    parser.add_argument(
        "--ema_decay", type=float, default=0.999, help="EMA decay (0.999–0.9999)"
    )
    parser.add_argument(
        "--ema_use_for_eval",
        type=str2bool,
        default=True,
        help="use EMA weights during eval/saving",
    )
    parser.add_argument(
        "--eval_tta_clips",
        type=int,
        default=1,
        help="number of temporal clips at eval (1 = off)",
    )
    parser.add_argument("--eval_tta_time_reverse", type=str2bool, default=False)
    parser.add_argument(
        "--vis-feature",
        type=str,
        default="feats",
        choices=["feats", "skel_feats", "concat", "concat_scaled", "logits", "probs"],
        help="Which representation to visualize when do_visualize=True (tsne/umap).",
    )
    parser.add_argument(
        "--center_loss",
        type=str2bool,
        default=True,
        help="enable center loss on features",
    )
    parser.add_argument(
        "--center_loss_weight",
        type=float,
        default=0.02,
        help="weight for center loss term",
    )
    parser.add_argument(
        "--center_alpha",
        type=float,
        default=0.2,
        help="EMA update rate for class centers",
    )
    parser.add_argument(
        "--center_feat_norm",
        type=str2bool,
        default=False,
        help="L2 normalize features for center loss",
    )
    return parser


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == "train":
            if not arg.train_feeder_args["debug"]:
                arg.model_saved_name = os.path.join(arg.work_dir, "runs")
                if os.path.isdir(arg.model_saved_name):
                    print("log_dir: ", arg.model_saved_name, "already exist")
                    answer = input("delete it? y/n:")
                    if answer == "y":
                        shutil.rmtree(arg.model_saved_name)
                        print("Dir removed: ", arg.model_saved_name)
                        input("Refresh the website of tensorboard by pressing any keys")
                    else:
                        print("Dir not removed: ", arg.model_saved_name)
                self.train_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "train"), "train"
                )
                self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "val"), "val"
                )
            else:
                self.train_writer = self.val_writer = SummaryWriter(
                    os.path.join(arg.model_saved_name, "test"), "test"
                )
        self.global_step = 0
        self.load_model()
        self.load_data()
        if self.arg.phase == "train":
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader["train"]))
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device,
                )
        dataset_name = None
        if "ntu" in self.arg.work_dir:
            dataset_name = "ntu60"
        elif "nw_ucla" in self.arg.work_dir:
            dataset_name = "nw_ucla"
        elif "thetis" in self.arg.work_dir:
            dataset_name = "thetis"
        elif "fft-10" in self.arg.work_dir:
            dataset_name = "fft10"
        selected_label_mapping = DATASET_LABEL_MAPS.get(dataset_name, None)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        worker_init = self._make_worker_init_fn(self.arg.seed)
        g = torch.Generator()
        g.manual_seed(self.arg.seed)
        if self.arg.phase == "train":
            train_dataset = Feeder(**self.arg.train_feeder_args)
            use_sampler = getattr(self.arg, "train_use_sampler", False)
            if use_sampler:
                import numpy as np
                from torch.utils.data import WeightedRandomSampler

                labels_np = np.array(train_dataset.label)
                num_classes = self.arg.model_args.get(
                    "num_classes", int(labels_np.max()) + 1
                )
                counts = np.bincount(labels_np, minlength=num_classes).astype(
                    np.float64
                )
                beta = getattr(self.arg, "class_weight_beta", 0.999)
                eff_num = 1.0 - np.power(beta, counts)
                class_weights = (1.0 - beta) / np.maximum(eff_num, 1e-08)
                class_weights = class_weights / (class_weights.mean() + 1e-12)
                sample_weights = class_weights[labels_np]
                sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(labels_np), replacement=True
                )
                self.data_loader["train"] = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=self.arg.batch_size,
                    sampler=sampler,
                    num_workers=self.arg.num_worker,
                    drop_last=True,
                    worker_init_fn=worker_init,
                    generator=g,
                )
                self.print_log(
                    f"Using WeightedRandomSampler for training (beta={beta})."
                )
            else:
                self.data_loader["train"] = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=self.arg.batch_size,
                    shuffle=True,
                    num_workers=self.arg.num_worker,
                    drop_last=True,
                    worker_init_fn=worker_init,
                    generator=g,
                )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_init,
        )
        try:
            if "train" in self.data_loader and (
                not hasattr(self, "class_weights_tensor")
            ):
                import numpy as np

                labels_np = np.array(self.data_loader["train"].dataset.label)
                num_classes = self.arg.model_args.get(
                    "num_classes", int(labels_np.max()) + 1
                )
                counts = np.bincount(labels_np, minlength=num_classes).astype(
                    np.float64
                )
                beta = getattr(self.arg, "class_weight_beta", 0.999)
                eff_num = 1.0 - np.power(beta, counts)
                class_weights = (1.0 - beta) / np.maximum(eff_num, 1e-08)
                class_weights = class_weights / (class_weights.mean() + 1e-12)
                cw_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda(
                    self.output_device
                )
                self.class_weights_tensor = cw_tensor
                if self.arg.loss_type == "CE":
                    self.loss = nn.CrossEntropyLoss(
                        weight=self.class_weights_tensor
                    ).cuda(self.output_device)
                else:
                    self.loss = LabelSmoothingCrossEntropy(
                        smoothing=0.1, weight=self.class_weights_tensor
                    ).cuda(self.output_device)
                self.print_log(
                    f"Class weights set. min={class_weights.min():.3f}, max={class_weights.max():.3f}"
                )
        except Exception as e:
            self.print_log(f"[Warning] Class weight computation failed/skipped: {e}")

    def load_model_old(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        if self.arg.loss_type == "CE":
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)
        if self.arg.weights:
            self.print_log("Load weights from {}.".format(self.arg.weights))
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "r") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            weights = OrderedDict(
                [
                    [k.split("module.")[-1], v.cuda(output_device)]
                    for k, v in weights.items()
                ]
            )
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log(
                                "Sucessfully Remove Weights: {}.".format(key)
                            )
                        else:
                            self.print_log("Can Not Remove Weights: {}.".format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print("Can not find these weights:")
                for d in diff:
                    print("  " + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_model(self):
        output_device = (
            self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        )
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        if self.arg.loss_type == "CE":
            self.loss = nn.CrossEntropyLoss(
                weight=self.class_weights_tensor
                if hasattr(self, "class_weights_tensor")
                else None
            ).cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(
                smoothing=0.1,
                weight=self.class_weights_tensor
                if hasattr(self, "class_weights_tensor")
                else None,
            ).cuda(output_device)
        self.model = self.model.cuda(output_device)
        self.ema = None
        self.center_loss_mod = None
        if getattr(self.arg, "center_loss", True):
            try:
                num_classes = int(self.arg.model_args.get("num_classes", 0))
                feat_dim = None
                if hasattr(self.model, "get_feature_dim") and callable(
                    getattr(self.model, "get_feature_dim")
                ):
                    tmp = self.model.get_feature_dim()
                    if tmp is not None:
                        feat_dim = int(tmp)
                if feat_dim is None and hasattr(self.model, "feat_dim"):
                    feat_dim = int(getattr(self.model, "feat_dim"))
                if feat_dim is None:
                    feat_dim = int(
                        self.arg.model_args.get(
                            "attn_dim",
                            self.arg.model_args.get("stage_dims", [192, 256, 384, 512])[
                                1
                            ],
                        )
                    )
                self.center_loss_mod = CenterLossEMA(
                    num_classes=num_classes,
                    feat_dim=feat_dim,
                    alpha=getattr(self.arg, "center_alpha", 0.3),
                    normalize_feats=getattr(self.arg, "center_feat_norm", False),
                    device=output_device,
                )
                self.print_log(
                    f"CenterLoss enabled (feat_dim={feat_dim}, weight={getattr(self.arg, 'center_loss_weight', 0.01)}, alpha={getattr(self.arg, 'center_alpha', 0.3)})"
                )
            except Exception as e:
                self.print_log(f"[Warning] CenterLoss init failed: {e}")
        else:
            self.print_log("CenterLoss not enabled.")
        if self.arg.weights:
            self.print_log("Load weights from {}.".format(self.arg.weights))
            ckpt = torch.load(self.arg.weights, map_location=f"cuda:{output_device}")
            ema_state = None
            if isinstance(ckpt, dict) and any(
                (k in ckpt for k in ("model", "state_dict", "weights", "ema"))
            ):
                state = ckpt.get("model", ckpt.get("state_dict", ckpt.get("weights")))
                ema_state = ckpt.get("ema", None)
                if "best_metric" in ckpt:
                    try:
                        self.best_metric_value = float(ckpt["best_metric"])
                        self.print_log(
                            f"Resumed best metric value: {self.best_metric_value:.4f}"
                        )
                    except Exception:
                        pass
            else:
                state = ckpt
            if not isinstance(state, (dict, OrderedDict)):
                raise RuntimeError(f"Unexpected checkpoint format: {type(state)}")
            state = OrderedDict(((k.split("module.")[-1], v) for k, v in state.items()))
            self.print_log("--- DIAGNOSTIC: CHECKING WEIGHTS ---")
            model_keys = set(self.model.state_dict().keys())
            loaded_keys = set(state.keys())
            if model_keys == loaded_keys:
                self.print_log("Keys match perfectly between model and checkpoint.")
            else:
                self.print_log("KEY MISMATCH DETECTED!")
                missing_in_checkpoint = model_keys - loaded_keys
                missing_in_model = loaded_keys - model_keys
                if missing_in_checkpoint:
                    self.print_log(f"Missing in checkpoint: {missing_in_checkpoint}")
                if missing_in_model:
                    self.print_log(f"Missing in model (unexpected): {missing_in_model}")
            self.print_log("--- END DIAGNOSTIC ---")

            def _safe_load(target_model, sd, desc):
                try:
                    target_model.load_state_dict(sd, strict=True)
                    self.print_log(f"Successfully loaded {desc} (strict=True).")
                except Exception as e:
                    self.print_log(f"[Warn] strict load failed for {desc}: {e}")
                    target_model.load_state_dict(sd, strict=False)
                    self.print_log(
                        f"Loaded {desc} with strict=False (missing/unexpected keys tolerated)."
                    )

            if ema_state is not None and getattr(self.arg, "ema_use_for_eval", True):
                ema_state = OrderedDict(
                    ((k.split("module.")[-1], v) for k, v in ema_state.items())
                )
                _safe_load(self.model, ema_state, "EMA weights from checkpoint")
                self.ema = ModelEMA(
                    self.model,
                    decay=getattr(self.arg, "ema_decay", 0.999),
                    device=f"cuda:{output_device}",
                    include_buffers=True,
                )
                self.ema.shadow = OrderedDict(
                    ((k, v.clone().detach()) for k, v in ema_state.items())
                )
            else:
                _safe_load(self.model, state, "weights")
        if self.ema is None and getattr(self.arg, "ema", True):
            self.ema = ModelEMA(
                self.model,
                decay=getattr(self.arg, "ema_decay", 0.999),
                device=f"cuda:{output_device}",
                include_buffers=True,
            )
            self.print_log(
                f"EMA enabled (decay={getattr(self.arg, 'ema_decay', 0.999)})"
            )
        try:

            def count_parameters(model):
                return sum((p.numel() for p in model.parameters() if p.requires_grad))

            self.print_log(f"# Parameters: {count_parameters(self.model)}")
        except Exception as e:
            self.print_log(f"[Warn] Failed to count parameters: {e}")

    def load_optimizer(self):
        def build_param_groups(model):
            decay, no_decay = ([], [])
            keys_no_wd = (
                "bias",
                "bn",
                "norm",
                "layernorm",
                "ln",
                "dt_bias",
                "a_log",
                "d",
            )
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                lname = name.lower()
                if (
                    p.ndim == 1
                    or lname.endswith(".bias")
                    or any((k in lname for k in keys_no_wd))
                ):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return [
                {"params": decay, "weight_decay": self.arg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]

        param_groups = build_param_groups(self.model)
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                param_groups,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(param_groups, lr=self.arg.base_lr)
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(param_groups, lr=self.arg.base_lr)
        else:
            raise ValueError()
        self.print_log("using warm up, epoch: {}".format(self.arg.warm_up_epoch))

    def load_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)
        self.lr_scheduler = None
        if self.arg.lr_scheduler == "cosine":
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=num_steps - warmup_steps
                if self.arg.warmup_prefix
                else num_steps,
                lr_min=self.arg.min_lr,
                warmup_lr_init=self.arg.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.arg.warmup_prefix,
            )
        else:
            raise ValueError()

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        if not os.path.exists(self.arg.visual_dir):
            os.makedirs(self.arg.visual_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + " ] " + str
        print(str)
        if self.arg.print_log:
            with open("{}/log.txt".format(self.arg.work_dir), "a") as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log("Training epoch: {}".format(epoch + 1))
        loader = self.data_loader["train"]
        loss_value = []
        acc_value = []
        self.train_writer.add_scalar("epoch", epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        for batch_idx, (
            skeleton_data,
            ball_data,
            ball_mask,
            label,
            sample_name,
            target_frames,
        ) in enumerate(process):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1
            with torch.no_grad():
                skeleton_data = skeleton_data.float().cuda(self.output_device)
                ball_data = ball_data.float().cuda(self.output_device)
                ball_mask = ball_mask.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer["dataloader"] += self.split_time()
            p_drop_skel = getattr(self.arg, "modality_dropout_skeleton", 0.05)
            p_drop_ball = getattr(self.arg, "modality_dropout_ball", 0.05)
            if p_drop_skel > 0 or p_drop_ball > 0:
                r = random.random()
                if r < p_drop_skel:
                    skeleton_data.zero_()
                elif r < p_drop_skel + p_drop_ball:
                    ball_mask.zero_()
                    ball_data[:, :6, :].zero_()
            mask_span_prob = getattr(self.arg, "ball_time_mask_prob", 0.2)
            max_spans = getattr(self.arg, "ball_time_mask_max_spans", 1)
            min_len = getattr(self.arg, "ball_time_mask_min_len", 4)
            max_len = getattr(self.arg, "ball_time_mask_max_len", 10)
            if mask_span_prob > 0 and max_spans > 0:
                B, T = ball_mask.shape
                for i in range(B):
                    if random.random() < mask_span_prob and T > 0:
                        n_spans = 1 if max_spans == 1 else random.randint(1, max_spans)
                        for _ in range(n_spans):
                            if T <= 1:
                                continue
                            span_len = min(max_len, max(min_len, max(1, T // 16)))
                            if span_len >= T:
                                span_len = max(1, T // 2)
                            start = random.randint(0, max(0, T - span_len))
                            end = min(T, start + span_len)
                            ball_mask[i, start:end] = 0.0
                            ball_data[i, :6, start:end] = 0.0
            use_center = (
                getattr(self.arg, "center_loss", True)
                and self.center_loss_mod is not None
            )
            pos_kv = torch.as_tensor(
                target_frames, device=ball_data.device, dtype=ball_data.dtype
            )
            pos_q = pos_kv
            if use_center:
                output, feats, _, ball_preds = self.model(
                    skeleton_data,
                    ball_data,
                    x_ball_mask=ball_mask,
                    pos_q=pos_q,
                    pos_kv=pos_kv,
                    return_features=True,
                )
            else:
                output, ball_preds = self.model(
                    skeleton_data,
                    ball_data,
                    x_ball_mask=ball_mask,
                    pos_q=pos_q,
                    pos_kv=pos_kv,
                    return_features=False,
                )
            if torch.isnan(output).any():
                raise ValueError("Detected NaN in model output!")
            cls_loss = self.loss(output, label)
            ball_gt = ball_data[:, 0:2, :].transpose(1, 2)
            mask_seq = ball_mask[:, 1:].unsqueeze(-1)
            aux_loss = F.mse_loss(
                ball_preds[:, :-1, :] * mask_seq, ball_gt[:, 1:, :] * mask_seq
            )
            loss = cls_loss + 0.1 * aux_loss
            if use_center:
                c_loss = self.center_loss_mod(feats, label)
                loss = cls_loss + getattr(self.arg, "center_loss_weight", 0.01) * c_loss
                try:
                    self.train_writer.add_scalar(
                        "center_loss", c_loss.detach().item(), self.global_step
                    )
                except Exception:
                    pass
            else:
                loss = cls_loss
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.arg.grad_max
                )
            self.optimizer.step()
            if getattr(self, "ema", None) is not None:
                self.ema.update(self.model)
            loss_value.append(loss.data.item())
            timer["model"] += self.split_time()
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar("acc", acc, self.global_step)
            self.train_writer.add_scalar("loss", loss.data.item(), self.global_step)
            self.lr = self.optimizer.param_groups[0]["lr"]
            self.train_writer.add_scalar("lr", self.lr, self.global_step)
            timer["statistics"] += self.split_time()
        proportion = {
            k: "{:02d}%".format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            "\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.".format(
                np.mean(loss_value), np.mean(acc_value) * 100
            )
        )
        self.print_log("\tLearning Rate: {:.4f}".format(self.lr))
        self.print_log(
            "\tTime consumption: [Data]{dataloader}, [Network]{model}".format(
                **proportion
            )
        )
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict(
                [[k.split("module.")[-1], v.cpu()] for k, v in state_dict.items()]
            )
            torch.save(
                weights,
                self.arg.model_saved_name
                + "-"
                + str(epoch + 1)
                + "-"
                + str(int(self.global_step))
                + ".pt",
            )

    def eval(
        self,
        epoch,
        save_score=False,
        loader_name=["test"],
        wrong_file=None,
        result_file=None,
        return_features=False,
        do_visualize=False,
        label_mapping=None,
    ):
        import numpy as np
        import csv
        import pickle
        import os
        from tqdm import tqdm
        from pathlib import Path
        from sklearn.metrics import confusion_matrix, f1_score

        if wrong_file is not None:
            f_w = open(wrong_file, "w")
        if result_file is not None:
            f_r = open(result_file, "w")
        use_ema_eval = (
            getattr(self.arg, "ema_use_for_eval", True)
            and hasattr(self, "ema")
            and (self.ema is not None)
        )
        if use_ema_eval:
            try:
                self.ema.store(self.model)
                self.ema.copy_to(self.model)
            except Exception:
                use_ema_eval = False
        self.model.eval()
        self.print_log("Eval epoch: {}".format(epoch + 1))
        all_feats = []
        all_labels = []
        vis_feature = getattr(self.arg, "vis_feature", "feats")
        if vis_feature not in (
            "feats",
            "skel_feats",
            "concat",
            "concat_scaled",
            "logits",
            "probs",
        ):
            vis_feature = "feats"
        need_model_feats = vis_feature in (
            "feats",
            "skel_feats",
            "concat",
            "concat_scaled",
        )
        try:
            _m = (
                self.model.module
                if isinstance(self.model, nn.DataParallel)
                else self.model
            )
            skeleton_skip_scale_val = float(
                getattr(_m, "skeleton_skip_scale").detach().cpu().item()
            )
        except Exception:
            skeleton_skip_scale_val = 1.0
        metrics_return = None
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            cov_vis_all = []
            spd_hi_all, spd_p50_all, spd_p90_all, spd_p95_all = ([], [], [], [])
            ang_p50_all, ang_p90_all, ang_p95_all = ([], [], [])
            per_class_cov = {}
            per_class_spdhi = {}
            step = 0
            process = tqdm(self.data_loader[ln], desc=f"Evaluating {ln}")
            for batch_idx, (
                skeleton_data,
                ball_data,
                ball_mask,
                label,
                sample_name,
                target_frames,
            ) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    skeleton_data = skeleton_data.float().cuda(self.output_device)
                    ball_data = ball_data.float().cuda(self.output_device)
                    ball_mask = ball_mask.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    if getattr(self.arg, "eval_ablate_ball", False):
                        print("Ablating ball modality for evaluation!")
                        ball_mask.zero_()
                        if ball_data.dim() == 3 and ball_data.size(1) >= 6:
                            ball_data[:, :6, :].zero_()
                    num_clips = int(getattr(self.arg, "eval_tta_clips", 1))
                    if num_clips > 1:
                        ds = self.data_loader[ln].dataset
                        logits_batch = []
                        vis_vec_batch = []
                        B = skeleton_data.size(0)
                        for i in range(B):
                            name_i = sample_name[i]
                            if (
                                hasattr(ds, "name_to_index")
                                and ds.name_to_index is not None
                            ):
                                idx_i = ds.name_to_index.get(
                                    name_i, ds.sample_name.index(name_i)
                                )
                            else:
                                idx_i = ds.sample_name.index(name_i)
                            sk_clips_np, ba_clips_np, mk_clips_np = ds.make_multi_clips(
                                idx_i, num_clips
                            )
                            sk_clips = (
                                torch.from_numpy(sk_clips_np)
                                .float()
                                .cuda(self.output_device)
                            )
                            ba_clips = (
                                torch.from_numpy(ba_clips_np)
                                .float()
                                .cuda(self.output_device)
                            )
                            mk_clips = (
                                torch.from_numpy(mk_clips_np)
                                .float()
                                .cuda(self.output_device)
                            )
                            with torch.no_grad():
                                w = mk_clips.mean(dim=1)
                                topk = (
                                    int(getattr(self.arg, "eval_tta_topk", 1))
                                    if hasattr(self.arg, "eval_tta_topk")
                                    else 1
                                )
                                idx_order = torch.argsort(w, descending=True)
                                sel_idx = (
                                    idx_order[:num_clips]
                                    if topk <= 0
                                    else idx_order[: min(topk, num_clips)]
                                )
                            logits_views = []
                            weights_views = []
                            fused_views = []
                            skel_views = []
                            add_time_rev = (
                                bool(getattr(self.arg, "eval_tta_time_reverse", True))
                                if hasattr(self.arg, "eval_tta_time_reverse")
                                else True
                            )
                            for k in sel_idx.tolist():
                                if return_features and need_model_feats:
                                    out_k, fused_k, skel_k, _ = self.model(
                                        sk_clips[k : k + 1],
                                        ba_clips[k : k + 1],
                                        x_ball_mask=mk_clips[k : k + 1],
                                        return_features=True,
                                    )
                                    fused_views.append(fused_k.squeeze(0))
                                    skel_views.append(skel_k.squeeze(0))
                                else:
                                    out_k, _ = self.model(
                                        sk_clips[k : k + 1],
                                        ba_clips[k : k + 1],
                                        x_ball_mask=mk_clips[k : k + 1],
                                        return_features=False,
                                    )
                                logits_views.append(out_k.squeeze(0))
                                weights_views.append(w[k])
                                if add_time_rev:
                                    sk_rev = sk_clips[k : k + 1].flip(dims=[2])
                                    ba_rev = ba_clips[k : k + 1].flip(dims=[2])
                                    mk_rev = mk_clips[k : k + 1].flip(dims=[1])
                                    if return_features and need_model_feats:
                                        out_r, fused_r, skel_r, _ = self.model(
                                            sk_rev,
                                            ba_rev,
                                            x_ball_mask=mk_rev,
                                            return_features=True,
                                        )
                                        fused_views.append(fused_r.squeeze(0))
                                        skel_views.append(skel_r.squeeze(0))
                                    else:
                                        out_r, _ = self.model(
                                            sk_rev,
                                            ba_rev,
                                            x_ball_mask=mk_rev,
                                            return_features=False,
                                        )
                                    logits_views.append(out_r.squeeze(0))
                                    weights_views.append(w[k])
                            if len(logits_views) == 0:
                                if return_features and need_model_feats:
                                    logits_i, fused_i, skel_i, _ = self.model(
                                        sk_clips[0:1],
                                        ba_clips[0:1],
                                        x_ball_mask=mk_clips[0:1],
                                        return_features=True,
                                    )
                                    fused_views = [fused_i.squeeze(0)]
                                    skel_views = [skel_i.squeeze(0)]
                                else:
                                    logits_i, _ = self.model(
                                        sk_clips[0:1],
                                        ba_clips[0:1],
                                        x_ball_mask=mk_clips[0:1],
                                        return_features=False,
                                    )
                                logits_i = logits_i.squeeze(0)
                            else:
                                L = torch.stack(logits_views, dim=0)
                                ww = torch.stack(weights_views, dim=0)
                                s = ww.sum()
                                if s.item() <= 1e-06:
                                    logits_i = L.mean(dim=0)
                                else:
                                    logits_i = (L * ww[:, None]).sum(dim=0) / s
                                if (
                                    return_features
                                    and need_model_feats
                                    and (len(fused_views) > 0)
                                    and (len(skel_views) > 0)
                                ):
                                    Fv = torch.stack(fused_views, dim=0)
                                    Sv = torch.stack(skel_views, dim=0)
                                    if s.item() <= 1e-06:
                                        fused_i = Fv.mean(dim=0)
                                        skel_i = Sv.mean(dim=0)
                                    else:
                                        fused_i = (Fv * ww[:, None]).sum(dim=0) / s
                                        skel_i = (Sv * ww[:, None]).sum(dim=0) / s
                            logits_batch.append(logits_i)
                            if return_features:
                                if vis_feature == "logits":
                                    vis_vec_batch.append(logits_i)
                                elif vis_feature == "probs":
                                    vis_vec_batch.append(torch.softmax(logits_i, dim=0))
                                elif need_model_feats:
                                    if vis_feature == "feats":
                                        vis_vec_batch.append(fused_i)
                                    elif vis_feature == "skel_feats":
                                        vis_vec_batch.append(skel_i)
                                    elif vis_feature == "concat":
                                        vis_vec_batch.append(
                                            torch.cat([fused_i, skel_i], dim=0)
                                        )
                                    elif vis_feature == "concat_scaled":
                                        vis_vec_batch.append(
                                            torch.cat(
                                                [
                                                    fused_i,
                                                    skel_i
                                                    * float(skeleton_skip_scale_val),
                                                ],
                                                dim=0,
                                            )
                                        )
                        output = torch.stack(logits_batch, dim=0)
                        feats = None
                        if return_features:
                            if len(vis_vec_batch) > 0:
                                all_feats.append(
                                    torch.stack(vis_vec_batch, dim=0)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                    else:
                        pos_kv = torch.as_tensor(
                            target_frames,
                            device=ball_data.device,
                            dtype=ball_data.dtype,
                        )
                        pos_q = pos_kv
                        if return_features:
                            output, feats, skel_feats, _ = self.model(
                                skeleton_data,
                                ball_data,
                                x_ball_mask=ball_mask,
                                pos_q=pos_q,
                                pos_kv=pos_kv,
                                return_features=True,
                            )
                            if vis_feature == "feats":
                                all_feats.append(feats.detach().cpu().numpy())
                            elif vis_feature == "skel_feats":
                                all_feats.append(skel_feats.detach().cpu().numpy())
                            elif vis_feature == "concat":
                                all_feats.append(
                                    torch.cat([feats, skel_feats], dim=1)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            elif vis_feature == "concat_scaled":
                                all_feats.append(
                                    torch.cat(
                                        [
                                            feats,
                                            skel_feats * float(skeleton_skip_scale_val),
                                        ],
                                        dim=1,
                                    )
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                            elif vis_feature == "logits":
                                all_feats.append(output.detach().cpu().numpy())
                            elif vis_feature == "probs":
                                all_feats.append(
                                    F.softmax(output, dim=1).detach().cpu().numpy()
                                )
                        else:
                            output, _ = self.model(
                                skeleton_data,
                                ball_data,
                                x_ball_mask=ball_mask,
                                pos_q=pos_q,
                                pos_kv=pos_kv,
                                return_features=False,
                            )
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.item())
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.cpu().numpy())
                    step += 1
                    try:
                        bd = ball_data.detach().cpu().numpy()
                        bm = ball_mask.detach().cpu().numpy().astype(bool)
                        spd = bd[:, 4, :]
                        ang = bd[:, 5, :]
                        cov = bm.mean(axis=1)
                        cov_vis_all.extend(cov.tolist())
                        thr_spd = 0.1
                        thr_ang = 1.0
                        spd_p50_list, spd_p90_list, spd_p95_list, spd_thr_list = (
                            [],
                            [],
                            [],
                            [],
                        )
                        ang_p50_list, ang_p90_list, ang_p95_list, ang_thr_list = (
                            [],
                            [],
                            [],
                            [],
                        )
                        for b in range(bd.shape[0]):
                            vis = bm[b]
                            if vis.any():
                                spd_vis = spd[b, vis]
                                ang_abs = np.abs(ang[b, vis])
                                spd_p50_list.append(float(np.percentile(spd_vis, 50)))
                                spd_p90_list.append(float(np.percentile(spd_vis, 90)))
                                spd_p95_list.append(float(np.percentile(spd_vis, 95)))
                                spd_thr_list.append(float((spd_vis > thr_spd).mean()))
                                ang_p50_list.append(float(np.percentile(ang_abs, 50)))
                                ang_p90_list.append(float(np.percentile(ang_abs, 90)))
                                ang_p95_list.append(float(np.percentile(ang_abs, 95)))
                                ang_thr_list.append(float((ang_abs > thr_ang).mean()))
                            else:
                                spd_p50_list.append(0.0)
                                spd_p90_list.append(0.0)
                                spd_p95_list.append(0.0)
                                spd_thr_list.append(0.0)
                                ang_p50_list.append(0.0)
                                ang_p90_list.append(0.0)
                                ang_p95_list.append(0.0)
                                ang_thr_list.append(0.0)
                        spd_hi_all.extend(spd_thr_list)
                        spd_p50_all.extend(spd_p50_list)
                        spd_p90_all.extend(spd_p90_list)
                        spd_p95_all.extend(spd_p95_list)
                        ang_p50_all.extend(ang_p50_list)
                        ang_p90_all.extend(ang_p90_list)
                        ang_p95_all.extend(ang_p95_list)
                        labs = label.detach().cpu().numpy()
                        for i in range(len(labs)):
                            c = int(labs[i])
                            per_class_cov.setdefault(c, []).append(float(cov[i]))
                            per_class_spdhi.setdefault(c, []).append(
                                float(spd_thr_list[i])
                            )
                    except Exception:
                        pass
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + "," + str(true[i]) + "\n")
                        if x != true[i] and wrong_file is not None:
                            f_w.write(
                                str(sample_name[i])
                                + ","
                                + str(x)
                                + ","
                                + str(true[i])
                                + "\n"
                            )
            score = np.concatenate(score_frag)
            avg_loss = np.mean(loss_value)
            if "ucla" in self.arg.feeder or "thetis" in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
            print("Accuracy: ", accuracy, " model: ", self.arg.model_saved_name)
            if self.arg.phase == "train":
                self.val_writer.add_scalar("loss", avg_loss, self.global_step)
                self.val_writer.add_scalar("acc", accuracy, self.global_step)
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(
                "\tMean {} loss of {} batches: {}.".format(
                    ln, len(self.data_loader[ln]), avg_loss
                )
            )
            for k in self.arg.show_topk:
                topk_acc = 100 * self.data_loader[ln].dataset.top_k(score, k)
                self.print_log("\tTop{}: {:.2f}%".format(k, topk_acc))
            if save_score:
                score_path = "{}/epoch{}_{}_score.pkl".format(
                    self.arg.work_dir, epoch + 1, ln
                )
                with open(score_path, "wb") as f:
                    pickle.dump(score_dict, f)
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            if return_features:
                all_labels.append(label_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            csv_path = "{}/epoch{}_{}_each_class_acc.csv".format(
                self.arg.work_dir, epoch + 1, ln
            )
            with open(csv_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)
            if len(cov_vis_all) > 0:
                try:
                    cov_mean = float(np.mean(cov_vis_all))
                    spd_hi_mean = (
                        float(np.mean(spd_hi_all))
                        if "spd_hi_all" in locals() or "spd_hi_all" in globals()
                        else 0.0
                    )
                    spd_p50_mean = (
                        float(np.mean(spd_p50_all))
                        if "spd_p50_all" in locals() or "spd_p50_all" in globals()
                        else 0.0
                    )
                    spd_p90_mean = (
                        float(np.mean(spd_p90_all))
                        if "spd_p90_all" in locals() or "spd_p90_all" in globals()
                        else 0.0
                    )
                    spd_p95_mean = (
                        float(np.mean(spd_p95_all))
                        if "spd_p95_all" in locals() or "spd_p95_all" in globals()
                        else 0.0
                    )
                    ang_p50_mean = (
                        float(np.mean(ang_p50_all))
                        if "ang_p50_all" in locals() or "ang_p50_all" in globals()
                        else 0.0
                    )
                    ang_p90_mean = (
                        float(np.mean(ang_p90_all))
                        if "ang_p90_all" in locals() or "ang_p90_all" in globals()
                        else 0.0
                    )
                    ang_p95_mean = (
                        float(np.mean(ang_p95_all))
                        if "ang_p95_all" in locals() or "ang_p95_all" in globals()
                        else 0.0
                    )
                    self.print_log(
                        f"[Diag] {ln} cov≈{cov_mean:.3f} | frac(speed>0.1)≈{spd_hi_mean:.3f}"
                    )
                    self.print_log(
                        f"[Diag] {ln} speed p50≈{spd_p50_mean:.3f}, p90≈{spd_p90_mean:.3f}, p95≈{spd_p95_mean:.3f}"
                    )
                    self.print_log(
                        f"[Diag] {ln} |angle| p50≈{ang_p50_mean:.3f}, p90≈{ang_p90_mean:.3f}, p95≈{ang_p95_mean:.3f}"
                    )
                    diag_csv = "{}/epoch{}_{}_ball_diag.csv".format(
                        self.arg.work_dir, epoch + 1, ln
                    )
                    with open(diag_csv, "w", newline="") as fcsv:
                        w = csv.writer(fcsv)
                        if label_mapping is not None:
                            w.writerow(
                                [
                                    "class_id",
                                    "class_name",
                                    "mean_coverage",
                                    "mean_frac_speed>0.1",
                                ]
                            )
                            for c in sorted(per_class_cov.keys()):
                                cov_m = (
                                    float(np.mean(per_class_cov[c]))
                                    if per_class_cov[c]
                                    else 0.0
                                )
                                hi_m = (
                                    float(np.mean(per_class_spdhi[c]))
                                    if per_class_spdhi.get(c)
                                    else 0.0
                                )
                                cname = label_mapping.get(c, str(c))
                                w.writerow([c, cname, cov_m, hi_m])
                        else:
                            w.writerow(
                                ["class", "mean_coverage", "mean_frac_speed>0.1"]
                            )
                            for c in sorted(per_class_cov.keys()):
                                cov_m = (
                                    float(np.mean(per_class_cov[c]))
                                    if per_class_cov[c]
                                    else 0.0
                                )
                                hi_m = (
                                    float(np.mean(per_class_spdhi[c]))
                                    if per_class_spdhi.get(c)
                                    else 0.0
                                )
                                w.writerow([c, cov_m, hi_m])
                    if self.arg.phase == "train":
                        self.val_writer.add_scalar(
                            f"{ln}/ball_cov_mean", cov_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/speed_frac_gt0.1", spd_hi_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/speed_p50", spd_p50_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/speed_p90", spd_p90_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/speed_p95", spd_p95_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/angle_abs_p50", ang_p50_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/angle_abs_p90", ang_p90_mean, self.global_step
                        )
                        self.val_writer.add_scalar(
                            f"{ln}/angle_abs_p95", ang_p95_mean, self.global_step
                        )
                except Exception as e:
                    self.print_log(f"[Diag] summary failed: {e}")
            try:
                macro_acc = float(np.mean(each_acc))
                self.print_log(f"\tMacroAcc: {macro_acc * 100:.2f}%")
                if self.arg.phase == "train":
                    self.val_writer.add_scalar(
                        f"{ln}/macro_acc", macro_acc, self.global_step
                    )
                macro_f1 = f1_score(label_list, pred_list, average="macro")
                if self.arg.phase == "train":
                    self.val_writer.add_scalar(
                        f"{ln}/macro_f1", macro_f1, self.global_step
                    )
                top1_acc = float(self.data_loader[ln].dataset.top_k(score, 1))
                metrics_return = {
                    "top1": top1_acc,
                    "macro_acc": macro_acc,
                    "macro_f1": float(macro_f1),
                }
            except Exception as e:
                self.print_log(f"[Warning] Macro metrics failed: {e}")
        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()
        if return_features and do_visualize and (len(all_feats) > 0):
            all_feats = np.concatenate(all_feats, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            print("All feature shape:", all_feats.shape)
            print("All label shape:", all_labels.shape)
            tsne_out_path = os.path.join(
                self.arg.visual_dir, f"epoch{epoch + 1}_tsne.png"
            )
            if vis_feature == "feats":
                feat_title = "t-SNE visualization of fused pre-logits features"
            elif vis_feature == "skel_feats":
                feat_title = "t-SNE visualization of skeleton pre-logits features"
            elif vis_feature == "concat":
                feat_title = "t-SNE visualization of concat([fused, skeleton]) pre-logits features"
            elif vis_feature == "concat_scaled":
                feat_title = "t-SNE visualization of concat([fused, skip_scale*skeleton]) pre-logits features"
            elif vis_feature == "logits":
                feat_title = "t-SNE visualization of combined logits"
            else:
                feat_title = "t-SNE visualization of softmax probabilities"
            feat_title = f"{feat_title} ({'+'.join(loader_name)})"
            visualize_features_tsne(
                all_feats,
                all_labels,
                output_path=tsne_out_path,
                label_mapping=label_mapping,
                title=feat_title,
            )
            umap_out_path = os.path.join(
                self.arg.visual_dir, f"epoch{epoch + 1}_umap.png"
            )
            if vis_feature == "feats":
                umap_title = "UMAP visualization of fused pre-logits features"
            elif vis_feature == "skel_feats":
                umap_title = "UMAP visualization of skeleton pre-logits features"
            elif vis_feature == "concat":
                umap_title = "UMAP visualization of concat([fused, skeleton]) pre-logits features"
            elif vis_feature == "concat_scaled":
                umap_title = "UMAP visualization of concat([fused, skip_scale*skeleton]) pre-logits features"
            elif vis_feature == "logits":
                umap_title = "UMAP visualization of combined logits"
            else:
                umap_title = "UMAP visualization of softmax probabilities"
            umap_title = f"{umap_title} ({'+'.join(loader_name)})"
            visualize_features_umap(
                all_feats,
                all_labels,
                output_path=umap_out_path,
                label_mapping=label_mapping,
                title=umap_title,
            )
        if use_ema_eval:
            self.ema.restore(self.model)
        return metrics_return

    def _make_worker_init_fn(self, base_seed: int):
        def _init_fn(worker_id: int):
            init_seed(base_seed + worker_id)

        return _init_fn

    def start(self):
        def get_label_mapping(config_args):
            if "ntu" in config_args.work_dir.lower():
                return DATASET_LABEL_MAPS.get("ntu", None)
            elif "nw_ucla" in config_args.work_dir.lower():
                return DATASET_LABEL_MAPS.get("nw_ucla", None)
            elif "thetis" in config_args.work_dir.lower():
                return DATASET_LABEL_MAPS.get("thetis", None)
            elif "tennis7" in config_args.work_dir.lower():
                return DATASET_LABEL_MAPS.get("tennis7", None)
            elif "fft-10" in config_args.work_dir.lower():
                return DATASET_LABEL_MAPS.get("fft10", None)
            return None

        if self.arg.phase == "train":
            self.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            num_batches_per_epoch = len(self.data_loader["train"])
            self.global_step = int(self.arg.start_epoch * num_batches_per_epoch)

            def count_parameters(model):
                return sum((p.numel() for p in model.parameters() if p.requires_grad))

            self.print_log(f"# Parameters: {count_parameters(self.model)}")
            self.best_metric_value = -1.0
            self.best_metric_epoch = -1
            self.best_ckpts = []
            best_key = getattr(self.arg, "best_metric", "top1")

            def _save_ckpt(path, epoch):
                use_ema = (
                    getattr(self.arg, "ema_use_for_eval", True)
                    and hasattr(self, "ema")
                    and (self.ema is not None)
                )
                state_dict = self.model.state_dict()
                out = OrderedDict()
                for k, v in state_dict.items():
                    kk = k.split("module.")[-1]
                    ema_key = kk
                    if use_ema and ema_key in getattr(self.ema, "shadow", {}):
                        out[kk] = self.ema.shadow[ema_key].detach().cpu()
                    else:
                        out[kk] = v.detach().cpu()
                weights = out
                payload = {"model": weights}
                if use_ema and hasattr(self.ema, "shadow"):
                    payload["ema"] = OrderedDict(
                        (
                            (k.split("module.")[-1], v.detach().cpu())
                            for k, v in self.ema.shadow.items()
                        )
                    )
                if getattr(self.arg, "save_optimizer", False):
                    payload["optimizer"] = self.optimizer.state_dict()
                    if self.lr_scheduler is not None:
                        payload["scheduler"] = self.lr_scheduler.state_dict()
                        payload["epoch"] = epoch + 1
                payload["best_metric"] = self.best_metric_value
                torch.save(payload, path)

            def _save_ckpt_unique(epoch, metric_value):
                use_ema = (
                    getattr(self.arg, "ema_use_for_eval", True)
                    and hasattr(self, "ema")
                    and (self.ema is not None)
                )
                state_dict = self.model.state_dict()
                out = OrderedDict()
                for k, v in state_dict.items():
                    kk = k.split("module.")[-1]
                    ema_key = kk
                    if use_ema and ema_key in getattr(self.ema, "shadow", {}):
                        out[kk] = self.ema.shadow[ema_key].detach().cpu()
                    else:
                        out[kk] = v.detach().cpu()
                payload = {"model": out}
                if use_ema and hasattr(self.ema, "shadow"):
                    payload["ema"] = OrderedDict(
                        (
                            (k.split("module.")[-1], v.detach().cpu())
                            for k, v in self.ema.shadow.items()
                        )
                    )
                if getattr(self.arg, "save_optimizer", False):
                    payload["optimizer"] = self.optimizer.state_dict()
                    if self.lr_scheduler is not None:
                        payload["scheduler"] = self.lr_scheduler.state_dict()
                        payload["epoch"] = epoch + 1
                payload["best_metric"] = self.best_metric_value
                uniq = os.path.join(
                    self.arg.work_dir,
                    f"best_{best_key}_e{epoch + 1}_{metric_value:.4f}.pt",
                )
                torch.save(payload, uniq)
                stable = os.path.join(self.arg.work_dir, f"best_{best_key}.pt")
                try:
                    if os.path.exists(stable):
                        os.remove(stable)
                except Exception:
                    pass
                shutil.copyfile(uniq, stable)
                return (uniq, stable)

            def _maybe_keep_topk(new_val, new_path):
                self.best_ckpts.append((new_val, new_path))
                self.best_ckpts.sort(key=lambda x: x[0], reverse=True)
                keep_k = max(1, int(getattr(self.arg, "keep_top_k", 1)))
                while len(self.best_ckpts) > keep_k:
                    _, drop_path = self.best_ckpts.pop()
                    try:
                        if os.path.exists(drop_path):
                            os.remove(drop_path)
                    except Exception:
                        pass

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch, save_model=False)
                eval_every = int(getattr(self.arg, "eval_interval_epoch", 1))
                do_eval = (
                    (epoch + 1) % eval_every == 0
                    or epoch == 0
                    or epoch + 1 == self.arg.num_epoch
                )
                if do_eval:
                    metrics = self.eval(
                        epoch=epoch,
                        save_score=False,
                        loader_name=["test"],
                        return_features=False,
                    )
                    _save_ckpt(os.path.join(self.arg.work_dir, "last.pt"), epoch)
                    if metrics and best_key in metrics:
                        current = float(metrics[best_key])
                        if current > self.best_metric_value:
                            self.best_metric_value = current
                            self.best_metric_epoch = epoch + 1
                            best_path = os.path.join(
                                self.arg.work_dir, f"best_{best_key}.pt"
                            )
                            uniq_path, stable_path = _save_ckpt_unique(epoch, current)
                            self.print_log(
                                f"[Best] {best_key} improved to {current:.4f} at epoch {epoch + 1}"
                            )
                            self.best_ckpts.append((current, uniq_path))
                            self.best_ckpts.sort(key=lambda x: x[0], reverse=True)
                            keep_k = max(1, int(getattr(self.arg, "keep_top_k", 1)))
                            while len(self.best_ckpts) > keep_k:
                                _, drop_path = self.best_ckpts.pop()
                                try:
                                    if os.path.exists(drop_path):
                                        os.remove(drop_path)
                                except Exception:
                                    pass
            best_path = os.path.join(self.arg.work_dir, f"best_{best_key}.pt")
            if not os.path.exists(best_path):
                self.print_log(
                    f"[Warn] best_{best_key}.pt not found; using last.pt instead."
                )
                best_path = os.path.join(self.arg.work_dir, "last.pt")
            payload = torch.load(best_path, map_location=f"cuda:{self.output_device}")
            weights = (
                payload["model"]
                if isinstance(payload, dict) and "model" in payload
                else payload
            )
            ema_shadow = payload.get("ema", None) if isinstance(payload, dict) else None
            if type(self.arg.device) is list and len(self.arg.device) > 1:
                weights = OrderedDict(
                    [
                        ["module." + k, v.cuda(self.output_device)]
                        for k, v in weights.items()
                    ]
                )
            try:
                self.model.load_state_dict(weights)
            except Exception as e:
                self.print_log(f"[Warn] strict load failed for final best model: {e}")
                self.model.load_state_dict(weights, strict=False)
            if getattr(self.arg, "ema", True):
                try:
                    self.ema = ModelEMA(
                        self.model,
                        decay=getattr(self.arg, "ema_decay", 0.999),
                        device=f"cuda:{self.output_device}",
                        include_buffers=True,
                    )
                    if ema_shadow is not None:
                        ema_proc = OrderedDict()
                        for k, v in ema_shadow.items():
                            kk = k if k.startswith("module.") else k
                            ema_proc[kk] = v.to(device=f"cuda:{self.output_device}")
                        self.ema.shadow = ema_proc
                        self.ema.copy_to(self.model)
                        self.print_log(
                            f"EMA restored from checkpoint snapshot (decay={getattr(self.arg, 'ema_decay', 0.999)})"
                        )
                    else:
                        self.print_log(
                            f"EMA reinitialized from best checkpoint (decay={getattr(self.arg, 'ema_decay', 0.999)})"
                        )
                except Exception as e:
                    self.print_log(
                        f"[Warn] EMA reinit from best checkpoint failed: {e}"
                    )
            wf = best_path.replace(".pt", "_wrong.txt")
            rf = best_path.replace(".pt", "_right.txt")
            final_label_mapping = get_label_mapping(self.arg)
            self.model.return_features = True
            self.arg.print_log = False
            self.eval(
                epoch=self.best_metric_epoch - 1,
                save_score=True,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
                return_features=True,
                do_visualize=True,
                label_mapping=final_label_mapping,
            )
            self.arg.print_log = True
            self.model.return_features = False
            num_params = sum(
                (p.numel() for p in self.model.parameters() if p.requires_grad)
            )
            self.print_log(f"Best {best_key}: {self.best_metric_value:.4f}")
            self.print_log(f"Epoch number: {self.best_metric_epoch}")
            self.print_log(f"Model name: {self.arg.work_dir}")
            self.print_log(f"Model total number of params: {num_params}")
            self.print_log(f"Weight decay: {self.arg.weight_decay}")
            self.print_log(f"Base LR: {self.arg.base_lr}")
            self.print_log(f"Batch Size: {self.arg.batch_size}")
            self.print_log(f"Test Batch Size: {self.arg.test_batch_size}")
            self.print_log(f"seed: {self.arg.seed}")
        elif self.arg.phase == "test":
            wf = self.arg.weights.replace(".pt", "_wrong.txt")
            rf = self.arg.weights.replace(".pt", "_right.txt")
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.arg.print_log = False
            self.print_log("Model:   {}.".format(self.arg.model))
            self.print_log("Weights: {}.".format(self.arg.weights))
            self.model.return_features = True
            final_label_mapping = get_label_mapping(self.arg)
            self.eval(
                epoch=0,
                save_score=True,
                loader_name=["test"],
                wrong_file=wf,
                result_file=rf,
                return_features=True,
                do_visualize=True,
                label_mapping=final_label_mapping,
            )
            self.model.return_features = False
            self.print_log("Done.\n")


if __name__ == "__main__":
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
