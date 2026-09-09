import os
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from feeders import tools
from feeders.fine_tennis10_v2_base import FeatureEngineer, resample_ball_series
from models.layers.graph_utils_coco17 import joint2bone_coco17

_MERGED_STROKE_TYPES = [
    "放小球",
    "擋小球",
    "殺球",
    "挑球",
    "長球",
    "平球",
    "切球",
    "推球",
    "撲球",
    "勾球",
    "發短球",
    "發長球",
]
CLASS_LIST_25: List[str] = (
    ["未知球種"]
    + ["Top_" + s for s in _MERGED_STROKE_TYPES]
    + ["Bottom_" + s for s in _MERGED_STROKE_TYPES]
)
assert len(CLASS_LIST_25) == 25
CLASS_TO_IDX_25 = {c: i for i, c in enumerate(CLASS_LIST_25)}


def active_player_index_from_label(label_str: str) -> int:
    if label_str.startswith("Top_"):
        return 0
    if label_str.startswith("Bottom_"):
        return 1
    return 0


def _make_seq_len_same(
    target_len: int, joints: np.ndarray, shuttle: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    T = len(shuttle)
    if T > target_len:
        need_padding = T % target_len > target_len // 2
        stride = T // target_len + int(need_padding)
        joints = joints[::stride][:target_len]
        shuttle = shuttle[::stride][:target_len]
        new_len = len(shuttle)
        if need_padding:
            pad = target_len - new_len
            joints = np.pad(joints, ((0, pad), (0, 0), (0, 0)))
            shuttle = np.pad(shuttle, ((0, pad), (0, 0)))
    else:
        new_len = T
        pad = target_len - new_len
        joints = np.pad(joints, ((0, pad), (0, 0), (0, 0)))
        shuttle = np.pad(shuttle, ((0, pad), (0, 0)))
    return (joints, shuttle, new_len)


class Feeder(Dataset):
    def __init__(
        self,
        data_path: str,
        ball_data_path: str = None,
        label_file: str = None,
        split: str = "train",
        data_type: str = "j",
        window_size: int = 30,
        aug_method: str = "a123589",
        intra_p: float = 0.5,
        inter_p: float = 0.1,
        ball_jitter_std: float = 0.0,
        ball_drop_prob: float = 0.0,
        ball_drop_max_ratio: float = 0.15,
        ball_impute_mode: str = "linear",
        ball_synth_spike_prob: float = 0.0,
        use_kendall: bool = False,
        kendall_ref_path: Optional[str] = None,
        tangent_noise_std: float = 0.0,
        tangent_joint_drop_prob: float = 0.0,
        tangent_joint_drop_max: int = 3,
        tangent_scale_range: Optional[tuple] = None,
        p_interval=None,
        thres: int = 64,
        uniform: bool = True,
        partition: bool = False,
        dataset: str = "shuttleset25",
        image_wh=None,
        num_joints: int = 17,
        debug: bool = False,
        active_player_unknown: str = "top",
        **_unused,
    ):
        super().__init__()
        if use_kendall:
            raise NotImplementedError(
                "use_kendall is FFT-10-only (h36m-17 reference shape). Set use_kendall=False on ShuttleSet."
            )
        self.data_path = data_path
        self.split = split
        self.data_type = data_type
        self.window_size = int(window_size)
        self.aug_method = aug_method
        self.intra_p = float(intra_p)
        self.inter_p = float(inter_p)
        self.ball_jitter_std = float(ball_jitter_std)
        self.ball_drop_prob = float(ball_drop_prob)
        self.ball_drop_max_ratio = float(ball_drop_max_ratio)
        self.ball_impute_mode = ball_impute_mode
        self.ball_synth_spike_prob = float(ball_synth_spike_prob)
        self.tangent_noise_std = float(tangent_noise_std)
        self.tangent_joint_drop_prob = float(tangent_joint_drop_prob)
        self.tangent_joint_drop_max = int(tangent_joint_drop_max)
        self.tangent_scale_range = (
            tuple(tangent_scale_range) if tangent_scale_range else None
        )
        self.num_joints = int(num_joints)
        self.debug = bool(debug)
        self.active_player_unknown = active_player_unknown
        if data_type not in ("j", "b", "jb"):
            raise ValueError(f"data_type must be 'j' | 'b' | 'jb'. Got {data_type}")
        self._scan_clips()

    def _scan_clips(self):
        split_dir = os.path.join(self.data_path, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.entries: List[Dict] = []
        class_dirs = sorted(os.listdir(split_dir))
        for cls_name in class_dirs:
            if cls_name not in CLASS_TO_IDX_25:
                continue
            cls_idx = CLASS_TO_IDX_25[cls_name]
            cls_dir = os.path.join(split_dir, cls_name)
            stems = sorted(
                {
                    fn.rsplit("_", 1)[0]
                    for fn in os.listdir(cls_dir)
                    if fn.endswith("_joints.npy")
                }
            )
            active_idx = active_player_index_from_label(cls_name)
            for stem in stems:
                self.entries.append(
                    {
                        "stem": stem,
                        "class_name": cls_name,
                        "label": cls_idx,
                        "active_idx": active_idx,
                        "cls_dir": cls_dir,
                    }
                )
        self.label = np.array([e["label"] for e in self.entries], dtype=np.int64)
        self.sample_name = [e["stem"] for e in self.entries]
        self.name_to_index = {name: i for i, name in enumerate(self.sample_name)}
        if self.debug:
            print(
                f"[Feeder ShuttleSet/{self.split}] {len(self.entries)} clips, {len(set((e['class_name'] for e in self.entries)))} classes"
            )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        ent = self.entries[index]
        cls_dir = ent["cls_dir"]
        stem = ent["stem"]
        label = ent["label"]
        active = ent["active_idx"]
        joints_full = np.load(os.path.join(cls_dir, stem + "_joints.npy")).astype(
            np.float32
        )
        shuttle_raw = np.load(os.path.join(cls_dir, stem + "_shuttle.npy")).astype(
            np.float32
        )
        joints = joints_full[:, active, :, :]
        del joints_full
        joints, shuttle, valid_len = _make_seq_len_same(
            self.window_size, joints, shuttle_raw
        )
        skeleton_data = joints.transpose(2, 0, 1)[..., None].astype(np.float32)
        if self.split == "train":
            two_d = skeleton_data.shape[0] != 3
            p = float(np.random.rand())
            if p < self.intra_p:
                if "1" in self.aug_method and (not two_d):
                    skeleton_data = tools.shear(skeleton_data, p=0.5)
                if "2" in self.aug_method and (not two_d):
                    skeleton_data = tools.rotate(skeleton_data, p=0.5)
                if "3" in self.aug_method and (not two_d):
                    skeleton_data = tools.scale(skeleton_data, p=0.5)
                if "5" in self.aug_method and random.random() < 0.5:
                    skeleton_data = skeleton_data[:, ::-1, :, :].copy()
                    shuttle = shuttle[::-1].copy()
                if "6" in self.aug_method:
                    skeleton_data = tools.gaussian_noise(skeleton_data, p=0.5)
                if "7" in self.aug_method and (not two_d):
                    skeleton_data = tools.gaussian_filter(skeleton_data, p=0.5)
                if "8" in self.aug_method and (not two_d):
                    skeleton_data = tools.drop_axis(skeleton_data, p=0.5)
                if "9" in self.aug_method and skeleton_data.shape[1] >= 32:
                    skeleton_data = tools.drop_joint(skeleton_data, p=0.5)
            elif p < self.intra_p + self.inter_p:
                pass
        else:
            skeleton_data = skeleton_data.copy()
        if self.data_type == "b":
            skeleton_data = joint2bone_coco17(skeleton_data)
        elif self.data_type == "jb":
            joints_aug = skeleton_data
            bones_aug = joint2bone_coco17(skeleton_data)
            skeleton_data = np.concatenate([joints_aug, bones_aug], axis=0)
        x_raw = shuttle[:, 0].astype(np.float32, copy=True)
        y_raw = shuttle[:, 1].astype(np.float32, copy=True)
        vis_raw = ((x_raw != 0.0) | (y_raw != 0.0)).astype(np.float32)
        x_for_eng = np.where(vis_raw >= 0.5, x_raw, np.nan).astype(np.float32)
        y_for_eng = np.where(vis_raw >= 0.5, y_raw, np.nan).astype(np.float32)
        frames = np.arange(self.window_size, dtype=np.float32)
        df_ball = pd.DataFrame(
            {
                "Frame": frames.astype(int),
                "Visibility": vis_raw,
                "X": x_for_eng,
                "Y": y_for_eng,
            }
        )
        if self.split == "train" and (
            self.ball_jitter_std > 0 or self.ball_drop_prob > 0
        ):
            df_ball = self._apply_ball_augmentation(df_ball)
        engineer = FeatureEngineer(df_ball, target_length=self.window_size)
        series = engineer.calculate_features()
        target_frames = frames.astype(np.float32)
        ball_tensor, vis_mask = resample_ball_series(
            series,
            self.window_size,
            target_frames=target_frames,
            impute=self.ball_impute_mode,
        )
        ball_tensor[-1, :] = vis_mask
        ball_tensor = ball_tensor.astype(np.float32)
        vis_mask = vis_mask.astype(np.float32)
        return (
            skeleton_data,
            ball_tensor,
            vis_mask,
            np.int64(label),
            stem,
            target_frames,
        )

    def make_multi_clips(
        self, index: int, num_clips: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        skeleton_data, ball_tensor, vis_mask, *_ = self[index]
        n = max(1, int(num_clips))
        return (
            np.repeat(skeleton_data[None, ...], n, axis=0).astype(np.float32),
            np.repeat(ball_tensor[None, ...], n, axis=0).astype(np.float32),
            np.repeat(vis_mask[None, ...], n, axis=0).astype(np.float32),
        )

    def top_k(self, score, top_k):
        score = np.asarray(score)
        if score.shape[0] != len(self.label):
            raise ValueError(
                f"score length ({score.shape[0]}) does not match labels ({len(self.label)})"
            )
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def _apply_ball_augmentation(self, df_ball: pd.DataFrame) -> pd.DataFrame:
        df_ball = df_ball.copy()
        visible_mask = df_ball["Visibility"] >= 1.0
        if self.ball_jitter_std > 0 and visible_mask.any():
            n_vis = int(visible_mask.sum())
            jitter_x = np.random.normal(0.0, self.ball_jitter_std, n_vis).astype(
                np.float32
            )
            jitter_y = np.random.normal(0.0, self.ball_jitter_std, n_vis).astype(
                np.float32
            )
            df_ball.loc[visible_mask, "X"] = (
                df_ball.loc[visible_mask, "X"].astype(np.float32) + jitter_x
            )
            df_ball.loc[visible_mask, "Y"] = (
                df_ball.loc[visible_mask, "Y"].astype(np.float32) + jitter_y
            )
        if self.ball_drop_prob > 0 and random.random() < self.ball_drop_prob:
            n_vis = int(visible_mask.sum())
            if n_vis > 1:
                max_drop = max(1, int(self.ball_drop_max_ratio * n_vis))
                drop_n = random.randint(1, max_drop)
                vis_idx = np.where(visible_mask.values)[0]
                drop_idx = np.random.choice(vis_idx, size=drop_n, replace=False)
                df_ball.loc[drop_idx, "Visibility"] = 0.0
                df_ball.loc[drop_idx, "X"] = np.nan
                df_ball.loc[drop_idx, "Y"] = np.nan
        return df_ball
