import math
import numpy as np
import random
import pandas as pd
import json
import os
from torch.utils.data import Dataset
from feeders import tools
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import datetime


def _centered_scaled(X):
    X = X - np.mean(X)
    norm = np.linalg.norm(X, "fro")
    if norm != 0:
        X = X / norm
    return X


def _inv_exp(X, Y):
    tr = abs(np.trace(X.dot(Y.T)))
    if tr > 1:
        tr = 1
    theta = math.acos(tr)
    if math.sin(theta) < 0.0001:
        theta = 0.1
    return theta / math.sin(theta) * (Y - math.cos(theta) * X)


def kendall_transform_numpy(skeleton, ref_shape):
    C, T, V, M = skeleton.shape
    out = skeleton.copy()
    for m in range(M):
        for t in range(T):
            frame = skeleton[:, t, :, m].T
            frame = _centered_scaled(frame)
            try:
                frame = _inv_exp(ref_shape, frame)
            except Exception:
                pass
            out[:, t, :, m] = frame.T
    return out


try:
    from scipy.interpolate import PchipInterpolator

    _HAS_PCHIP = True
except ImportError:
    _HAS_PCHIP = False


class FeatureEngineer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_length: int = None,
        outlier_filter: bool = True,
        outlier_tau: float = 0.25,
        outlier_k: float = 8.0,
    ):
        required_cols = ["Frame", "Visibility", "X", "Y"]
        self.df = df.copy()
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0
        self.df = self.df[required_cols]
        self.df["Frame"] = self.df["Frame"].astype(int)
        self.df["Visibility"] = self.df["Visibility"].fillna(0).astype(float)
        self.df[["X", "Y"]] = self.df[["X", "Y"]].astype(float)
        self.df.sort_values("Frame", inplace=True)
        if target_length is None:
            if self.df.empty:
                target_length = 0
            else:
                target_length = int(self.df["Frame"].max()) + 1
        self.target_length = int(target_length)
        self.frames = np.arange(self.target_length, dtype=np.float32)
        self.visibility = np.zeros(self.target_length, dtype=np.float32)
        self.x = np.full(self.target_length, np.nan, dtype=np.float32)
        self.y = np.full(self.target_length, np.nan, dtype=np.float32)
        for _, row in self.df.iterrows():
            frame = int(row["Frame"])
            if frame < 0 or frame >= self.target_length:
                continue
            vis = float(row["Visibility"])
            if vis >= 1.0:
                self.visibility[frame] = 1.0
                self.x[frame] = float(row["X"])
                self.y[frame] = float(row["Y"])
        self._enable_outlier_filter = outlier_filter
        self._outlier_tau = float(outlier_tau)
        self._outlier_k = float(outlier_k)
        self.segments = self._find_segments(self.visibility)
        if self._enable_outlier_filter and len(self.segments) > 0:
            self._filter_outliers()
            self.segments = self._find_segments(self.visibility)

    @staticmethod
    def _find_segments(visibility: np.ndarray) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        start = None
        for idx, flag in enumerate(visibility):
            if flag >= 0.5:
                if start is None:
                    start = idx
            elif start is not None:
                segments.append((start, idx - 1))
                start = None
        if start is not None:
            segments.append((start, len(visibility) - 1))
        return segments

    @staticmethod
    def _segment_interpolate(
        rel_frames: np.ndarray, values: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        if values.size == 0:
            return np.zeros_like(query, dtype=np.float32)
        unique_mask = np.diff(rel_frames, prepend=rel_frames[:1] - 1.0) > 0
        rel_frames = rel_frames[unique_mask]
        values = values[unique_mask]
        if rel_frames.size == 1:
            return np.full_like(query, values[0], dtype=np.float32)
        if _HAS_PCHIP and rel_frames.size >= 2:
            try:
                interpolator = PchipInterpolator(rel_frames, values, extrapolate=False)
                result = interpolator(query)
                if np.isnan(result).any():
                    result = np.interp(query, rel_frames, values)
                return result.astype(np.float32)
            except Exception:
                pass
        return np.interp(query, rel_frames, values).astype(np.float32)

    @staticmethod
    def _wrap_angle(values: np.ndarray) -> np.ndarray:
        return (values + np.pi) % (2 * np.pi) - np.pi

    def _filter_outliers(self) -> None:
        for start, end in list(self.segments):
            L = end - start + 1
            if L < 3:
                continue
            xs = self.x[start : end + 1].copy()
            ys = self.y[start : end + 1].copy()
            finite = np.isfinite(xs) & np.isfinite(ys)
            if finite.sum() < 3:
                continue
            seg_x = xs[finite]
            seg_y = ys[finite]
            range_x = float(np.max(seg_x) - np.min(seg_x)) if seg_x.size else 0.0
            range_y = float(np.max(seg_y) - np.min(seg_y)) if seg_y.size else 0.0
            scale = max(range_x, range_y, 1.0)
            dx = np.diff(xs)
            dy = np.diff(ys)
            s = np.sqrt(dx**2 + dy**2)
            s_finite = s[np.isfinite(s)]
            if s_finite.size == 0:
                continue
            med = float(np.median(s_finite))
            mad = float(np.median(np.abs(s_finite - med)))
            if mad == 0.0:
                mad = 1e-06
            flagged = []
            for i in range(1, L - 1):
                if not (
                    np.isfinite(xs[i - 1])
                    and np.isfinite(xs[i])
                    and np.isfinite(xs[i + 1])
                    and np.isfinite(ys[i - 1])
                    and np.isfinite(ys[i])
                    and np.isfinite(ys[i + 1])
                ):
                    continue
                p_prev = np.array([xs[i - 1], ys[i - 1]], dtype=np.float32)
                p_cur = np.array([xs[i], ys[i]], dtype=np.float32)
                p_next = np.array([xs[i + 1], ys[i + 1]], dtype=np.float32)
                p_interp = 0.5 * (p_prev + p_next)
                err = float(np.linalg.norm(p_cur - p_interp))
                condA = err / scale > self._outlier_tau
                s_left = float(np.linalg.norm(p_cur - p_prev))
                s_right = float(np.linalg.norm(p_next - p_cur))
                condB = max(s_left, s_right) > med + self._outlier_k * mad
                if condA and condB:
                    flagged.append(i)
            if len(flagged) == 0:
                continue
            guard_threshold = max(4, int(0.1 * L))
            if len(flagged) > guard_threshold:
                continue
            mask = np.zeros(L, dtype=bool)
            mask[flagged] = True
            refined_mask = np.zeros_like(mask)
            inds = np.where(mask)[0]
            if inds.size > 0:
                run_start = inds[0]
                prev = inds[0]
                for idx in inds[1:]:
                    if idx == prev + 1:
                        prev = idx
                        continue
                    s_run, e_run = (run_start, prev)
                    if e_run - s_run + 1 <= 2:
                        refined_mask[s_run : e_run + 1] = True
                    elif s_run + 1 <= e_run - 1:
                        refined_mask[s_run + 1 : e_run] = True
                    run_start = idx
                    prev = idx
                s_run, e_run = (run_start, prev)
                if e_run - s_run + 1 <= 2:
                    refined_mask[s_run : e_run + 1] = True
                elif s_run + 1 <= e_run - 1:
                    refined_mask[s_run + 1 : e_run] = True
            final_flagged = np.where(refined_mask)[0].tolist()
            if len(final_flagged) == 0:
                continue
            mask = np.zeros(L, dtype=bool)
            mask[final_flagged] = True
            inds = np.where(mask)[0]
            if inds.size == 0:
                continue
            runs = []
            s0 = inds[0]
            p = inds[0]
            for ii in inds[1:]:
                if ii == p + 1:
                    p = ii
                    continue
                runs.append((s0, p))
                s0 = ii
                p = ii
            runs.append((s0, p))
            for rs, re in runs:
                global_left = start + rs - 1
                global_right = start + re + 1
                if global_left < 0 or global_right >= len(self.x):
                    continue
                if not (
                    np.isfinite(self.x[global_left])
                    and np.isfinite(self.y[global_left])
                    and np.isfinite(self.x[global_right])
                    and np.isfinite(self.y[global_right])
                ):
                    continue
                f_left = (
                    float(self.frame[global_left])
                    if hasattr(self, "frame")
                    else float(global_left)
                )
                f_right = (
                    float(self.frame[global_right])
                    if hasattr(self, "frame")
                    else float(global_right)
                )
                x_left = float(self.x[global_left])
                y_left = float(self.y[global_left])
                x_right = float(self.x[global_right])
                y_right = float(self.y[global_right])
                endpoint_dist = float(np.hypot(x_right - x_left, y_right - y_left))
                guard_dist = 3.0 * float(scale)
                if endpoint_dist > guard_dist:
                    for im in range(rs, re + 1):
                        gidx = start + im
                        self.visibility[gidx] = 0.0
                        self.x[gidx] = np.nan
                        self.y[gidx] = np.nan
                    try:
                        with open("feeder_synthetic_corrections.log", "a") as lf:
                            lf.write(
                                f"{datetime.datetime.now().isoformat()} - REJECTED_CORRECTION run {start + rs}-{start + re} endpoint_dist={endpoint_dist:.1f} guard_dist={guard_dist:.1f} -> marked occluded\n"
                            )
                    except Exception:
                        pass
                    continue
                for im in range(rs, re + 1):
                    gidx = start + im
                    f_i = (
                        float(self.frame[gidx])
                        if hasattr(self, "frame")
                        else float(gidx)
                    )
                    alpha = (
                        (f_i - f_left) / (f_right - f_left)
                        if f_right - f_left != 0
                        else 0.5
                    )
                    syn_x = (1.0 - alpha) * x_left + alpha * x_right
                    syn_y = (1.0 - alpha) * y_left + alpha * y_right
                    self.x[gidx] = syn_x
                    self.y[gidx] = syn_y
                    try:
                        with open("feeder_synthetic_corrections.log", "a") as lf:
                            lf.write(
                                f"{datetime.datetime.now().isoformat()} - corrected frame {gidx} -> ({syn_x:.1f},{syn_y:.1f}) from left@{global_left}=({x_left:.1f},{y_left:.1f}) right@{global_right}=({x_right:.1f},{y_right:.1f})\n"
                            )
                    except Exception:
                        pass

    def calculate_features(self) -> Dict[str, np.ndarray]:
        feature_dim = 6
        total_frames = self.frames.shape[0]
        features = np.zeros((total_frames, feature_dim), dtype=np.float32)
        if total_frames == 0:
            return {
                "frames": self.frames,
                "visibility": self.visibility,
                "features": features,
                "segments": self.segments,
            }
        x_smooth = np.full(total_frames, np.nan, dtype=np.float32)
        y_smooth = np.full(total_frames, np.nan, dtype=np.float32)
        for start, end in self.segments:
            idx = slice(start, end + 1)
            frames_seg = self.frames[idx]
            rel_frames = frames_seg - frames_seg[0]
            x_values = self.x[idx]
            y_values = self.y[idx]
            valid_x = ~np.isnan(x_values)
            valid_y = ~np.isnan(y_values)
            if valid_x.sum() == 0 or valid_y.sum() == 0:
                continue
            x_smooth[idx] = self._segment_interpolate(
                rel_frames[valid_x], x_values[valid_x], rel_frames
            )
            y_smooth[idx] = self._segment_interpolate(
                rel_frames[valid_y], y_values[valid_y], rel_frames
            )
        vis_mask = self.visibility >= 0.5
        if vis_mask.sum() > 0:
            visible_x = x_smooth[vis_mask]
            visible_y = y_smooth[vis_mask]
            finite_mask = np.isfinite(visible_x) & np.isfinite(visible_y)
            if finite_mask.any():
                visible_x = visible_x[finite_mask]
                visible_y = visible_y[finite_mask]
                center_x = float(np.mean(visible_x))
                center_y = float(np.mean(visible_y))
                range_x = float(np.max(visible_x) - np.min(visible_x))
                range_y = float(np.max(visible_y) - np.min(visible_y))
                scale = max(range_x, range_y, 1.0)
            else:
                center_x = 0.0
                center_y = 0.0
                scale = 1.0
        else:
            center_x = 0.0
            center_y = 0.0
            scale = 1.0
        x_centered = x_smooth - center_x
        y_centered = y_smooth - center_y
        x_norm = np.zeros_like(x_centered)
        y_norm = np.zeros_like(y_centered)
        x_norm[vis_mask] = (x_centered[vis_mask] / scale).astype(np.float32)
        y_norm[vis_mask] = (y_centered[vis_mask] / scale).astype(np.float32)
        vx = np.zeros(total_frames, dtype=np.float32)
        vy = np.zeros(total_frames, dtype=np.float32)
        for start, end in self.segments:
            idx = slice(start, end + 1)
            frames_seg = self.frames[idx]
            if frames_seg.size <= 1:
                continue
            x_seg = x_centered[idx]
            y_seg = y_centered[idx]
            if np.isnan(x_seg).all() or np.isnan(y_seg).all():
                continue
            vx_seg = np.gradient(x_seg, frames_seg, edge_order=1)
            vy_seg = np.gradient(y_seg, frames_seg, edge_order=1)
            vx[idx] = (vx_seg / scale).astype(np.float32)
            vy[idx] = (vy_seg / scale).astype(np.float32)
        speed = np.sqrt(vx.astype(np.float64) ** 2 + vy.astype(np.float64) ** 2).astype(
            np.float32
        )
        angle = np.arctan2(vy.astype(np.float64), vx.astype(np.float64)).astype(
            np.float32
        )
        features[:, 0] = x_norm
        features[:, 1] = y_norm
        features[:, 2] = vx
        features[:, 3] = vy
        features[:, 4] = speed
        features[:, 5] = angle
        return {
            "frames": self.frames,
            "visibility": self.visibility,
            "features": features,
            "segments": self.segments,
        }


from typing import Optional


def resample_ball_series(
    series: Dict[str, np.ndarray],
    window_size: int,
    target_frames: Optional[np.ndarray] = None,
    impute: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    frames = series["frames"]
    visibility = series["visibility"]
    features = series["features"]
    segments = series["segments"]
    feature_dim = features.shape[1] if features.ndim == 2 else 0
    output = np.full((window_size, feature_dim), np.nan, dtype=np.float32)
    vis_mask = np.zeros(window_size, dtype=np.float32)
    if window_size <= 0 or feature_dim == 0:
        ball_tensor = np.concatenate(
            [np.zeros((feature_dim, 0), dtype=np.float32), vis_mask[None, :]], axis=0
        )
        return (ball_tensor.astype(np.float32), vis_mask.astype(np.float32))
    max_frame = float(frames[-1]) if frames.size > 0 else 0.0
    if target_frames is None:
        target_frames = (
            np.zeros(window_size, dtype=np.float32)
            if max_frame == 0.0
            else np.linspace(0.0, max_frame, window_size, dtype=np.float32)
        )
    else:
        target_frames = target_frames.astype(np.float32)
    if len(segments) > 0:
        for start, end in segments:
            seg_frames = frames[start : end + 1]
            seg_features = features[start : end + 1]
            if seg_frames.size == 0:
                continue
            mask = (target_frames >= seg_frames[0]) & (target_frames <= seg_frames[-1])
            if not np.any(mask):
                continue
            query = target_frames[mask]
            for channel in range(feature_dim):
                values = seg_features[:, channel]
                if np.isnan(values).all():
                    continue
                if seg_frames.size == 1:
                    output[mask, channel] = values[0]
                else:
                    output[mask, channel] = np.interp(query, seg_frames, values)
            vis_mask[mask] = 1.0
    t_idx = np.arange(window_size, dtype=np.float32)
    filled = output.copy()
    for ch in range(feature_dim):
        vals = filled[:, ch]
        missing = ~np.isfinite(vals)
        if missing.all():
            filled[:, ch] = 0.0
            continue
        known_idx = np.where(~missing)[0].astype(np.float32)
        known_vals = vals[~missing]
        if impute == "nearest":
            interp = np.interp(
                t_idx, known_idx, known_vals, left=known_vals[0], right=known_vals[-1]
            )
            filled[:, ch] = interp.astype(np.float32)
        else:
            interp = np.interp(
                t_idx, known_idx, known_vals, left=known_vals[0], right=known_vals[-1]
            )
            filled[:, ch] = interp.astype(np.float32)
    ball_tensor = np.concatenate([filled.T, vis_mask[None, :]], axis=0)
    return (ball_tensor.astype(np.float32), vis_mask.astype(np.float32))


def read_annotations(annotation_file_path):
    annotations = {}
    with open(annotation_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                filename_label = line.split()
                if len(filename_label) == 2:
                    filename, label = filename_label
                    annotations[filename.replace(".json", "")] = int(label)
                else:
                    print(f"Invalid line in annotation file: {line}")
    return annotations


def create_data_dict(json_directory, annotation_file_path):
    annotations = read_annotations(annotation_file_path)
    data_dict = []
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(json_directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    file_name = data.get("file_name", filename.replace(".json", ""))
                    skeletons = data.get("skeletons", [])
                    length = len(skeletons)
                    label = annotations.get(file_name)
                    if label is None:
                        print(f"No label found for file '{file_name}'. Skipping.")
                        continue
                    data_dict.append(
                        {"file_name": file_name, "length": length, "label": label}
                    )
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON file '{filename}': {e}")
    return data_dict


def generate_data_dict(json_directory, annotation_file_name="annotation.txt"):
    annotation_file_path = os.path.join(json_directory, annotation_file_name)
    if not os.path.exists(annotation_file_path):
        print(
            f"Annotation file '{annotation_file_name}' not found in directory '{json_directory}'."
        )
        return []
    data_dict = create_data_dict(json_directory, annotation_file_path)
    return data_dict


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_file,
        ball_data_path,
        label_path=None,
        p_interval=1,
        split="train",
        data_type="j",
        aug_method="z",
        intra_p=0.5,
        inter_p=0.0,
        window_size=-1,
        debug=False,
        thres=64,
        uniform=False,
        partition=False,
        dataset="h36m",
        ball_jitter_std=0.0,
        ball_drop_prob=0.0,
        ball_drop_max_ratio=0.15,
        ball_impute_mode: str = "linear",
        ball_synth_spike_prob: float = 0.01,
        use_kendall: bool = False,
        kendall_ref_path: str = None,
        image_wh=None,
        tangent_noise_std: float = 0.0,
        tangent_joint_drop_prob: float = 0.0,
        tangent_joint_drop_max: int = 3,
        tangent_scale_range: tuple = None,
    ):
        self.debug = debug
        self.data_path = data_path
        self.ball_data_path = ball_data_path
        self.label_path = label_path
        self.label_file = label_file
        self.split = split
        self.data_type = data_type
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.dataset = dataset
        self.ball_jitter_std = ball_jitter_std
        self.ball_drop_prob = ball_drop_prob
        self.ball_drop_max_ratio = ball_drop_max_ratio
        self.ball_impute_mode = ball_impute_mode
        self.ball_synth_spike_prob = float(ball_synth_spike_prob)
        self.use_kendall = use_kendall
        self.kendall_ref_shape = None
        self.image_wh = image_wh
        self.tangent_noise_std = tangent_noise_std
        self.tangent_joint_drop_prob = tangent_joint_drop_prob
        self.tangent_joint_drop_max = tangent_joint_drop_max
        self.tangent_scale_range = (
            tuple(tangent_scale_range) if tangent_scale_range else None
        )
        self.load_data()
        if self.use_kendall:
            if kendall_ref_path is not None and os.path.exists(kendall_ref_path):
                self.kendall_ref_shape = np.load(kendall_ref_path).astype(np.float32)
                print(f"[Kendall] Loaded reference shape from {kendall_ref_path}")
            else:
                first_sample = self.data[0]
                first_frame = first_sample[:, 0, :, 0].T
                self.kendall_ref_shape = _centered_scaled(first_frame).astype(
                    np.float32
                )
                auto_path = os.path.join(self.data_path, "kendall_ref_shape.npy")
                np.save(auto_path, self.kendall_ref_shape)
                print(
                    f"[Kendall] Computed reference shape from first sample, saved to {auto_path}"
                )

    def load_data(self):
        self.data_dict = generate_data_dict(self.data_path)
        try:
            allowed_ids = set()
            with open(self.label_file, "r", encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cls_id = int(parts[-1])
                        allowed_ids.add(cls_id)
                    except Exception:
                        continue
            if allowed_ids:
                for it in self.data_dict:
                    assert (
                        it["label"] in allowed_ids
                    ), f"Unknown class id {it['label']} for {it['file_name']}"
        except Exception as e:
            print(f"[Warning] Label sanity check skipped or failed: {e}")
        skeleton_data_list = []
        ball_data_list = []
        label_list = []
        M, V, C = (1, 25, 3)
        self.sample_name = []
        for idx, data_item in enumerate(self.data_dict):
            file_name = data_item["file_name"]
            label = data_item["label"]
            skeleton_file_path = os.path.join(self.data_path, file_name + ".json")
            try:
                with open(skeleton_file_path, "r", encoding="utf-8") as f:
                    json_file = json.load(f)
                skeletons = json_file.get("skeletons", [])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading skeleton file {skeleton_file_path}: {e}")
                continue
            num_frames = len(skeletons)
            frames_to_process = num_frames
            sample_data = np.zeros((frames_to_process, M, V, C), dtype=np.float32)
            for t in range(frames_to_process):
                frame = skeletons[t]
                if len(frame) == V:
                    sample_data[t, 0, :, :] = np.array(frame, dtype=np.float32)
            frame_range = np.arange(frames_to_process, dtype=np.float32)
            x_raw = np.full(frames_to_process, np.nan, dtype=np.float32)
            y_raw = np.full(frames_to_process, np.nan, dtype=np.float32)
            visibility_raw = np.zeros(frames_to_process, dtype=np.float32)
            ball_file_path = os.path.join(self.ball_data_path, file_name + "_ball.csv")
            if os.path.exists(ball_file_path):
                try:
                    df_ball = pd.read_csv(ball_file_path)
                    if not df_ball.empty and frames_to_process > 0:
                        df_ball["Frame"] = df_ball["Frame"].astype(int)
                        df_ball = df_ball[df_ball["Frame"] < frames_to_process]
                        df_ball = df_ball.sort_values("Frame")
                        for _, row in df_ball.iterrows():
                            frame_idx = int(row["Frame"])
                            if frame_idx < 0 or frame_idx >= frames_to_process:
                                continue
                            vis_value = float(row.get("Visibility", 0))
                            if vis_value >= 1.0:
                                visibility_raw[frame_idx] = 1.0
                                x_raw[frame_idx] = float(row.get("X", np.nan))
                                y_raw[frame_idx] = float(row.get("Y", np.nan))
                except Exception as e:
                    print(f"Error processing ball file {ball_file_path}: {e}")
            else:
                print(f"Warning: Ball data file not found: {ball_file_path}")
            skeleton_data_list.append(sample_data.transpose(3, 0, 2, 1))
            ball_data_list.append(
                {
                    "frames": frame_range.astype(np.float32, copy=False),
                    "x": x_raw.astype(np.float32, copy=False),
                    "y": y_raw.astype(np.float32, copy=False),
                    "visibility": visibility_raw.astype(np.float32, copy=False),
                }
            )
            label_list.append(label)
            self.sample_name.append(file_name)
        self.data = skeleton_data_list
        self.ball_data = ball_data_list
        self.label = np.array(label_list)
        print(f"{self.data_path}: Skeleton samples: {len(self.data)} (ragged lengths)")
        print(f"{self.data_path}: Ball samples: {len(self.ball_data)}")
        print(f"{self.data_path}: Labels shape: {self.label.shape}")
        try:
            self.name_to_index = {name: i for i, name in enumerate(self.sample_name)}
        except Exception:
            self.name_to_index = None

    def make_multi_clips(
        self, index: int, num_clips: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_clips = max(1, int(num_clips))
        window = int(self.window_size if self.window_size > 0 else 64)
        skel = self.data[index]
        C, Ti, V, M = skel.shape
        valid_mask_t = skel.sum(0).sum(-1).sum(-1) != 0
        valid_size = int(valid_mask_t.sum()) if valid_mask_t.ndim == 1 else Ti
        valid_size = max(valid_size, Ti)
        clips_skel = []
        clips_ball = []
        clips_mask = []
        max_start = max(0, valid_size - window)
        for k in range(num_clips):
            if num_clips == 1:
                start = max_start // 2
            else:
                start = int(round(k * max_start / (num_clips - 1)))
            end = min(Ti, start + window)
            seg_len = end - start
            seg = skel[:, start:end, :, :]
            if seg_len <= 0:
                seg = np.zeros((C, min(window, Ti), V, M), dtype=np.float32)
                seg_len = seg.shape[1]
            x = torch.tensor(seg, dtype=torch.float32)
            x = x.permute(2, 3, 0, 1).contiguous().view(V * M, C, seg_len)
            x = F.interpolate(x, size=window, mode="linear", align_corners=False)
            x = x.view(V, M, C, window).permute(2, 3, 0, 1).contiguous()
            clips_skel.append(x.numpy())
            idx = torch.arange(start, start + seg_len, dtype=torch.float32)
            idx = (
                F.interpolate(
                    idx[None, None, :], size=window, mode="linear", align_corners=False
                )
                .squeeze()
                .numpy()
            )
            index_t_clip = 2 * idx / max(1, valid_size) - 1
            ball_raw = self.ball_data[index]
            ball_tensor, vis_mask, _ = self._build_ball_stream(
                ball_raw, augment=False, index_t=index_t_clip
            )
            clips_ball.append(ball_tensor)
            clips_mask.append(vis_mask)
        return (
            np.stack(clips_skel, axis=0).astype(np.float32),
            np.stack(clips_ball, axis=0).astype(np.float32),
            np.stack(clips_mask, axis=0).astype(np.float32),
        )

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        skeleton_data_numpy = self.data[index]
        ball_raw = self.ball_data[index]
        label = self.label[index]
        sample_name = self.sample_name[index]
        skeleton_data_numpy = np.array(skeleton_data_numpy)
        valid_frame_num = np.sum(skeleton_data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if self.uniform:
            skeleton_data_numpy, index_t, _ = tools.valid_crop_uniform(
                skeleton_data_numpy,
                valid_frame_num,
                self.p_interval,
                self.window_size,
                self.thres,
            )
        else:
            skeleton_data_numpy, index_t, _ = tools.valid_crop_resize(
                skeleton_data_numpy,
                valid_frame_num,
                self.p_interval,
                self.window_size,
                self.thres,
            )
        ball_tensor, vis_mask, target_frames = self._build_ball_stream(
            ball_raw, augment=self.split == "train", index_t=index_t
        )
        if self.split == "train":
            num_people = np.sum(skeleton_data_numpy.sum(0).sum(0).sum(0) != 0)
            p = np.random.rand(1)
            if p < self.intra_p:
                if "a" in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            skeleton_data_numpy = skeleton_data_numpy[
                                :, :, :, np.array([1, 0])
                            ]
                if "b" in self.aug_method:
                    if num_people == 2:
                        if np.random.rand(1) < 0.5:
                            axis_next = np.random.randint(0, 1)
                            temp = skeleton_data_numpy.copy()
                            C, T, V, M = skeleton_data_numpy.shape
                            x_new = np.zeros((C, T, V))
                            temp[:, :, :, axis_next] = x_new
                            skeleton_data_numpy = temp
                if "1" in self.aug_method:
                    skeleton_data_numpy = tools.shear(skeleton_data_numpy, p=0.5)
                if "2" in self.aug_method:
                    skeleton_data_numpy = tools.rotate(skeleton_data_numpy, p=0.5)
                if "3" in self.aug_method and (not self.use_kendall):
                    skeleton_data_numpy = tools.scale(skeleton_data_numpy, p=0.5)
                if "4" in self.aug_method:
                    skeleton_data_numpy = tools.spatial_flip(
                        skeleton_data_numpy, p=0.5, dataset=self.dataset
                    )
                if "5" in self.aug_method and random.random() < 0.5:
                    skeleton_data_numpy = skeleton_data_numpy[:, ::-1, :, :].copy()
                    ball_tensor = np.flip(ball_tensor, axis=1).copy()
                    vis_mask = np.flip(vis_mask, axis=0).copy()
                    index_t = -index_t
                if "6" in self.aug_method:
                    skeleton_data_numpy = tools.gaussian_noise(
                        skeleton_data_numpy, p=0.5
                    )
                if "7" in self.aug_method:
                    skeleton_data_numpy = tools.gaussian_filter(
                        skeleton_data_numpy, p=0.5
                    )
                if "8" in self.aug_method:
                    skeleton_data_numpy = tools.drop_axis(skeleton_data_numpy, p=0.5)
                if "9" in self.aug_method:
                    skeleton_data_numpy = tools.drop_joint(skeleton_data_numpy, p=0.5)
            elif p >= self.intra_p and p < self.intra_p + self.inter_p:
                adain_pool = np.where(self.label == label)[0]
                if adain_pool.size > 0:
                    adain_idx = random.choice(adain_pool)
                    if adain_idx != index:
                        data_adain = self.data[adain_idx]
                        Ti = (
                            int(data_adain.shape[1])
                            if isinstance(data_adain, np.ndarray)
                            else int(np.array(data_adain).shape[1])
                        )
                        if Ti > 1:
                            t_idx = (
                                ((index_t + 1.0) * 0.5 * (Ti - 1))
                                .round()
                                .astype(np.int64)
                            )
                            t_idx = np.clip(t_idx, 0, Ti - 1)
                            assert t_idx.min() >= 0 and t_idx.max() < Ti, (
                                t_idx.min(),
                                t_idx.max(),
                                Ti,
                            )
                            data_adain_aligned = data_adain[:, t_idx]
                            skeleton_data_numpy = tools.skeleton_adain_bone_length(
                                skeleton_data_numpy, data_adain_aligned, self.dataset
                            )
        else:
            skeleton_data_numpy = skeleton_data_numpy.copy()
        if self.use_kendall and self.kendall_ref_shape is not None:
            skeleton_data_numpy = kendall_transform_numpy(
                skeleton_data_numpy, self.kendall_ref_shape
            )
            if self.split == "train":
                C, T, V, M = skeleton_data_numpy.shape
                if self.tangent_noise_std > 0 and random.random() < 0.5:
                    noise = np.random.normal(
                        0, self.tangent_noise_std, skeleton_data_numpy.shape
                    ).astype(np.float32)
                    skeleton_data_numpy = skeleton_data_numpy + noise
                if (
                    self.tangent_joint_drop_prob > 0
                    and random.random() < self.tangent_joint_drop_prob
                ):
                    n_drop = random.randint(1, self.tangent_joint_drop_max)
                    drop_joints = np.random.choice(
                        V, size=min(n_drop, V), replace=False
                    )
                    skeleton_data_numpy[:, :, drop_joints, :] = 0.0
                if self.tangent_scale_range is not None and random.random() < 0.5:
                    lo, hi = self.tangent_scale_range
                    scale = np.random.uniform(lo, hi)
                    skeleton_data_numpy = skeleton_data_numpy * scale
        if self.data_type == "b":
            j2b = tools.joint2bone()
            skeleton_data_numpy = j2b(skeleton_data_numpy, self.dataset)
        elif self.data_type == "jm":
            skeleton_data_numpy = tools.to_motion(skeleton_data_numpy)
        elif self.data_type == "bm":
            j2b = tools.joint2bone()
            skeleton_data_numpy = j2b(skeleton_data_numpy, self.dataset)
            skeleton_data_numpy = tools.to_motion(skeleton_data_numpy)
        else:
            skeleton_data_numpy = skeleton_data_numpy.copy()
        if self.partition:
            skeleton_data_numpy = skeleton_data_numpy[:, :, self.new_idx]
        return (
            skeleton_data_numpy,
            ball_tensor,
            vis_mask,
            label,
            sample_name,
            target_frames,
        )

    def _build_ball_stream(
        self,
        ball_raw: Dict[str, np.ndarray],
        augment: bool,
        index_t: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = ball_raw["frames"]
        visibility = ball_raw["visibility"].copy()
        x = ball_raw["x"].copy()
        y = ball_raw["y"].copy()
        if frames.size == 0:
            target_length = self.window_size if self.window_size > 0 else 0
            ball_tensor = np.zeros((7, max(target_length, 0)), dtype=np.float32)
            vis_mask = np.zeros(max(target_length, 0), dtype=np.float32)
            target_frames = np.zeros(max(target_length, 0), dtype=np.float32)
            return (ball_tensor, vis_mask, target_frames)
        df_ball = pd.DataFrame(
            {"Frame": frames.astype(int), "Visibility": visibility, "X": x, "Y": y}
        ).set_index("Frame")
        if augment:
            df_ball = self._apply_ball_augmentation(df_ball)
        df_ball = df_ball.reset_index()
        engineer = FeatureEngineer(df_ball, target_length=len(frames))
        series = engineer.calculate_features()
        if (
            augment
            and self.ball_synth_spike_prob > 0.0
            and (random.random() < float(self.ball_synth_spike_prob))
        ):
            try:
                feats = series["features"]
                vis = series["visibility"]
                T = feats.shape[0]
                valid_idx = np.where(vis >= 0.5)[0]
                valid_idx = valid_idx[(valid_idx > 0) & (valid_idx < T - 1)]
                if valid_idx.size >= 1:
                    sp = int(np.random.choice(valid_idx))
                    offset_scale = 4.0
                    offset = offset_scale * np.random.randn(2).astype(np.float32)
                    old_x, old_y = (feats[sp, 0], feats[sp, 1])
                    feats[sp, 0] = old_x + offset[0]
                    feats[sp, 1] = old_y + offset[1]
                    vis[sp] = 1.0
                    series["features"] = feats
                    series["visibility"] = vis
                    try:
                        with open("feeder_synthetic_corrections.log", "a") as lf:
                            lf.write(
                                f"{datetime.datetime.now().isoformat()} - INJECTED_SYNTHETIC_SPIKE sample={(self.sample_name[-1] if hasattr(self, 'sample_name') else 'unknown')} frame={sp} old=({old_x:.3f},{old_y:.3f}) offset=({offset[0]:.3f},{offset[1]:.3f})\n"
                            )
                    except Exception:
                        pass
            except Exception as e:
                print(f"[Warning] synthetic spike injection failed: {e}")
        target_length = self.window_size if self.window_size > 0 else len(frames)
        if index_t is not None:
            max_frame = float(len(frames) - 1)
            target_frames = (index_t.astype(np.float32) + 1.0) * 0.5 * max_frame
        else:
            target_frames = None
        ball_tensor, vis_mask = resample_ball_series(
            series,
            target_length,
            target_frames=target_frames,
            impute=self.ball_impute_mode,
        )
        ball_tensor[-1, :] = vis_mask
        if target_frames is None:
            target_frames = (
                np.linspace(
                    0.0,
                    max(float(len(frames) - 1), 0.0),
                    target_length,
                    dtype=np.float32,
                )
                if target_length > 0
                else np.zeros(0, dtype=np.float32)
            )
        return (
            ball_tensor.astype(np.float32),
            vis_mask.astype(np.float32),
            target_frames.astype(np.float32),
        )

    def _apply_ball_augmentation(self, df_ball: pd.DataFrame) -> pd.DataFrame:
        df_ball = df_ball.copy()
        visible_mask = df_ball["Visibility"] >= 1.0
        if self.ball_jitter_std > 0.0 and visible_mask.any():
            x_vals = df_ball.loc[visible_mask, "X"].to_numpy()
            y_vals = df_ball.loc[visible_mask, "Y"].to_numpy()
            range_x = x_vals.max() - x_vals.min() if x_vals.size > 0 else 0.0
            range_y = y_vals.max() - y_vals.min() if y_vals.size > 0 else 0.0
            scale = max(range_x, range_y, 1.0)
            noise_x = np.random.normal(
                0.0, self.ball_jitter_std * scale, size=x_vals.shape
            ).astype(np.float32)
            noise_y = np.random.normal(
                0.0, self.ball_jitter_std * scale, size=y_vals.shape
            ).astype(np.float32)
            x_new = (x_vals.astype(np.float32) + noise_x).astype(np.float32)
            y_new = (y_vals.astype(np.float32) + noise_y).astype(np.float32)
            df_ball.loc[visible_mask, "X"] = x_new
            df_ball.loc[visible_mask, "Y"] = y_new
        if self.ball_drop_prob > 0.0:
            visible_indices = df_ball.index[visible_mask].to_numpy()
            if visible_indices.size > 0 and random.random() < self.ball_drop_prob:
                max_drop = int(max(1, self.ball_drop_max_ratio * visible_indices.size))
                max_drop = min(max_drop, visible_indices.size)
                if max_drop > 0:
                    drop_count = random.randint(1, max_drop)
                    drop_frames = np.random.choice(
                        visible_indices, size=drop_count, replace=False
                    )
                    df_ball.loc[drop_frames, "Visibility"] = 0.0
                    df_ball.loc[drop_frames, ["X", "Y"]] = np.nan
        return df_ball

    def debug_ball_sample(self, index: int, sample_points: int = 5):
        ball_raw = self.ball_data[index]
        frames = ball_raw["frames"]
        visibility = ball_raw["visibility"]
        x = ball_raw["x"]
        y = ball_raw["y"]
        print(
            f"Sample {self.sample_name[index]} | raw detections (first {sample_points}):"
        )
        for frame, vis, x_val, y_val in zip(
            frames[:sample_points],
            visibility[:sample_points],
            x[:sample_points],
            y[:sample_points],
        ):
            print(f"  frame={frame:.0f}, vis={vis:.0f}, x={x_val}, y={y_val}")
        ball_tensor, vis_mask, _ = self._build_ball_stream(
            ball_raw, augment=False, index_t=None
        )
        print(
            f"Processed ball tensor shape: {ball_tensor.shape}, visibility mask sum: {vis_mask.sum()}"
        )
        print(
            "First timestep features:",
            ball_tensor[:, 0] if ball_tensor.shape[1] else "N/A",
        )

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
