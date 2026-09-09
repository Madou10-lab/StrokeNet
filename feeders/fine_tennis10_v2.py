"""H36M-17 Fine-Tennis10 loader and joint/bone transforms for StrokeNet-v2."""
import json
import os
import random
import numpy as np
from feeders import tools
from feeders.fine_tennis10_v2_base import (
    Feeder as BaseFeeder,
    generate_data_dict,
    kendall_transform_numpy,
)

PAIRS_17 = [
    (0, 7),
    (7, 7),
    (9, 8),
    (10, 9),
    (14, 8),
    (15, 14),
    (16, 15),
    (11, 8),
    (12, 11),
    (13, 12),
    (1, 0),
    (2, 1),
    (3, 2),
    (4, 0),
    (5, 4),
    (6, 5),
    (8, 7),
]
H36M17_FLIP_ORDER = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
CENTER_17 = 7
PAIRS_1_17 = [(0, 7), (8, 7)]
PAIRS_2_17 = [(9, 8), (14, 8), (11, 8), (1, 0), (4, 0)]
PAIRS_3_17 = [(10, 9), (15, 14), (12, 11), (2, 1), (5, 4)]
PAIRS_4_17 = [(16, 15), (13, 12), (3, 2), (6, 5)]


def spatial_flip_h36m17(data_numpy, p=0.5):
    if random.random() < p:
        return data_numpy[:, :, H36M17_FLIP_ORDER, :]
    return data_numpy.copy()


def joint2bone_h36m17(joint):
    bone = np.zeros_like(joint)
    for v1, v2 in PAIRS_17:
        bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
    return bone


def bone2joint_h36m17(bone, center):
    joint = np.zeros_like(bone)
    joint[:, :, CENTER_17, :] = center
    for pairs in (PAIRS_1_17, PAIRS_2_17, PAIRS_3_17, PAIRS_4_17):
        for v1, v2 in pairs:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
    return joint


def skeleton_adain_bone_length_h36m17(input_data, ref_data):
    eps = 1e-05
    ref_center = ref_data[:, :, CENTER_17, :]
    bone_i = joint2bone_h36m17(input_data)
    bone_r = joint2bone_h36m17(ref_data)
    bone_length_i = np.linalg.norm(bone_i, axis=0)
    bone_length_r = np.linalg.norm(bone_r, axis=0)
    bone_length_scale = (bone_length_r + eps) / (bone_length_i + eps)
    bone_length_scale = np.expand_dims(bone_length_scale, axis=0)
    bone_i = bone_i * bone_length_scale
    return bone2joint_h36m17(bone_i, ref_center)


class Feeder(BaseFeeder):
    def __init__(self, *args, num_joints=17, **kwargs):
        self.num_joints = int(num_joints)
        super().__init__(*args, **kwargs)

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
                        allowed_ids.add(int(parts[-1]))
                    except Exception:
                        continue
            if allowed_ids:
                for item in self.data_dict:
                    assert (
                        item["label"] in allowed_ids
                    ), f"Unknown class id {item['label']} for {item['file_name']}"
        except Exception as exc:
            print(f"[Warning] Label sanity check skipped or failed: {exc}")
        skeleton_data_list = []
        ball_data_list = []
        label_list = []
        M, V, C = (1, self.num_joints, 3)
        self.sample_name = []
        for data_item in self.data_dict:
            file_name = data_item["file_name"]
            label = data_item["label"]
            skeleton_file_path = os.path.join(self.data_path, file_name + ".json")
            try:
                with open(skeleton_file_path, "r", encoding="utf-8") as handle:
                    json_file = json.load(handle)
                skeletons = json_file.get("skeletons", [])
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                print(f"Error loading skeleton file {skeleton_file_path}: {exc}")
                continue
            num_frames = len(skeletons)
            sample_data = np.zeros((num_frames, M, V, C), dtype=np.float32)
            for t in range(num_frames):
                frame = skeletons[t]
                if len(frame) == V:
                    sample_data[t, 0, :, :] = np.array(frame, dtype=np.float32)
            frame_range = np.arange(num_frames, dtype=np.float32)
            x_raw = np.full(num_frames, np.nan, dtype=np.float32)
            y_raw = np.full(num_frames, np.nan, dtype=np.float32)
            visibility_raw = np.zeros(num_frames, dtype=np.float32)
            ball_file_path = os.path.join(self.ball_data_path, file_name + "_ball.csv")
            if os.path.exists(ball_file_path):
                try:
                    import pandas as pd

                    df_ball = pd.read_csv(ball_file_path)
                    if not df_ball.empty and num_frames > 0:
                        df_ball["Frame"] = df_ball["Frame"].astype(int)
                        df_ball = df_ball[df_ball["Frame"] < num_frames]
                        df_ball = df_ball.sort_values("Frame")
                        for _, row in df_ball.iterrows():
                            frame_idx = int(row["Frame"])
                            if frame_idx < 0 or frame_idx >= num_frames:
                                continue
                            vis_value = float(row.get("Visibility", 0))
                            if vis_value >= 1.0:
                                visibility_raw[frame_idx] = 1.0
                                x_raw[frame_idx] = float(row.get("X", np.nan))
                                y_raw[frame_idx] = float(row.get("Y", np.nan))
                except Exception as exc:
                    print(f"Error processing ball file {ball_file_path}: {exc}")
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
        self.index_by_name = {name: idx for idx, name in enumerate(self.sample_name)}
        self.name_to_index = self.index_by_name

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
                if (
                    "a" in self.aug_method
                    and num_people == 2
                    and (np.random.rand(1) < 0.5)
                ):
                    skeleton_data_numpy = skeleton_data_numpy[:, :, :, np.array([1, 0])]
                if (
                    "b" in self.aug_method
                    and num_people == 2
                    and (np.random.rand(1) < 0.5)
                ):
                    axis_next = np.random.randint(0, 1)
                    temp = skeleton_data_numpy.copy()
                    c_dim, t_dim, v_dim, _ = skeleton_data_numpy.shape
                    temp[:, :, :, axis_next] = np.zeros((c_dim, t_dim, v_dim))
                    skeleton_data_numpy = temp
                if "1" in self.aug_method:
                    skeleton_data_numpy = tools.shear(skeleton_data_numpy, p=0.5)
                if "2" in self.aug_method:
                    skeleton_data_numpy = tools.rotate(skeleton_data_numpy, p=0.5)
                if "3" in self.aug_method and (not self.use_kendall):
                    skeleton_data_numpy = tools.scale(skeleton_data_numpy, p=0.5)
                if "4" in self.aug_method:
                    skeleton_data_numpy = spatial_flip_h36m17(
                        skeleton_data_numpy, p=0.5
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
                        donor_t = int(np.array(data_adain).shape[1])
                        if donor_t > 1:
                            t_idx = (
                                ((index_t + 1.0) * 0.5 * (donor_t - 1))
                                .round()
                                .astype(np.int64)
                            )
                            t_idx = np.clip(t_idx, 0, donor_t - 1)
                            data_adain_aligned = data_adain[:, t_idx]
                            skeleton_data_numpy = skeleton_adain_bone_length_h36m17(
                                skeleton_data_numpy, data_adain_aligned
                            )
        else:
            skeleton_data_numpy = skeleton_data_numpy.copy()
        if self.use_kendall and self.kendall_ref_shape is not None:
            skeleton_data_numpy = kendall_transform_numpy(
                skeleton_data_numpy, self.kendall_ref_shape
            )
            if self.split == "train":
                _, _, v_dim, _ = skeleton_data_numpy.shape
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
                        v_dim, size=min(n_drop, v_dim), replace=False
                    )
                    skeleton_data_numpy[:, :, drop_joints, :] = 0.0
                if self.tangent_scale_range is not None and random.random() < 0.5:
                    lo, hi = self.tangent_scale_range
                    scale = np.random.uniform(lo, hi)
                    skeleton_data_numpy = skeleton_data_numpy * scale
        if self.data_type == "b":
            skeleton_data_numpy = joint2bone_h36m17(skeleton_data_numpy)
        elif self.data_type == "jm":
            skeleton_data_numpy = tools.to_motion(skeleton_data_numpy)
        elif self.data_type == "bm":
            skeleton_data_numpy = joint2bone_h36m17(skeleton_data_numpy)
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
