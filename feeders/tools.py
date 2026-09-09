import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window, thres):
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias : end - bias, :, :]
        cropped_length = data.shape[1]
        c_b = begin + bias
        c_e = end - bias
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), thres), valid_size
        )
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias : begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)
        c_b = begin + bias
        c_e = begin + bias + cropped_length
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, cropped_length)
    data = F.interpolate(data, size=window, mode="linear", align_corners=False)
    data = (
        data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    )
    index_t = torch.arange(start=c_b, end=c_e, dtype=torch.float)
    index_t = F.interpolate(
        index_t[None, None, :], size=window, mode="linear", align_corners=False
    ).squeeze()
    index_t = 2 * index_t / valid_size - 1
    return (data, index_t.numpy(), (c_b, c_e))


def valid_crop_uniform(data_numpy, valid_frame_num, p_interval, window, thres):
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin
    if len(p_interval) == 1:
        p = p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), thres), valid_size
        )
        bias = int((valid_size - cropped_length) / 2)
        if cropped_length < window:
            inds = np.arange(cropped_length)
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            bst = bids[:window]
            inds = bst
        inds = inds + bias
        data = data_numpy[:, inds, :, :]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), thres), valid_size
        )
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        if cropped_length < window:
            inds = np.arange(cropped_length)
        elif window <= cropped_length < 2 * window:
            basic = np.arange(window)
            inds = np.random.choice(window + 1, cropped_length - window, replace=False)
            offset = np.zeros(window + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            bsize = np.diff(bids)
            bst = bids[:window]
            offset = np.random.randint(bsize)
            inds = bst + offset
        inds = inds + bias
        data = data_numpy[:, inds, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)
    data = torch.tensor(data, dtype=torch.float)
    index_t = torch.tensor(inds, dtype=torch.float)
    data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, len(inds))
    if len(inds) != window:
        data = F.interpolate(data, size=window, mode="linear", align_corners=False)
        index_t = F.interpolate(
            index_t[None, None, :], size=window, mode="linear", align_corners=False
        ).squeeze()
    data = (
        data.contiguous().view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
    )
    index_t = 2 * index_t / valid_size - 1
    return (data, index_t.numpy(), inds)


def scale(data_numpy, scale=0.2, p=0.5):
    if random.random() < p:
        scale = 1 + np.random.uniform(-1, 1, size=(3, 1, 1, 1)) * np.array(scale)
        return data_numpy * scale
    else:
        return data_numpy.copy()


transform_order = {
    "ntu": [
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        4,
        5,
        6,
        7,
        16,
        17,
        18,
        19,
        12,
        13,
        14,
        15,
        20,
        23,
        24,
        21,
        22,
    ],
    "h36m": [
        0,
        4,
        5,
        6,
        1,
        2,
        3,
        7,
        8,
        9,
        10,
        14,
        15,
        16,
        11,
        12,
        13,
        19,
        20,
        17,
        18,
        23,
        24,
        21,
        22,
    ],
}


def subtract(data_numpy, p=0.5):
    joint = random.randint(0, 24)
    C, T, V, M = data_numpy.shape
    if random.random() < p:
        data_numpy_new = np.zeros((C, T, V, M))
        for i in range(V):
            data_numpy_new[:, :, i, :] = (
                data_numpy[:, :, i, :] - data_numpy[:, :, joint, :]
            )
        return data_numpy_new
    else:
        return data_numpy.copy()


def temporal_flip(data_numpy, index_t, p=0.5):
    C, T, V, M = data_numpy.shape
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return (data_numpy[:, time_range_reverse, :, :], -index_t)
    else:
        return (data_numpy.copy(), index_t.copy())


def spatial_flip(data_numpy, p=0.5, dataset="ntu"):
    if random.random() < p:
        index = transform_order[dataset]
        return data_numpy[:, :, index, :]
    else:
        return data_numpy.copy()


def rotate(data_numpy, axis=None, angle=None, p=0.5):
    if axis != None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)
    if angle != None:
        angle_next = random.uniform(-angle, angle)
    else:
        angle_next = random.uniform(-30, 30)
    if random.random() < p:
        temp = data_numpy.copy()
        angle = math.radians(angle_next)
        if axis_next == 0:
            R = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(angle), math.sin(angle)],
                    [0, -math.sin(angle), math.cos(angle)],
                ]
            )
        if axis_next == 1:
            R = np.array(
                [
                    [math.cos(angle), 0, -math.sin(angle)],
                    [0, 1, 0],
                    [math.sin(angle), 0, math.cos(angle)],
                ]
            )
        if axis_next == 2:
            R = np.array(
                [
                    [math.cos(angle), math.sin(angle), 0],
                    [-math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp
    else:
        return data_numpy.copy()


def shear(data_numpy, s1=None, s2=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        if s1 != None:
            s1_list = s1
        else:
            s1_list = [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
            ]
        if s2 != None:
            s2_list = s2
        else:
            s2_list = [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
            ]
        R = np.array(
            [
                [1, s1_list[0], s2_list[0]],
                [s1_list[1], 1, s2_list[1]],
                [s1_list[2], s2_list[2], 1],
            ]
        )
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp
    else:
        return data_numpy.copy()


def drop_axis(data_numpy, axis=None, p=0.5):
    if axis != None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp
    else:
        return data_numpy.copy()


def drop_joint(data_numpy, joint_list=None, time_range=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        if joint_list != None:
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, joint_list)
            joint_list_ = sorted(joint_list_)
        else:
            random_int = random.randint(5, 15)
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, random_int)
            joint_list_ = sorted(joint_list_)
        if time_range != None:
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, time_range)
            time_range_ = sorted(time_range_)
        else:
            random_int = random.randint(16, 32)
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, random_int)
            time_range_ = sorted(time_range_)
        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp
    else:
        return data_numpy.copy()


def gaussian_noise(data_numpy, mean=0, std=0.05, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        return temp + noise
    else:
        return data_numpy.copy()


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel=15, sigma=[0.1, 2], p=0.5):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)
        self.p = p

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(
            -np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0))
        )
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        prob = np.random.random_sample()
        x = torch.from_numpy(x).double()
        if prob < self.p:
            x = x.permute(3, 0, 2, 1)
            x = F.conv2d(
                x,
                self.weight,
                padding=(0, int((self.kernel - 1) / 2)),
                groups=self.channels,
            )
            x = x.permute(1, -1, -2, 0)
        return x.numpy()


def gaussian_filter(data_numpy, kernel=15, sig_list=[0.1, 2], p=0.5):
    g = GaussianBlurConv(3, kernel, sig_list, p)
    return g(data_numpy)


def skeleton_adain_bone_length(input, ref, dataset):
    eps = 1e-05
    center = 7
    ref_c = ref[:, :, center, :]
    j2b = joint2bone()
    bone_i = j2b(input, dataset)
    bone_r = j2b(ref, dataset)
    bone_length_i = np.linalg.norm(bone_i, axis=0)
    bone_length_r = np.linalg.norm(bone_r, axis=0)
    bone_length_scale = (bone_length_r + eps) / (bone_length_i + eps)
    bone_length_scale = np.expand_dims(bone_length_scale, axis=0)
    bone_i = bone_i * bone_length_scale
    b2j = bone2joint()
    joint = b2j(bone_i, ref_c, dataset)
    return joint


class joint2bone(nn.Module):
    def __init__(self):
        super(joint2bone, self).__init__()
        self.pairs_ntu = [
            (0, 1),
            (1, 1),
            (2, 20),
            (3, 2),
            (4, 20),
            (5, 4),
            (6, 5),
            (7, 6),
            (8, 20),
            (9, 8),
            (10, 9),
            (11, 10),
            (12, 0),
            (13, 12),
            (14, 13),
            (15, 14),
            (16, 0),
            (17, 16),
            (18, 17),
            (19, 18),
            (20, 1),
            (21, 7),
            (22, 7),
            (23, 11),
            (24, 11),
        ]
        self.pairs_25 = [
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
            (21, 3),
            (22, 3),
            (4, 0),
            (5, 4),
            (6, 5),
            (23, 6),
            (24, 6),
            (8, 7),
            (17, 16),
            (18, 16),
            (19, 13),
            (20, 13),
        ]

    def __call__(self, joint, dataset="ntu"):
        bone = np.zeros_like(joint)
        if dataset == "ntu":
            self.pairs = self.pairs_ntu
        elif dataset == "h36m":
            self.pairs = self.pairs_25
        for v1, v2 in self.pairs:
            bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
        return bone


class bone2joint(nn.Module):
    def __init__(self):
        super(bone2joint, self).__init__()
        self.center_ntu = 1
        self.pairs_1_ntu = [(0, 1), (20, 1)]
        self.pairs_2_ntu = [(2, 20), (4, 20), (8, 20), (12, 0), (16, 0)]
        self.pairs_3_ntu = [(3, 2), (5, 4), (9, 8), (13, 12), (17, 16)]
        self.pairs_4_ntu = [(6, 5), (10, 9), (14, 13), (18, 17)]
        self.pairs_5_ntu = [(7, 6), (11, 10), (15, 14), (19, 18)]
        self.pairs_6_ntu = [(21, 7), (22, 7), (23, 11), (24, 11)]
        self.center_25 = 7
        self.pairs_1_25 = [(0, 7), (8, 7)]
        self.pairs_2_25 = [(9, 8), (14, 8), (11, 8), (1, 0), (4, 0)]
        self.pairs_3_25 = [(10, 9), (15, 14), (12, 11), (2, 1), (5, 4)]
        self.pairs_4_25 = [(16, 15), (13, 12), (3, 2), (6, 5)]
        self.pairs_5_25 = [(17, 16), (18, 16), (19, 13), (20, 13)]
        self.pairs_6_25 = [(21, 3), (22, 3), (23, 6), (24, 6)]

    def __call__(self, bone, center, dataset):
        if dataset == "ntu":
            self.pairs_1 = self.pairs_1_ntu
            self.pairs_2 = self.pairs_2_ntu
            self.pairs_3 = self.pairs_3_ntu
            self.pairs_4 = self.pairs_4_ntu
            self.pairs_5 = self.pairs_5_ntu
            self.pairs_6 = self.pairs_6_ntu
            self.center = self.center_ntu
        elif dataset == "h36m":
            self.pairs_1 = self.pairs_1_25
            self.pairs_2 = self.pairs_2_25
            self.pairs_3 = self.pairs_3_25
            self.pairs_4 = self.pairs_4_25
            self.pairs_5 = self.pairs_5_25
            self.pairs_6 = self.pairs_6_25
            self.center = self.center_25
        joint = np.zeros_like(bone)
        joint[:, :, self.center, :] = center
        for v1, v2 in self.pairs_1:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_2:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_3:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_4:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_5:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_6:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        return joint


def to_motion(input):
    C, T, V, M = input.shape
    motion = np.zeros_like(input)
    motion[:, : T - 1] = np.diff(input, axis=1)
    return motion
