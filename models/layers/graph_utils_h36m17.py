import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torchgeometry.core as tgmc

PAIRS_25 = [
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


def _get_default_pairs(num_joints):
    if num_joints == 17:
        return PAIRS_17
    if num_joints == 25:
        return PAIRS_25
    raise ValueError(f"Unsupported h36m joint count: {num_joints}")


def build_adjacency_matrix(num_joints=25, pairs=None):
    if pairs is None:
        pairs = _get_default_pairs(num_joints)
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for v1, v2 in pairs:
        if v1 != v2:
            A[v1, v2] = 1.0
            A[v2, v1] = 1.0
    A += np.eye(num_joints, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-08)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


def compute_graph_distances(num_joints=25, pairs=None):
    if pairs is None:
        pairs = _get_default_pairs(num_joints)
    adj = {i: set() for i in range(num_joints)}
    for v1, v2 in pairs:
        if v1 != v2:
            adj[v1].add(v2)
            adj[v2].add(v1)
    dist = np.full((num_joints, num_joints), fill_value=num_joints, dtype=np.int64)
    for src in range(num_joints):
        dist[src, src] = 0
        queue = deque([src])
        visited = {src}
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist[src, neighbor] = dist[src, node] + 1
                    queue.append(neighbor)
    return dist


def build_spatial_partition_adjacency(num_joints=17, pairs=None, center=7):
    if pairs is None:
        pairs = _get_default_pairs(num_joints)
    dist_matrix = compute_graph_distances(num_joints, pairs)
    center_dist = dist_matrix[center]
    A = np.zeros((3, num_joints, num_joints), dtype=np.float32)
    for i in range(num_joints):
        A[0, i, i] = 1.0
    for i, j in pairs:
        if i == j:
            continue
        for src, dst in [(i, j), (j, i)]:
            if center_dist[dst] < center_dist[src]:
                A[1, src, dst] = 1.0
            elif center_dist[dst] > center_dist[src]:
                A[2, src, dst] = 1.0
            else:
                A[0, src, dst] = 1.0
    return A


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, num_joints=25, pairs=None):
        super().__init__()
        A_norm = build_adjacency_matrix(num_joints, pairs)
        self.register_buffer("A_norm", torch.from_numpy(A_norm))
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A_norm, x)
        x = self.W(x)
        x = self.act(x)
        return x


class GraphDistanceEmbedding(nn.Module):
    def __init__(self, embed_dim, max_dist=10, root_joint=7, num_joints=25, pairs=None):
        super().__init__()
        self.embed = nn.Embedding(max_dist + 1, embed_dim)
        dist_matrix = compute_graph_distances(num_joints, pairs)
        dist_from_root = dist_matrix[root_joint]
        dist_from_root = np.clip(dist_from_root, 0, max_dist)
        self.register_buffer("dist_indices", torch.from_numpy(dist_from_root).long())

    def forward(self):
        return self.embed(self.dist_indices)


class RigidTransformInit(nn.Module):
    def __init__(self, num_frames, num_joints):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        angles = torch.empty(num_frames, 3)
        nn.init.uniform_(angles, a=-0.3, b=0.3)
        rot_matrices = tgmc.conversions.angle_axis_to_rotation_matrix(angles)[:, :3, :3]
        self.weights = nn.Parameter(rot_matrices)

    def forward(self, x):
        return torch.matmul(x, self.weights)


class NonRigidTransformInit(nn.Module):
    def __init__(self, num_frames, num_joints):
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        angles = torch.empty(num_frames * num_joints, 3)
        nn.init.uniform_(angles, a=-0.3, b=0.3)
        rot_matrices = tgmc.conversions.angle_axis_to_rotation_matrix(angles)[:, :3, :3]
        self.weights = nn.Parameter(rot_matrices)

    def forward(self, x):
        B, T, V, dims = x.shape
        x = x.view(B, T * V, 1, dims)
        x = torch.matmul(x, self.weights)
        return x.view(B, T, V, dims)


class KendallTransformLayer(nn.Module):
    def __init__(self, transform_type="rigid", num_frames=64, num_joints=25):
        super().__init__()
        self.transform_type = transform_type
        if transform_type == "rigid":
            self.transform = RigidTransformInit(num_frames, num_joints)
        elif transform_type == "nonrigid":
            self.transform = NonRigidTransformInit(num_frames, num_joints)
        else:
            self.transform = None

    def forward(self, x):
        if self.transform is None:
            return x
        x = x.permute(0, 2, 3, 1)
        x = self.transform(x)
        return x.permute(0, 3, 1, 2)
