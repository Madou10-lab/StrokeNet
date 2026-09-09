import numpy as np
from collections import deque

PAIRS_COCO17 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def build_adjacency_matrix(num_joints=17, pairs=None):
    if pairs is None:
        pairs = PAIRS_COCO17
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for v1, v2 in pairs:
        if v1 != v2:
            A[v1, v2] = 1.0
            A[v2, v1] = 1.0
    A += np.eye(num_joints, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-08)))
    return D_inv_sqrt @ A @ D_inv_sqrt


def compute_graph_distances(num_joints=17, pairs=None):
    if pairs is None:
        pairs = PAIRS_COCO17
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


BONE_PAIRS_COCO17 = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 1),
    (4, 2),
    (5, 0),
    (6, 0),
    (7, 5),
    (9, 7),
    (8, 6),
    (10, 8),
    (11, 5),
    (12, 6),
    (13, 11),
    (15, 13),
    (14, 12),
    (16, 14),
]


def joint2bone_coco17(joint):
    bone = np.zeros_like(joint)
    for child, parent in BONE_PAIRS_COCO17:
        bone[:, :, child, :] = joint[:, :, child, :] - joint[:, :, parent, :]
    return bone
