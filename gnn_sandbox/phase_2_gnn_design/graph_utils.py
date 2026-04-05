from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def flatten_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    if feature_map.dim() != 4:
        raise ValueError(
            f"Expected feature_map with shape [batch, channels, height, width], got {tuple(feature_map.shape)}"
        )

    batch_size, channels, height, width = feature_map.shape
    return feature_map.view(batch_size, channels, height * width).permute(0, 2, 1).contiguous()


def unflatten_node_tensor(
    nodes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if nodes.dim() != 3:
        raise ValueError(
            f"Expected nodes with shape [batch, num_nodes, channels], got {tuple(nodes.shape)}"
        )

    batch_size, num_nodes, channels = nodes.shape
    expected = height * width
    if num_nodes != expected:
        raise ValueError(f"Expected {expected} nodes for {height}x{width}, got {num_nodes}")

    return nodes.permute(0, 2, 1).reshape(batch_size, channels, height, width).contiguous()


def build_grid_edge_index(
    height: int,
    width: int,
    connectivity: int = 4,
    self_loops: bool = False,
) -> torch.Tensor:
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    edges = []
    if connectivity == 4:
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        offsets = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    for row in range(height):
        for col in range(width):
            src = row * width + col
            if self_loops:
                edges.append((src, src))
            for d_row, d_col in offsets:
                n_row = row + d_row
                n_col = col + d_col
                if 0 <= n_row < height and 0 <= n_col < width:
                    dst = n_row * width + n_col
                    edges.append((src, dst))

    return _edges_to_tensor(edges)


def build_feature_knn_edge_index(
    node_features: torch.Tensor,
    k: int = 8,
    metric: str = "cosine",
    self_loops: bool = False,
    symmetric: bool = True,
) -> torch.Tensor:
    if node_features.dim() != 2:
        raise ValueError(
            f"Expected node_features with shape [num_nodes, channels], got {tuple(node_features.shape)}"
        )

    num_nodes = node_features.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=node_features.device)

    k = max(1, min(k, num_nodes - (1 if not self_loops else 0)))
    similarity = _pairwise_similarity(node_features, metric=metric)

    if not self_loops:
        similarity.fill_diagonal_(-float("inf"))

    knn_indices = similarity.topk(k=k, dim=1).indices
    source = torch.arange(num_nodes, device=node_features.device).unsqueeze(1).expand_as(knn_indices)
    edge_index = torch.stack((source.reshape(-1), knn_indices.reshape(-1)), dim=0)
    return make_edge_index_undirected(edge_index) if symmetric else edge_index.contiguous()


def build_topk_similarity_edge_index(
    node_features: torch.Tensor,
    topk_ratio: float = 0.15,
    metric: str = "cosine",
    self_loops: bool = False,
) -> torch.Tensor:
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")

    num_nodes = node_features.size(0)
    max_neighbors = num_nodes if self_loops else max(1, num_nodes - 1)
    topk = max(1, int(round(max_neighbors * topk_ratio)))
    return build_feature_knn_edge_index(
        node_features=node_features,
        k=topk,
        metric=metric,
        self_loops=self_loops,
        symmetric=True,
    )


def build_hybrid_edge_index(
    node_features: torch.Tensor,
    height: int,
    width: int,
    k: int = 8,
    grid_connectivity: int = 4,
) -> torch.Tensor:
    grid_edges = build_grid_edge_index(height=height, width=width, connectivity=grid_connectivity)
    knn_edges = build_feature_knn_edge_index(node_features=node_features, k=k)
    merged = torch.cat((grid_edges, knn_edges), dim=1)
    return unique_edge_index(merged)


def make_edge_index_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    reversed_edge_index = edge_index.flip(0)
    return unique_edge_index(torch.cat((edge_index, reversed_edge_index), dim=1))


def unique_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.contiguous()
    unique_edges = torch.unique(edge_index.t(), dim=0)
    return unique_edges.t().contiguous()


def edge_index_summary(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, float]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Expected edge_index with shape [2, num_edges], got {tuple(edge_index.shape)}")

    degrees = torch.bincount(edge_index[0], minlength=num_nodes).float()
    return {
        "num_edges": float(edge_index.size(1)),
        "avg_out_degree": float(degrees.mean().item()),
        "min_out_degree": float(degrees.min().item()),
        "max_out_degree": float(degrees.max().item()),
        "density": float(edge_index.size(1) / max(1, num_nodes * num_nodes)),
    }


def _pairwise_similarity(node_features: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "cosine":
        normalized = F.normalize(node_features, p=2, dim=1)
        return normalized @ normalized.t()
    if metric == "dot":
        return node_features @ node_features.t()
    raise ValueError(f"Unsupported metric: {metric}")


def _edges_to_tensor(edges: list[Tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
