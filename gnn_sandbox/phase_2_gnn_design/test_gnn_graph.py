from __future__ import annotations

from typing import Callable, Dict

import torch

from phase_2_gnn_design.log_utils import get_logger
from phase_2_gnn_design.graph_utils import (
    build_feature_knn_edge_index,
    build_grid_edge_index,
    build_hybrid_edge_index,
    build_topk_similarity_edge_index,
    edge_index_summary,
    flatten_feature_map,
    unflatten_node_tensor,
)


def fallback_message_passing(nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    source, target = edge_index
    aggregated = torch.zeros_like(nodes)
    counts = torch.zeros(nodes.size(0), 1, device=nodes.device)

    aggregated.index_add_(0, target, nodes[source])
    counts.index_add_(0, target, torch.ones_like(target, dtype=nodes.dtype).unsqueeze(1))
    counts = counts.clamp_min(1.0)
    return aggregated / counts


def run_graph_block(
    nodes: torch.Tensor,
    edge_builder: Callable[[torch.Tensor], torch.Tensor],
    backend_name: str,
) -> Dict[str, torch.Tensor]:
    edge_indices = [edge_builder(nodes[batch_idx]) for batch_idx in range(nodes.size(0))]

    try:
        from torch_geometric.nn import GATConv  # type: ignore

        channels = nodes.size(-1)
        gat_layer = GATConv(channels, channels, heads=1, concat=False)
        gnn_batches = [gat_layer(nodes[batch_idx], edge_indices[batch_idx]) for batch_idx in range(nodes.size(0))]
        gnn_out = torch.stack(gnn_batches, dim=0)
        backend = "torch_geometric.nn.GATConv"
    except ImportError:
        gnn_out = torch.stack(
            [fallback_message_passing(nodes[batch_idx], edge_indices[batch_idx]) for batch_idx in range(nodes.size(0))],
            dim=0,
        )
        backend = f"fallback mean message passing ({backend_name})"

    return {
        "gnn_out": gnn_out,
        "edge_index_batch0": edge_indices[0],
        "backend": backend,
    }


def run_graph_demo() -> None:
    logger = get_logger("test_gnn_graph")
    torch.manual_seed(11)

    cnn_out = torch.randn(2, 256, 8, 8)
    batch_size, channels, height, width = cnn_out.shape

    nodes = flatten_feature_map(cnn_out)
    num_nodes = nodes.size(1)

    strategies = {
        "grid_4n": lambda batch_nodes: build_grid_edge_index(height, width, connectivity=4),
        "grid_8n": lambda batch_nodes: build_grid_edge_index(height, width, connectivity=8),
        "feature_knn_k8": lambda batch_nodes: build_feature_knn_edge_index(batch_nodes, k=8),
        "topk_similarity_15pct": lambda batch_nodes: build_topk_similarity_edge_index(
            batch_nodes, topk_ratio=0.15
        ),
        "hybrid_grid4_plus_knn4": lambda batch_nodes: build_hybrid_edge_index(
            batch_nodes, height=height, width=width, k=4, grid_connectivity=4
        ),
    }

    logger.info("=== test_gnn_graph.py started ===")
    logger.info("1. CNN feature map shape: %s = [batch, channels, height, width]", tuple(cnn_out.shape))
    logger.info("2. Flattened node tensor shape: %s = [batch, num_nodes, channels]", tuple(nodes.shape))

    for step_idx, (strategy_name, edge_builder) in enumerate(strategies.items(), start=3):
        result = run_graph_block(nodes, edge_builder=edge_builder, backend_name=strategy_name)
        edge_index = result["edge_index_batch0"]
        edge_stats = edge_index_summary(edge_index, num_nodes=num_nodes)

        logger.info("--- Strategy: %s ---", strategy_name)
        logger.info(
            "%s. edge_index shape (batch 0): %s = [2, num_edges]",
            step_idx,
            tuple(edge_index.shape),
        )
        logger.info(
            "%s. edge stats (batch 0): num_edges=%s avg_out_degree=%.2f min_out_degree=%.0f max_out_degree=%.0f density=%.4f",
            step_idx + 1,
            int(edge_stats["num_edges"]),
            edge_stats["avg_out_degree"],
            edge_stats["min_out_degree"],
            edge_stats["max_out_degree"],
            edge_stats["density"],
        )
        logger.info("%s. GNN backend: %s", step_idx + 2, result["backend"])
        logger.info(
            "%s. GNN output node shape: %s = [batch, num_nodes, channels]",
            step_idx + 3,
            tuple(result["gnn_out"].shape),
        )

        output = unflatten_node_tensor(result["gnn_out"], height=height, width=width)
        logger.info(
            "%s. Reshaped feature map shape: %s = [batch, channels, height, width]",
            step_idx + 4,
            tuple(output.shape),
        )

    logger.info("=== test_gnn_graph.py finished ===")


if __name__ == "__main__":
    run_graph_demo()
