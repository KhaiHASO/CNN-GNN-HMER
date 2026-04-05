from __future__ import annotations

import torch

from phase_2_gnn_design.log_utils import get_logger
from phase_2_gnn_design.sandbox_encoder import Encoder


def run_encoder_graph_demo() -> None:
    logger = get_logger("test_encoder_graph_block")
    torch.manual_seed(17)

    img = torch.randn(2, 1, 128, 128)
    img_mask = torch.ones(2, 128, 128, dtype=torch.long)

    encoder = Encoder(
        d_model=256,
        growth_rate=24,
        num_layers=4,
        use_graph_refinement=True,
        edge_mode="hybrid",
        grid_connectivity=4,
        knn_k=4,
    )

    with torch.no_grad():
        cnn_feature, downsampled_mask = encoder.model(img, img_mask)
        projected = encoder.feature_proj(cnn_feature)
        refined, edge_stats = encoder.graph_block(projected)
        feature = refined.permute(0, 2, 3, 1)
        feature = encoder.pos_enc_2d(feature, downsampled_mask)
        feature = encoder.norm(feature)

    logger.info("=== test_encoder_graph_block.py started ===")
    logger.info("1. Input image shape: %s = [batch, channels, height, width]", tuple(img.shape))
    logger.info("2. CNN feature map shape: %s = [batch, channels, h', w']", tuple(cnn_feature.shape))
    logger.info("3. Projected feature map shape: %s = [batch, d_model, h', w']", tuple(projected.shape))
    logger.info("4. Refined feature map shape: %s = [batch, d_model, h', w']", tuple(refined.shape))
    logger.info("5. Final encoder feature shape: %s = [batch, h', w', d_model]", tuple(feature.shape))
    logger.info("6. Downsampled mask shape: %s = [batch, h', w']", tuple(downsampled_mask.shape))
    logger.info(
        "7. Graph stats: num_edges=%s avg_out_degree=%.2f min_out_degree=%.0f max_out_degree=%.0f density=%.4f",
        int(edge_stats["num_edges"]),
        edge_stats["avg_out_degree"],
        edge_stats["min_out_degree"],
        edge_stats["max_out_degree"],
        edge_stats["density"],
    )
    logger.info("=== test_encoder_graph_block.py finished ===")


if __name__ == "__main__":
    run_encoder_graph_demo()
