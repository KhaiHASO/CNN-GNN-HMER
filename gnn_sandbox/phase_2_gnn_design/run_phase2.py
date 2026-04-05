from __future__ import annotations

import torch
import torch.nn.functional as F

from phase_1_baseline_analysis.sandbox_dataset import (
    build_extreme_demo_batch,
    summarize_encoder_input,
    summarize_tensor_stats,
)
from phase_2_gnn_design.log_utils import get_logger
from phase_2_gnn_design.sandbox_encoder import Encoder


def main() -> None:
    logger = get_logger("run_phase2")
    torch.manual_seed(23)

    batch, mask, fnames, captions = build_extreme_demo_batch(batch_size=4, seed=7)

    phase1_encoder = Encoder(
        d_model=256,
        growth_rate=24,
        num_layers=4,
        use_graph_refinement=False,
    )
    phase2_encoder = Encoder(
        d_model=256,
        growth_rate=24,
        num_layers=4,
        use_graph_refinement=True,
        edge_mode="hybrid",
        grid_connectivity=4,
        knn_k=4,
    )
    phase2_encoder.load_state_dict(phase1_encoder.state_dict(), strict=False)

    with torch.no_grad():
        phase1_feature, phase1_mask = phase1_encoder(batch, mask)

        cnn_feature, downsampled_mask = phase2_encoder.model(batch, mask)
        projected = phase2_encoder.feature_proj(cnn_feature)
        refined_map, edge_stats = phase2_encoder.graph_block(projected)
        phase2_feature = refined_map.permute(0, 2, 3, 1)
        phase2_feature = phase2_encoder.pos_enc_2d(phase2_feature, downsampled_mask)
        phase2_feature = phase2_encoder.norm(phase2_feature)
        phase2_mask = downsampled_mask

    feature_delta = (phase2_feature - phase1_feature).abs()
    cosine = F.cosine_similarity(
        phase1_feature.reshape(phase1_feature.size(0), -1),
        phase2_feature.reshape(phase2_feature.size(0), -1),
        dim=1,
    )

    logger.info("=== Phase 2 started ===")
    logger.info("Phase name: GNN Design & Edge Generation")
    logger.info("1. %s", summarize_encoder_input(batch))
    logger.info("2. Sample names: %s", ", ".join(fnames))
    logger.info("3. Caption lengths: %s", [len(caption.split()) for caption in captions])
    logger.info("4. %s", summarize_tensor_stats("Shared input batch", batch))
    logger.info("5. Shared active mask pixels per sample: %s", mask.sum(dim=(1, 2)).tolist())

    logger.info("--- Phase 1 baseline on shared data ---")
    logger.info("6. Phase 1 feature shape: %s", tuple(phase1_feature.shape))
    logger.info("7. %s", summarize_tensor_stats("Phase 1 feature", phase1_feature))
    logger.info("8. Phase 1 mask shape: %s", tuple(phase1_mask.shape))
    logger.info("9. Phase 1 active output mask pixels: %s", phase1_mask.sum(dim=(1, 2)).tolist())

    logger.info("--- Phase 2 graph refinement on shared data ---")
    logger.info("10. CNN feature map before graph block: %s", tuple(cnn_feature.shape))
    logger.info("11. Projected feature map before graph block: %s", tuple(projected.shape))
    logger.info(
        "12. Graph stats: num_edges=%s avg_out_degree=%.2f min_out_degree=%.0f max_out_degree=%.0f density=%.4f",
        int(edge_stats["num_edges"]),
        edge_stats["avg_out_degree"],
        edge_stats["min_out_degree"],
        edge_stats["max_out_degree"],
        edge_stats["density"],
    )
    logger.info("13. Phase 2 feature shape: %s", tuple(phase2_feature.shape))
    logger.info("14. %s", summarize_tensor_stats("Phase 2 feature", phase2_feature))
    logger.info("15. Phase 2 mask shape: %s", tuple(phase2_mask.shape))
    logger.info("16. Phase 2 active output mask pixels: %s", phase2_mask.sum(dim=(1, 2)).tolist())

    logger.info("--- Comparison with Phase 1 ---")
    logger.info("17. Mean absolute feature delta: %.6f", feature_delta.mean().item())
    logger.info("18. Max absolute feature delta: %.6f", feature_delta.max().item())
    logger.info("19. Cosine similarity per sample: %s", [round(x, 6) for x in cosine.tolist()])

    same_shape = tuple(phase1_feature.shape) == tuple(phase2_feature.shape)
    same_mask = torch.equal(phase1_mask, phase2_mask)
    conclusion = (
        "Phase 2 giu nguyen giao dien tensor cua Phase 1 "
        f"(same_feature_shape={same_shape}, same_mask={same_mask}) "
        f"nhung bo sung message passing tren do thi hybrid voi {int(edge_stats['num_edges'])} canh/batch-0. "
        f"Dieu nay tao bien doi dac trung co kiem soat (mean_abs_delta={feature_delta.mean().item():.6f}) "
        "ma khong lam vo luong du lieu dau ra cua encoder."
    )
    logger.info("20. Conclusion: %s", conclusion)
    logger.info("=== Phase 2 finished ===")


if __name__ == "__main__":
    main()
