from phase_1_baseline_analysis.log_utils import get_logger
from phase_1_baseline_analysis.sandbox_dataset import (
    build_extreme_demo_batch,
    summarize_encoder_input,
    summarize_tensor_stats,
)
from phase_2_gnn_design.sandbox_encoder import Encoder


def main() -> None:
    logger = get_logger("test_data")
    batch, mask, fnames, captions = build_extreme_demo_batch(batch_size=4, seed=7)

    logger.info("=== test_data.py started ===")
    logger.info("1. %s", summarize_encoder_input(batch))
    logger.info("2. Sample names: %s", ", ".join(fnames))
    logger.info("3. Caption lengths: %s", [len(caption.split()) for caption in captions])
    logger.info("4. %s", summarize_tensor_stats("Input batch", batch))
    logger.info("5. Active mask pixels per sample: %s", mask.sum(dim=(1, 2)).tolist())

    encoder = Encoder(d_model=256, growth_rate=24, num_layers=4)
    feature, feature_mask = encoder(batch, mask)

    logger.info("6. Encoder output feature shape: %s", tuple(feature.shape))
    logger.info("7. %s", summarize_tensor_stats("Encoder feature", feature))
    logger.info("8. Encoder output mask shape: %s", tuple(feature_mask.shape))
    logger.info("9. Active output mask pixels per sample: %s", feature_mask.sum(dim=(1, 2)).tolist())
    logger.info("=== test_data.py finished ===")


if __name__ == "__main__":
    main()
