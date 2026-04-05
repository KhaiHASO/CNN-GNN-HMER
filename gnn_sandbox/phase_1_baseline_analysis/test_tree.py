from phase_1_baseline_analysis.log_utils import get_logger
from phase_1_baseline_analysis.sandbox_dataset import build_extreme_demo_samples
from phase_1_baseline_analysis.sandbox_latex2gtd import node2list, tex2tree, tree_to_lines


def main() -> None:
    logger = get_logger("test_tree")
    logger.info("=== test_tree.py started ===")
    for sample_idx, (sample_name, _imgs, latex_formula) in enumerate(
        build_extreme_demo_samples(seed=7), start=1
    ):
        tree = tex2tree(latex_formula)
        gtd = node2list(tree)

        logger.info("--- Sample %s: %s ---", sample_idx, sample_name)
        logger.info("Cong thuc goc: %s", latex_formula)
        logger.info("Cay khong gian:")
        for line in tree_to_lines(tree):
            logger.info("%s", line)

        logger.info("Danh sach canh/quan he:")
        for token, idx, parent, parent_idx, relation in gtd:
            token_text = getattr(token, "token", str(token))
            parent_text = getattr(parent, "token", str(parent))
            logger.info(
                "node=%4s idx=%2s parent=%5s parent_idx=%2s relation=%s",
                token_text,
                idx,
                parent_text,
                parent_idx,
                relation,
            )
    logger.info("=== test_tree.py finished ===")


if __name__ == "__main__":
    main()
