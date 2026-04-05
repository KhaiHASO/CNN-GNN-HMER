from typing import Callable, Tuple

from phase_1_baseline_analysis.log_utils import get_logger


def run_step(logger_name: str, title: str, fn: Callable[[], None]) -> Tuple[str, bool]:
    logger = get_logger(logger_name)
    logger.info("=== %s ===", title)
    try:
        fn()
    except Exception:
        logger.exception("Step failed: %s", title)
        return title, False
    logger.info("Step completed: %s", title)
    return title, True


def run_data_scenario() -> None:
    from phase_1_baseline_analysis.test_data import main as run_data_test

    run_data_test()


def run_tree_scenario() -> None:
    from phase_1_baseline_analysis.test_tree import main as run_tree_test

    run_tree_test()


def run_graph_scenario() -> None:
    from phase_2_gnn_design.test_gnn_graph import run_graph_demo

    run_graph_demo()


def main() -> None:
    logger = get_logger("run_phase1")
    logger.info("=== Phase 1 started ===")
    logger.info("Phase name: Sandbox Decoupling & Pure Algorithm Prototyping")

    steps = [
        run_step("run_phase1", "Scenario 1 - Dataset to Encoder", run_data_scenario),
        run_step("run_phase1", "Scenario 2 - Latex to Spatial Tree", run_tree_scenario),
        run_step("run_phase1", "Scenario 3 - CNN Feature Map to Graph", run_graph_scenario),
    ]

    passed = sum(1 for _, ok in steps if ok)
    total = len(steps)
    logger.info("Phase 1 summary: %s/%s scenarios passed", passed, total)
    if passed != total:
        raise SystemExit(1)

    logger.info("=== Phase 1 finished ===")


if __name__ == "__main__":
    main()
