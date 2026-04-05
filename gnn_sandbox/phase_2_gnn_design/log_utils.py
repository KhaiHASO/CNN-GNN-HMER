from __future__ import annotations

import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"


def get_logger(log_name: str) -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)

    logger_name = f"gnn_sandbox.phase2.{log_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_DIR / f"{log_name}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
