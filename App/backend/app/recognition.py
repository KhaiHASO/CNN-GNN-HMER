from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

import requests

from .latex_tools import clean_latex, validate_latex_basic
from .models import ExpressionBox, now_iso
from .services import static_to_path


class RecognitionError(RuntimeError):
    pass


class M4Recognizer:
    def recognize(self, image_path: str) -> dict:
        raise NotImplementedError


class LocalHttpM4Recognizer(M4Recognizer):
    def __init__(self) -> None:
        self.url = os.getenv("M4_API_URL", "http://127.0.0.1:7860/recognize")
        self.timeout = float(os.getenv("M4_TIMEOUT_SECONDS", "120"))
        self.field = os.getenv("M4_IMAGE_FIELD", "image")

    def recognize(self, image_path: str) -> dict:
        with open(image_path, "rb") as handle:
            response = requests.post(self.url, files={self.field: (Path(image_path).name, handle, "image/png")}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        latex = payload.get("latex") or payload.get("pred") or payload.get("prediction")
        if not latex:
            raise RecognitionError("M4 response không có latex/pred")
        return {"latex": latex, "confidence": payload.get("confidence", payload.get("score"))}


class MockM4Recognizer(M4Recognizer):
    def recognize(self, image_path: str) -> dict:
        return {"latex": r"(a+b)^n=\sum_{k=0}^{n} C_n^k a^{n-k}b^k", "confidence": 0.99}


def recognizer_from_env() -> M4Recognizer:
    backend = os.getenv("M4_BACKEND", "local_http").lower()
    if backend == "local_http":
        return LocalHttpM4Recognizer()
    if backend == "mock":
        return MockM4Recognizer()
    raise RecognitionError(f"M4_BACKEND không hỗ trợ: {backend}")


def run_recognition(expr: ExpressionBox) -> ExpressionBox:
    if expr.status in ("rejected", "noise", "fragment") or expr.candidateType in ("noise", "fragment", "multiline_block", "uncertain"):
        expr.latexStatus = "model_error"
        expr.recognitionHistory.append(
            {
                "run_id": f"run_{uuid4().hex[:8]}",
                "model_name": "M4",
                "input_image_path": expr.normalizedUrl,
                "latex_raw": "",
                "status": "model_error",
                "error": "M4 chỉ nhận single_expression đã được duyệt",
                "created_at": now_iso(),
            }
        )
        return expr
    if not expr.normalizedUrl:
        raise RecognitionError("Expression chưa có normalized_crohme.png")
    image_path = static_to_path(expr.normalizedUrl)
    if not image_path.exists():
        raise RecognitionError("Không tìm thấy normalized image")

    started = time.perf_counter()
    expr.latexStatus = "running"
    try:
        result = recognizer_from_env().recognize(str(image_path))
        raw = result["latex"]
        clean = clean_latex(raw)
        validation = validate_latex_basic(clean)
        expr.latexRaw = raw
        expr.latexClean = clean
        expr.latexConfidence = result.get("confidence")
        expr.latexStatus = "ok" if validation.ok else "syntax_error"
        expr.recognitionHistory.append(
            {
                "run_id": f"run_{uuid4().hex[:8]}",
                "model_name": "M4",
                "input_image_path": str(image_path),
                "latex_raw": raw,
                "latex_clean": clean,
                "confidence": expr.latexConfidence,
                "status": expr.latexStatus,
                "error": validation.error,
                "created_at": now_iso(),
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        )
    except Exception as exc:
        expr.latexStatus = "model_error"
        expr.recognitionHistory.append(
            {
                "run_id": f"run_{uuid4().hex[:8]}",
                "model_name": "M4",
                "input_image_path": str(image_path),
                "latex_raw": "",
                "status": "model_error",
                "error": str(exc),
                "created_at": now_iso(),
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        )
    expr.updatedAt = now_iso()
    return expr
