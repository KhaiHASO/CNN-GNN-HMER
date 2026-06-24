from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
TAMER_ROOT = ROOT / "chuyende_tamer_temp" / "1-cnn-gnn"
DEFAULT_CKPT = ROOT / "chuyende_tamer_temp" / "KetQua" / "4_Coord_Aware_GAT_1L_4H" / "checkpoints" / "best_model.ckpt"
DEFAULT_DATA = TAMER_ROOT / "data" / "crohme"

if str(TAMER_ROOT) not in sys.path:
    sys.path.insert(0, str(TAMER_ROOT))

try:
    import torch
    from tamer.datamodule import vocab
    from tamer.datamodule.transforms import ScaleToLimitRange
    from tamer.lit_tamer import LitTAMER
except Exception as exc:  # pragma: no cover - surfaced at startup/request time
    torch = None
    vocab = None
    ScaleToLimitRange = None
    LitTAMER = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


app = FastAPI(title="Real M4 TAMER Server")
MODEL = None
DEVICE = None


def load_model():
    global MODEL, DEVICE
    if IMPORT_ERROR is not None:
        raise RuntimeError(f"Thiếu dependency để chạy M4 thật: {IMPORT_ERROR}")
    if MODEL is not None:
        return MODEL

    ckpt_path = Path(os.getenv("M4_CKPT_PATH", str(DEFAULT_CKPT)))
    data_dir = Path(os.getenv("M4_CROHME_DATA_DIR", str(DEFAULT_DATA)))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint M4: {ckpt_path}")
    dict_path = data_dir / "dictionary.txt"
    if not dict_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dictionary CROHME: {dict_path}")

    vocab.init(str(dict_path))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.getenv("M4_FORCE_CPU", "0") != "1" else "cpu")
    # Local trusted Lightning checkpoint. PyTorch >=2.6 defaults to
    # weights_only=True, which cannot load legacy Lightning callback metadata.
    MODEL = LitTAMER.load_from_checkpoint(str(ckpt_path), map_location=DEVICE, weights_only=False)
    MODEL.eval()
    MODEL.to(DEVICE)
    return MODEL


@app.on_event("startup")
def startup_load_model():
    # Load sớm để lỗi checkpoint/dependency hiện ngay khi start server.
    load_model()


@app.get("/health")
def health():
    return {
        "ok": MODEL is not None,
        "checkpoint": str(Path(os.getenv("M4_CKPT_PATH", str(DEFAULT_CKPT)))),
        "device": str(DEVICE),
    }


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    started = time.perf_counter()
    model = load_model()
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="File ảnh rỗng")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        tensor, mask = image_to_batch(tmp_path)
        with torch.no_grad():
            hyps = model.approximate_joint_search(tensor.to(DEVICE), mask.to(DEVICE))
        if not hyps:
            raise HTTPException(status_code=500, detail="M4 không trả hypothesis")
        best = hyps[0]
        tokens = vocab.indices2words(best.seq)
        latex = " ".join(tokens).strip()
        return {
            "latex": latex,
            "confidence": None,
            "score": float(best.score) if hasattr(best, "score") else None,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "device": str(DEVICE),
        }
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def image_to_batch(path: Path):
    gray = np.array(Image.open(path).convert("L"))
    # M4/CROHME input is black background with white foreground. If a white
    # background image is accidentally sent, invert it to match training data.
    if np.mean(gray == 255) > np.mean(gray == 0):
        gray = 255 - gray
    gray = np.where(gray > 127, 255, 0).astype(np.uint8)
    gray = ScaleToLimitRange(w_lo=16, w_hi=1024, h_lo=16, h_hi=256)(gray)
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    arr = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    mask = torch.zeros(1, tensor.shape[2], tensor.shape[3], dtype=torch.bool)
    return tensor, mask
