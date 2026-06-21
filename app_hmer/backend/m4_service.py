import os
import re
from threading import Lock
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from tamer.datamodule import vocab
from tamer.model.tamer import TAMER
from crohme_normalizer import CROHMENormalizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "chuyende_tamer_temp",
    "KetQua",
    "4_Coord_Aware_GAT_1L_4H",
    "checkpoints",
    "best_model.ckpt",
)
DICTIONARY_PATH = os.path.join(BASE_DIR, "tamer", "dictionary.txt")


class M4Service:
    """Dịch vụ suy luận cho mô hình M4 Coordinate-Aware GAT."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.checkpoint_path = os.environ.get(
            "M4_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH
        )
        self.model = None
        self.normalizer = CROHMENormalizer(target_height=128, width_multiple=16)
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def checkpoint_available(self) -> bool:
        return os.path.isfile(self.checkpoint_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return

        with self._load_lock:
            if self.loaded:
                return
            if not self.checkpoint_available:
                raise FileNotFoundError(
                    f"Không tìm thấy checkpoint M4 tại: {self.checkpoint_path}"
                )

            vocab.init(DICTIONARY_PATH)
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            hyperparameters: Dict = checkpoint["hyper_parameters"]
            model_keys = {
                "d_model",
                "growth_rate",
                "num_layers",
                "nhead",
                "num_decoder_layers",
                "dim_feedforward",
                "dropout",
                "dc",
                "cross_coverage",
                "self_coverage",
                "vocab_size",
                "use_gat",
                "gat_num_layers",
                "gat_num_heads",
                "gat_hidden_dim",
                "gat_dropout",
            }
            model_args = {
                key: value
                for key, value in hyperparameters.items()
                if key in model_keys
            }

            model = TAMER(**model_args)
            state_dict = {
                key.removeprefix("tamer_model."): value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith("tamer_model.")
            }
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            self.model = model

    @staticmethod
    def _prepare_image(image: Image.Image) -> torch.Tensor:
        grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

        # Chuẩn hóa ảnh chụp về định dạng CROHME: nền đen, nét viết trắng.
        if float(grayscale.mean()) > 127:
            normalized = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
        else:
            normalized = cv2.threshold(
                grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        normalized = cv2.morphologyEx(
            normalized, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )

        non_empty = np.argwhere(normalized > 8)
        if non_empty.size:
            y_min, x_min = non_empty.min(axis=0)
            y_max, x_max = non_empty.max(axis=0) + 1
            margin = max(8, int(min(normalized.shape) * 0.04))
            y_min = max(0, y_min - margin)
            x_min = max(0, x_min - margin)
            y_max = min(normalized.shape[0], y_max + margin)
            x_max = min(normalized.shape[1], x_max + margin)
            normalized = normalized[y_min:y_max, x_min:x_max]

        height, width = normalized.shape
        if height == 0 or width == 0:
            raise ValueError("Ảnh không chứa nội dung có thể nhận dạng.")

        min_height, max_height = 16, 256
        min_width, max_width = 16, 1024
        scale = min(max_height / height, max_width / width, 1.0)
        if scale < 1.0:
            normalized = cv2.resize(
                normalized,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            scale = max(min_height / height, min_width / width, 1.0)
            if scale > 1.0:
                normalized = cv2.resize(
                    normalized,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR,
                )

        tensor = torch.from_numpy(normalized.copy()).float().div_(255.0)
        return tensor.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def segment_formula_regions(image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Tách ảnh nhiều dòng thành các vùng công thức bằng phép chiếu ngang."""
        grayscale = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        if float(grayscale.mean()) < 127:
            grayscale = 255 - grayscale

        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        height, width = binary.shape

        # Loại nhiễu nhỏ, sau đó nối các nét thuộc cùng một dòng công thức.
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        joined = cv2.dilate(
            binary,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(9, width // 120), 5)
            ),
            iterations=1,
        )
        row_ink = np.count_nonzero(joined, axis=1)
        active_rows = row_ink > max(3, int(width * 0.002))

        spans = []
        start = None
        for row, active in enumerate(active_rows):
            if active and start is None:
                start = row
            elif not active and start is not None:
                spans.append([start, row])
                start = None
        if start is not None:
            spans.append([start, height])

        # Gộp các phần tử trên/dưới như số mũ, cận tích phân vào dòng gần nhất.
        merged = []
        max_gap = max(8, height // 100)
        for span in spans:
            if merged and span[0] - merged[-1][1] <= max_gap:
                merged[-1][1] = span[1]
            else:
                merged.append(span)

        regions = []
        pad_x = max(10, width // 100)
        pad_y = max(8, height // 100)
        for y1, y2 in merged:
            if y2 - y1 < max(12, height // 80):
                continue
            if y1 == 0 and y2 - y1 < height * 0.08:
                # Thường là mép giấy hoặc đường kẻ sát đầu ảnh.
                continue
            band = binary[y1:y2]
            columns = np.where(np.count_nonzero(band, axis=0) > 0)[0]
            if columns.size == 0:
                continue
            x1, x2 = int(columns[0]), int(columns[-1] + 1)
            if x2 - x1 < max(30, width // 30):
                continue
            regions.append(
                (
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(width, x2 + pad_x),
                    min(height, y2 + pad_y),
                )
            )

        # Với ảnh đã được người dùng cắt sát, giữ nguyên toàn bộ vùng.
        if not regions:
            return [(0, 0, width, height)]
        return regions[:12]

    @staticmethod
    def latex_quality(text: str) -> float:
        """Ước lượng chất lượng để phát hiện chuỗi lặp hoặc LaTeX hỏng."""
        if not text or not text.strip():
            return -100.0

        tokens = text.split()
        score = 8.0
        if len(tokens) > 120:
            score -= 6.0
        elif len(tokens) > 70:
            score -= 3.0

        if tokens:
            unique_ratio = len(set(tokens)) / len(tokens)
            score += unique_ratio * 5.0
            most_common = max(tokens.count(token) for token in set(tokens))
            if most_common / len(tokens) > 0.35:
                score -= 8.0

        repeated = re.search(r"(.{3,25})(?:\s+\1){3,}", text)
        if repeated:
            score -= 8.0
        if text.count("{") != text.count("}"):
            score -= 3.0
        if re.search(r"(?:\\leq|=|E|n)(?:\s+(?:\\leq|=|E|n)){10,}", text):
            score -= 12.0
        return score

    def predict(self, image: Image.Image) -> Dict:
        self.load()
        variants = self.normalizer.normalize(image)
        candidates = []
        for variant in variants:
            prediction = self.predict_normalized(variant.image)
            model_score = prediction["score"]
            final_score = model_score + variant.quality["score"]
            candidates.append(
                {
                    **prediction,
                    "variant": variant.name,
                    "normalized_image": variant.image,
                    "quality": variant.quality,
                    "final_score": final_score,
                }
            )
        return max(candidates, key=lambda item: item["final_score"])

    def predict_normalized(self, image: Image.Image) -> Dict:
        self.load()
        normalized = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        image_tensor = (
            torch.from_numpy(normalized.copy())
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        _, _, height, width = image_tensor.shape
        image_mask = torch.zeros(
            (1, height, width), dtype=torch.bool, device=self.device
        )

        with self._inference_lock, torch.inference_mode():
            hypotheses = self.model.beam_search(
                image_tensor,
                image_mask,
                beam_size=10,
                max_len=150,
                alpha=1.0,
                early_stopping=False,
                temperature=1.0,
            )

        hypothesis = hypotheses[0]
        tokens = vocab.indices2words(hypothesis.seq)
        latex = " ".join(tokens).replace("<space>", " ").strip()
        score = float(hypothesis.score.detach().cpu())
        return {"latex": latex, "score": score}
