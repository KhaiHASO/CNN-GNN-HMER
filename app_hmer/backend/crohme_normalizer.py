from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class NormalizedVariant:
    name: str
    image: Image.Image
    quality: Dict


class CROHMENormalizer:
    """Đưa ảnh biểu thức đời thực về phân phối ảnh gần với CROHME."""

    def __init__(
        self,
        target_height: int = 128,
        width_multiple: int = 16,
        max_width: int = 1024,
    ) -> None:
        self.target_height = target_height
        self.width_multiple = width_multiple
        self.max_width = max_width

    @staticmethod
    def _remove_background(gray: np.ndarray) -> np.ndarray:
        sigma = max(15, min(gray.shape) // 8)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        normalized = cv2.divide(gray, background, scale=255)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(normalized)

    @staticmethod
    def _binary_candidates(clean_gray: np.ndarray) -> Dict[str, np.ndarray]:
        otsu = cv2.threshold(
            clean_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        adaptive = cv2.adaptiveThreshold(
            clean_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )
        return {"otsu": otsu, "adaptive": adaptive}

    @staticmethod
    def _remove_noise(binary: np.ndarray) -> np.ndarray:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 1:
            return binary

        components = stats[1:]
        areas = components[:, cv2.CC_STAT_AREA]
        heights = components[:, cv2.CC_STAT_HEIGHT]
        median_area = max(1.0, float(np.median(areas)))
        median_height = max(1.0, float(np.median(heights)))
        cleaned = np.zeros_like(binary)

        for label, stat in enumerate(components, start=1):
            x, y, width, height, area = map(int, stat)
            is_border_artifact = (
                (x == 0 or y == 0 or x + width >= binary.shape[1] or y + height >= binary.shape[0])
                and area > 12 * median_area
            )
            is_tiny = area < max(2, 0.015 * median_area) and height < 0.2 * median_height
            if not is_border_artifact and not is_tiny:
                cleaned[labels == label] = 255
        return cleaned

    @staticmethod
    def _tight_crop(binary: np.ndarray) -> np.ndarray:
        points = np.argwhere(binary > 0)
        if not points.size:
            raise ValueError("Không tìm thấy nét viết trong vùng ảnh.")
        y1, x1 = points.min(axis=0)
        y2, x2 = points.max(axis=0) + 1
        return binary[y1:y2, x1:x2]

    def _pad_resize(self, foreground_white: np.ndarray) -> np.ndarray:
        height, width = foreground_white.shape
        pad_top = max(8, int(0.18 * height))
        pad_bottom = max(8, int(0.18 * height))
        pad_left = max(8, int(0.08 * width))
        pad_right = max(8, int(0.08 * width))
        padded = cv2.copyMakeBorder(
            foreground_white,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        scale = self.target_height / padded.shape[0]
        target_width = max(1, int(round(padded.shape[1] * scale)))
        target_height = self.target_height
        if target_width > self.max_width:
            scale = self.max_width / padded.shape[1]
            target_width = self.max_width
            target_height = max(1, int(round(padded.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(
            padded, (target_width, target_height), interpolation=interpolation
        )
        final_width = (
            (target_width + self.width_multiple - 1) // self.width_multiple
        ) * self.width_multiple
        final = np.zeros((self.target_height, final_width), dtype=np.uint8)
        offset_x = (final_width - target_width) // 2
        offset_y = (self.target_height - target_height) // 2
        final[
            offset_y : offset_y + target_height,
            offset_x : offset_x + target_width,
        ] = resized
        return final

    @staticmethod
    def quality_check(image: np.ndarray) -> Dict:
        foreground = image > 8
        foreground_ratio = float(foreground.mean())
        touches_border = bool(
            foreground[0].any()
            or foreground[-1].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        )
        component_count = max(
            0, cv2.connectedComponents((foreground.astype(np.uint8) * 255), 8)[0] - 1
        )
        aspect_ratio = float(image.shape[1] / image.shape[0])

        if foreground_ratio < 0.003:
            status = "empty_or_too_light"
        elif foreground_ratio > 0.35:
            status = "too_dark_or_bad_threshold"
        elif touches_border:
            status = "need_expand_bbox"
        elif aspect_ratio > 14:
            status = "possibly_multiple_expressions"
        else:
            status = "ok"

        quality_score = 1.0
        quality_score -= min(0.5, abs(foreground_ratio - 0.08) * 2.5)
        if touches_border:
            quality_score -= 0.25
        if status not in {"ok", "possibly_multiple_expressions"}:
            quality_score -= 0.35
        return {
            "foreground_ratio": foreground_ratio,
            "touch_border": touches_border,
            "aspect_ratio": aspect_ratio,
            "component_count": component_count,
            "status": status,
            "score": max(0.0, quality_score),
            "size": [int(image.shape[1]), int(image.shape[0])],
        }

    def normalize(self, image: Image.Image) -> List[NormalizedVariant]:
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
        clean_gray = self._remove_background(gray)
        variants = []
        for name, binary in self._binary_candidates(clean_gray).items():
            cleaned = self._remove_noise(binary)
            try:
                cropped = self._tight_crop(cleaned)
            except ValueError:
                continue
            normalized = self._pad_resize(cropped)
            quality = self.quality_check(normalized)
            variants.append(
                NormalizedVariant(
                    name=name,
                    image=Image.fromarray(normalized, mode="L"),
                    quality=quality,
                )
            )
        if not variants:
            raise ValueError("Không thể tạo ảnh chuẩn hóa CROHME từ vùng đã chọn.")
        return variants
