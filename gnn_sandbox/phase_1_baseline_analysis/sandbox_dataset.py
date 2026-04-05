from typing import List, Sequence, Tuple

import torch
from torch.utils.data.dataset import Dataset

from phase_1_baseline_analysis.log_utils import get_logger

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


class SimpleHMEDataset(Dataset):
    def __init__(self, ds: Sequence[Tuple[str, Sequence[torch.Tensor], str]]) -> None:
        super().__init__()
        self.ds = ds

    def __getitem__(self, idx: int):
        fname, imgs, caption = self.ds[idx]
        normalized = [self._to_image_tensor(im) for im in imgs]
        return fname, normalized, caption

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _to_image_tensor(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.dim() != 3:
            raise ValueError(f"Expected image tensor with 2 or 3 dims, got {image.dim()}")
        return image.float()


def build_dummy_megabatch(
    batch_size: int = 8,
    channels: int = 1,
    height: int = 128,
    width: int = 128,
) -> torch.Tensor:
    return torch.randn(batch_size, channels, height, width)


def build_dummy_mask(
    batch_size: int = 8, height: int = 128, width: int = 128
) -> torch.Tensor:
    return torch.ones(batch_size, height, width, dtype=torch.long)


def summarize_encoder_input(batch: torch.Tensor) -> str:
    return (
        "Dataset -> Encoder tensor shape: "
        f"{tuple(batch.shape)} = [batch, channel, height, width]"
    )


def summarize_tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float()
    return (
        f"{name} stats: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.4f}, "
        f"max={tensor.max().item():.4f}, "
        f"mean={tensor.mean().item():.4f}, "
        f"std={tensor.std(unbiased=False).item():.4f}"
    )


def build_extreme_demo_samples(
    seed: int = 7,
) -> List[Tuple[str, List[torch.Tensor], str]]:
    generator = torch.Generator().manual_seed(seed)

    return [
        # Mẫu 1: Nhị thức Newton (Đầy đủ Tổng Sigma, Chỉ số trên/dưới, Phân số, Giai thừa)
        # Bắt parser xử lý một nhánh right rất dài với nhiều cụm sub/sup liên tiếp.
        (
            "sample_newton_binomial",
            [torch.randn(200, 600, generator=generator)],
            r"( x + y ) ^ { n } = \sum _ { k = 0 } ^ { n } \frac { n ! } { k ! ( n - k ) ! } x ^ { n - k } y ^ { k }"
        ),
        
        # Mẫu 2: Tổng Sigma kép bọc ngoài một Ma trận nằm trong Căn bậc 3
        # Tổ hợp cực kỳ phức tạp: above/below (Sigma) -> leftup/inside (Căn) -> Mstart/nextline (Ma trận) -> sub/sup (Bên trong ma trận)
        (
            "sample_nested_sigma_matrix",
            [torch.randn(300, 500, generator=generator)],
            r"\sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \sqrt [ 3 ] { \begin{matrix} i ^ { 2 } & j \\ i + j & i - j \end{matrix} }"
        ),

        # Mẫu 3: Phân số liên tục (Continued Fraction) kết hợp Giới hạn (Limit)
        # Thử thách độ sâu (depth) của cây. Mỗi lần gọi \frac là cây lại đâm sâu thêm 1 tầng.
        (
            "sample_limit_continued_frac",
            [torch.randn(400, 400, generator=generator)],
            r"\lim _ { x \rightarrow \infty } \frac { 1 } { 1 + \frac { 1 } { x + \sqrt { x ^ { 2 } + 1 } } }"
        ),
        
        # Mẫu 4: Tích phân kép kết hợp Tổng Sigma và Đạo hàm riêng (Chuỗi Taylor/Maclaurin mở rộng)
        # Test khả năng đọc các ký hiệu lạ (\iint, \partial) và xử lý chỉ số k lặp lại liên tục.
        (
            "sample_calculus_monster",
            [torch.randn(250, 700, generator=generator)],
            r"\iint _ { D } \sum _ { k = 0 } ^ { \infty } \frac { \partial ^ { k } f } { \partial x ^ { k } } ( x - x _ { 0 } ) ^ { k } d x d y"
        )
    ]


def build_extreme_demo_batch(
    batch_size: int = 4,
    seed: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    samples = build_extreme_demo_samples(seed=seed)
    selected = [samples[i % len(samples)] for i in range(batch_size)]

    heights = [imgs[0].shape[-2] for _, imgs, _ in selected]
    widths = [imgs[0].shape[-1] for _, imgs, _ in selected]
    max_height = max(heights)
    max_width = max(widths)

    batch = torch.zeros(batch_size, 1, max_height, max_width)
    mask = torch.zeros(batch_size, max_height, max_width, dtype=torch.long)
    fnames = []
    captions = []

    for idx, (fname, imgs, caption) in enumerate(selected):
        image = SimpleHMEDataset._to_image_tensor(imgs[0])
        _, height, width = image.shape
        batch[idx, :, :height, :width] = image
        mask[idx, :height, :width] = 1
        fnames.append(fname)
        captions.append(caption)

    return batch, mask, fnames, captions

if __name__ == "__main__":
    logger = get_logger("sandbox_dataset")
    dataset = SimpleHMEDataset(build_extreme_demo_samples())
    fname, imgs, caption = dataset[0]
    logger.info("=== sandbox_dataset.py started ===")
    logger.info("fname: %s", fname)
    logger.info("num images: %s", len(imgs))
    logger.info("first image shape: %s", tuple(imgs[0].shape))
    logger.info("caption: %s", caption)
    logger.info("=== sandbox_dataset.py finished ===")
