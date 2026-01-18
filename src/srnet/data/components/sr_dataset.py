import random
from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


class SRDataset(Dataset):
    """
    Super-Resolution dataset wrapper using torch-only interpolation.

    Returns:
        lr: Tensor (C, H/scale, W/scale)
        hr: Tensor (C, H, W)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        scale: int = 4,
        hr_crop_size: Optional[int] = 256,
        interpolation: str = "bicubic",
        training: bool = True,
    ):
        """
        Args:
            base_dataset: Any dataset returning PIL image or (PIL image, target)
            scale: SR scale factor (2, 4, 8)
            hr_crop_size: HR crop size (divisible by scale). If None, full image.
            training: Random crop if True, else center crop
        """
        assert scale in (2, 4, 8), "scale must be 2, 4, or 8"

        if hr_crop_size is not None:
            assert hr_crop_size % scale == 0, "hr_crop_size must be divisible by scale"

        self.base_dataset = base_dataset
        self.scale = scale
        self.hr_crop_size = hr_crop_size
        self.interpolation = interpolation
        self.training = training

    def __len__(self):
        return len(self.base_dataset)

    def _extract_image(self, item):
        """Handle datasets returning (img) or (img, label)."""
        if isinstance(item, (tuple, list)):
            img = item[0]
        else:
            img = item

        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        elif torch.is_tensor(img):
            pass
        else:
            raise TypeError("Dataset must return PIL Image or Tensor")

        return img  # (C, H, W) in [0,1]

    def _crop_hr(self, hr: torch.Tensor) -> torch.Tensor:
        """Crop HR tensor."""
        _, h, w = hr.shape

        if self.hr_crop_size is None:
            h = h - (h % self.scale)
            w = w - (w % self.scale)
            return hr[:, :h, :w]

        th = tw = self.hr_crop_size

        if self.training:
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)
        else:
            top = (h - th) // 2
            left = (w - tw) // 2

        return hr[:, top : top + th, left : left + tw]

    def _downsample(self, hr: torch.Tensor) -> torch.Tensor:
        """Torch bicubic downsampling with antialias."""
        return F.interpolate(
            hr.unsqueeze(0),
            scale_factor=1 / self.scale,
            mode=self.interpolation,
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        hr = self._extract_image(item)
        hr = self._crop_hr(hr)
        lr = self._downsample(hr)

        return lr, hr
