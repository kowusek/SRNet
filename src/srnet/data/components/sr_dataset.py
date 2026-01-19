import random
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """Super-Resolution dataset wrapper using torch-only interpolation.

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
        assert scale in (2, 4, 8), "scale must be 2, 4, or 8"

        if hr_crop_size is not None:
            assert hr_crop_size % scale == 0, "hr_crop_size must be divisible by scale"

        self.base_dataset = base_dataset
        self.scale = scale
        self.hr_crop_size = hr_crop_size
        self.interpolation = interpolation
        self.training = training

        self.valid_indices: List[int] = self._filter_valid_images()

    def _filter_valid_images(self) -> List[int]:
        """Keep only images large enough for the required HR crop size."""
        valid = []

        for idx in range(len(self.base_dataset)):
            item = self.base_dataset[idx]
            img = item[0] if isinstance(item, (tuple, list)) else item

            if isinstance(img, Image.Image):
                w, h = img.size
            elif torch.is_tensor(img):
                _, h, w = img.shape
            else:
                continue

            if self.hr_crop_size is None:
                # Only need divisibility by scale
                if h >= self.scale and w >= self.scale:
                    valid.append(idx)
            else:
                if h >= self.hr_crop_size and w >= self.hr_crop_size:
                    valid.append(idx)

        return valid

    def __len__(self):
        return len(self.valid_indices)

    def _extract_image(self, item):
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

        return img  # (C, H, W)

    def _crop_hr(self, hr: torch.Tensor) -> torch.Tensor:
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
        return torch.clamp(
            F.interpolate(
                hr.unsqueeze(0),
                size=(hr.shape[2] // self.scale, hr.shape[1] // self.scale),
                mode=self.interpolation,
                align_corners=False,
                antialias=True,
            ),
            0.0,
            1.0,
        ).squeeze(0)

    def __getitem__(self, idx):
        base_idx = self.valid_indices[idx]
        item = self.base_dataset[base_idx]

        hr = self._extract_image(item)
        hr = self._crop_hr(hr)
        lr = self._downsample(hr)

        return lr, hr
