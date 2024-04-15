import random
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

from data.outlier_injection import OutlierInjection

class OutlierDataset(Dataset):

    def __init__(
            self,
            dataset: Dataset,
            outliers: Dataset,
            target_size: int = 256,
            scale: Tuple[float] = (.5, 1.5),
            horizon: float = 0,
            alpha_blend: float = 1,
            histogram_matching: bool = False,
            blur: int = 0
            ) -> None:
        super().__init__()

        self.dataset = dataset
        self.outlier_source = outliers
        self.horizon = horizon
        self.target_size = target_size
        self.scale = scale
        self.random_flip = T.RandomHorizontalFlip(p=.5)
        self.outlier_injection = OutlierInjection(
            alpha_blend = alpha_blend,
            histogram_matching = histogram_matching,
            blur = blur)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns rgb mixed content image, labels with outliers marked as ignore
        # and a outlier mask
        rgb, lbl = self.dataset[index]
        joint = torch.cat([rgb, lbl.unsqueeze(0)], dim=0)

        # scale the image from half size to 50% extra size, and take a random square crop
        scale = random.uniform(self.scale[0], self.scale[1])
        joint = torch.cat([rgb, lbl.unsqueeze(0)], dim=0)
        new_size = int(scale * min(joint.shape[1], joint.shape[2]))

        resized = T.functional.resize(joint, new_size, interpolation=T.InterpolationMode.NEAREST)
        resized = self.random_flip(resized)

        # get random crop
        x_index = random.randint(0, resized.shape[1] - self.target_size)
        y_index = random.randint(0, resized.shape[2] - self.target_size)

        # horizon position
        horizon = int(resized.shape[1] * self.horizon - y_index)

        resized = resized[:, x_index:x_index+self.target_size, y_index:y_index+self.target_size]

        # get the random synthetic outlier
        outlier, outlier_mask = self.outlier_source[random.randint(0, len(self.outlier_source)-1)]

        # put outlier in image, and return outlier mask
        rgb, lbl = self.outlier_injection(
            resized[:3], resized[3], outlier, outlier_mask, horizon)

        return (rgb, lbl)


class RandomCropFlipDataset(Dataset):

    def __init__(
            self,
            dataset: Dataset,
            target_size: int = 256,
            scale: Tuple[float] = (.5, 1.5),
            ) -> None:
        super().__init__()

        self.dataset = dataset
        self.target_size = target_size
        self.scale = scale
        self.random_flip = T.RandomHorizontalFlip(p=.5)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns rgb mixed content image, labels with outliers marked as ignore
        # and a outlier mask
        rgb, lbl = self.dataset[index]
        joint = torch.cat([rgb, lbl.unsqueeze(0)], dim=0)

        # scale the image from half size to 50% extra size, and take a random square crop
        scale = random.uniform(self.scale[0], self.scale[1])
        joint = torch.cat([rgb, lbl.unsqueeze(0)], dim=0)
        new_size = int(scale * min(joint.shape[1], joint.shape[2]))

        resized = T.functional.resize(joint, new_size, interpolation=T.InterpolationMode.NEAREST)
        resized = self.random_flip(resized)

        # get random crop
        x_index = random.randint(0, resized.shape[1] - self.target_size)
        y_index = random.randint(0, resized.shape[2] - self.target_size)

        resized = resized[:, x_index:x_index+self.target_size, y_index:y_index+self.target_size]

        rgb = resized[:3]
        lbl = resized[3]

        return (rgb, lbl)
