import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image

from data.utils import walk_path

class COCODataset(Dataset):

    def __init__(
            self,
            dataset_dir
            ) -> None:
        super().__init__()
        self.root_dir = os.path.join(dataset_dir)

        self.files = walk_path(
            os.path.join(dataset_dir), extension=".jpg")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.files[index]
        lbl_path = img_path.replace("img/", "semseg/").replace("img_front.jpg", "semseg_front.png")
        rgb = read_image(img_path) # shape (3, w, h)
        lbl = read_image(lbl_path)[0].long() # shape (w, h)

        return (rgb, lbl)
