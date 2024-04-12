import os
from typing import Tuple
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image

from data.utils import walk_path

StreetHazardsLabel = namedtuple("StreetHazardsLabel", ["id", "name", "rgb"])

streethazards_labels = ( # id (in files), name, color
    StreetHazardsLabel(0, "unlabeled", (0,0,0)),
    StreetHazardsLabel(1, "background", (0,0,0)),
    StreetHazardsLabel(2, "building", ( 70,  70,  70)),
    StreetHazardsLabel(3, "fence", (190, 153, 153)),
    StreetHazardsLabel(4, "other", (250, 170, 160)),
    StreetHazardsLabel(5, "pedestrian", (220,  20,  60)),
    StreetHazardsLabel(6, "pole", (153, 153, 153)),
    StreetHazardsLabel(7, "road line", (157, 234,  50)),
    StreetHazardsLabel(8, "road", (128,  64, 128)),
    StreetHazardsLabel(9, "sidewalk", (244,  35, 232)),
    StreetHazardsLabel(10, "vegentation", (107, 142,  35)),
    StreetHazardsLabel(11, "car", (  0,   0, 142)),
    StreetHazardsLabel(12, "wall", (102, 102, 156)),
    StreetHazardsLabel(13, "traffic sign", (220, 220,   0)),
    StreetHazardsLabel(14, "anomaly", ( 60, 250, 240)),
    StreetHazardsLabel(-1, "ignore", (0,0,0)),
)

# the ignore_label is 100, while the ood_label is 101

streethazards_label_mapping = {
    "normal": torch.tensor(
        [100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 101, 100]).long()
}

class StreetHazardsDataset(Dataset):

    def __init__(
            self,
            dataset_dir,
            split,
            label_mapping = None
            ) -> None:
        super().__init__()
        if split in ["train", "val"]:
            dataset_dir = os.path.join(dataset_dir, "streethazards_train/train")
        else:
            dataset_dir = os.path.join(dataset_dir, "streethazards_test/test")

        if split == "train":
            split = "training"
        elif split == "val":
            split = "validation"

        self.root_dir = os.path.join(dataset_dir, "annotations", split)
        self.files = walk_path(self.root_dir, extension=".png")

        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        lbl_path = self.files[index]
        img_path = lbl_path.replace("annotations/", "images/")
        rgb = read_image(img_path)[:3] # shape (3, w, h)
        lbl = read_image(lbl_path)[0].long() # shape (w, h)

        if not self.label_mapping is None:
            lbl = self.label_mapping[lbl]

        return (rgb, lbl)
