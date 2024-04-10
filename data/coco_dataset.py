import os
from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image
from pycocotools.coco import COCO

USED_CATEGORIES = [
    "outdoor", "animal", "accessory",
    "sports", "kitchen", "food",
    "furniture", "electronic", "appliance",
    "indoor"
]

class COCODataset(Dataset):

    def __init__(
            self,
            dataset_dir,
            sup_cats = None
            ) -> None:
        super().__init__()
        self.coco = COCO(
            os.path.join(dataset_dir, "annotations2014/annotations/instances_val2014.json"))
        self.img_dir = os.path.join(dataset_dir, "val2014/val2014")

        if not sup_cats is None:
            self.cat_ids = self.coco.getCatIds(supNms=sup_cats)
            self.img_ids = list(set(
                x for cat in self.cat_ids for x in self.coco.getImgIds(catIds=cat)
            ))
        else:
            self.cat_ids = self.coco.getCatIds()
            self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.coco.imgs[self.img_ids[index]]
        img_path = os.path.join(self.img_dir, img["file_name"])

        rgb = read_image(img_path) # shape (3, w, h)

        anns_id = self.coco.getAnnIds(imgIds=img["id"], catIds=self.cat_ids)
        anns = self.coco.loadAnns(anns_id)
        mask = self.coco.annToMask(random.choice(anns))

        nz = mask.nonzero()
        min_x = nz[0].min()
        max_x = nz[0].max()
        min_y = nz[1].min()
        max_y = nz[1].max()

        mask = mask[min_x:max_x, min_y:max_y]
        rgb = rgb[:, min_x:max_x, min_y:max_y]

        return (rgb, mask)
