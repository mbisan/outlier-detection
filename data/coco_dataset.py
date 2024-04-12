import os
from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image
import torchvision.transforms.v2 as T
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
            target_size,
            max_size,
            sup_cats = None
            ) -> None:
        super().__init__()
        self.coco = COCO(
            os.path.join(dataset_dir, "annotations2014/annotations/instances_val2014.json"))
        self.img_dir = os.path.join(dataset_dir, "val2014/val2014")
        self.size = target_size
        self.max_size = max_size

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
        # reads coco image and returns the image randomly placed on an image of shape size x size
        # and mask where the synthetic outlier is

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

        rgb = rgb[:, min_x:max_x, min_y:max_y]
        mask = torch.from_numpy(mask[min_x:max_x, min_y:max_y])

        joint = torch.cat([rgb, mask.unsqueeze(0)], dim=0) # shape (4, w, h)

        aspect_ratio = (max_x - min_x) / (max_y - min_y)
        resize_size = random.randint(int(self.max_size/4), self.max_size)
        if resize_size*aspect_ratio<resize_size:
            resize_size = int(aspect_ratio*resize_size)

        resized = T.functional.resize(joint, resize_size)
        # resized = self.random_resize(joint)

        rgb_final = torch.zeros((3, self.size, self.size), dtype=torch.uint8)
        mask_final = torch.zeros((self.size, self.size), dtype=torch.uint8)

        # random x position
        x_shape, y_shape = resized.shape[1], resized.shape[2]
        x_index = random.randint(0, self.size - x_shape)
        y_index = random.randint(0, self.size - y_shape)

        rgb_final[:, x_index:x_index+x_shape, y_index:y_index+y_shape] = resized[:3]
        mask_final[x_index:x_index+x_shape, y_index:y_index+y_shape] = resized[3]

        return (rgb_final, mask_final)

    def get_random_negative(self, horizon) -> Tuple[torch.Tensor, torch.Tensor]:
        # reads coco image and returns the image randomly placed on an image of shape size x size
        # and mask where the synthetic outlier is

        img = self.coco.imgs[random.choice(self.img_ids)]
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

        rgb = rgb[:, min_x:max_x, min_y:max_y]
        mask = torch.from_numpy(mask[min_x:max_x, min_y:max_y])

        joint = torch.cat([rgb, mask.unsqueeze(0)], dim=0) # shape (4, w, h)

        aspect_ratio = (max_x - min_x) / (max_y - min_y)
        resize_size = random.randint(int(self.max_size/4), self.max_size)
        if resize_size*aspect_ratio<resize_size:
            resize_size = int(aspect_ratio*resize_size)

        resized = T.functional.resize(joint, resize_size)
        # resized = self.random_resize(joint)

        rgb_final = torch.zeros((3, self.size, self.size), dtype=torch.uint8)
        mask_final = torch.zeros((self.size, self.size), dtype=torch.uint8)

        # random x position
        x_shape, y_shape = resized.shape[1], resized.shape[2]
        x_index = random.randint(0, self.size - x_shape)
        y_index = random.randint(min(horizon, self.size - y_shape - 1), self.size - y_shape)

        rgb_final[:, x_index:x_index+x_shape, y_index:y_index+y_shape] = resized[:3]
        mask_final[x_index:x_index+x_shape, y_index:y_index+y_shape] = resized[3]

        return (rgb_final, mask_final)
