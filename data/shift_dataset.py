import os
from typing import Tuple
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image

from data.utils import walk_path

ShiftLabel = namedtuple("ShiftLabel", ["id", "name", "rgb", "cityscapes", "ignore_in_eval"])

shift_labels = [
    ShiftLabel(0,'unlabeled',     ( 0, 0, 0),     0,  True    ),
    ShiftLabel(1,'building',      ( 70, 70, 70),  11, False   ),
    ShiftLabel(2,'fence',         (100, 40, 40),  13, False   ),
    ShiftLabel(3,'other',         ( 55, 90, 80),  0,  True    ),
    ShiftLabel(4,'pedestrian',    (220, 20, 60),  24, False   ),
    ShiftLabel(5,'pole',          (153, 153, 153),17, False   ),
    ShiftLabel(6,'road line',     (157, 234, 50), 7,  False   ),
    ShiftLabel(7,'road',          (128, 64, 128), 7,  False   ),
    ShiftLabel(8,'sidewalk',      (244, 35, 232), 8,  False   ),
    ShiftLabel(9,'vegetation',    (107, 142, 35), 21, False   ),
    ShiftLabel(10,'vehicle',      ( 0, 0, 142),   26, False   ),
    ShiftLabel(11,'wall',         (102, 102, 156),12, False   ),
    ShiftLabel(12,'traffic sign', (220, 220, 0),  20, False   ),
    ShiftLabel(13,'sky',          ( 70, 130, 180),23, False   ),
    ShiftLabel(14,'ground',       ( 81, 0, 81),   6,  True    ),
    ShiftLabel(15,'bridge',       (150, 100, 100),15, True    ),
    ShiftLabel(16,'rail track',   (230, 150, 140),10, True    ),
    ShiftLabel(17,'guard rail',   (180, 165, 180),14, True    ),
    ShiftLabel(18,'traffic light',(250, 170, 30), 19, False   ),
    ShiftLabel(19,'static',       (110, 190, 160),4,  True    ),
    ShiftLabel(20,'dynamic',      (170, 120, 50), 5,  True    ),
    ShiftLabel(21,'water',        ( 45, 60, 150), 0,  True    ),
    ShiftLabel(22,'terrain',      (145, 170, 100),22, False   ),
]

def process_file(file_array_tuple):
    filename, array_chunk = file_array_tuple
    lbl = read_image(filename)[0]
    unique, counts = lbl.unique(return_counts=True)
    array_chunk[unique.long()] = counts

def compute_per_class_counts(root_dir, split, save_dir):

    files = walk_path(
        os.path.join(root_dir, "discrete/images", split, "front/semseg"), extension=".png")

    counts_matrix = torch.zeros((len(files), len(shift_labels)), dtype=torch.int64)

    with ThreadPoolExecutor() as executor:
        results = executor.map(
            process_file, [(files[i], counts_matrix[i]) for i in range(len(files))])
    _ = list(results)

    df = pd.DataFrame(counts_matrix)
    files = [x.replace(root_dir + "/", "") for x in files]
    df["files"] = files

    df.to_csv(save_dir)

# the ignore_label is 100, while the ood_label is 101

shift_label_mapping = {
    "normal": torch.tensor(
        [100, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]).long(),
    "ood_pedestrian": torch.tensor(
        [100, 0, 1, 2, 101, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).long(),
}

LabelFilter = namedtuple("LabelFilter", ["label_id", "min_amount", "max_amount"])

no_pedestrian_filter = LabelFilter("4", -1, 0)
pedestrian_filter_10_15k = LabelFilter("4", 10000, 15000)

class ShiftDataset(Dataset):

    def __init__(
            self,
            dataset_dir,
            split,
            label_mapping = None,
            label_filter: LabelFilter = None,
            ) -> None:
        super().__init__()
        self.root_dir = os.path.join(dataset_dir, "discrete/images", split)

        if not label_filter is None:
            class_counts = pd.read_csv(os.path.join(dataset_dir, f"counts_{split}.csv"))
            counts = class_counts[label_filter.label_id].to_numpy()
            selected_ids = np.argwhere(
                (counts > label_filter.min_amount) & (counts <= label_filter.max_amount))
            self.files = [
                os.path.join(dataset_dir, x.item())
                for x in class_counts["files"].to_numpy()[selected_ids]
            ]
        else:
            self.files = walk_path(
                os.path.join(
                    dataset_dir, "discrete/images", split, "front/semseg"), extension=".png")

        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        lbl_path = self.files[index]
        img_path = lbl_path.replace("semseg/", "img/").replace("semseg_front.png", "img_front.jpg")
        rgb = read_image(img_path) # shape (3, w, h)
        lbl = read_image(lbl_path)[0].long() # shape (w, h)

        if not self.label_mapping is None:
            lbl = self.label_mapping[lbl]

        return (rgb, lbl)

if __name__ == "__main__":
    compute_per_class_counts("./datasets/SHIFT", "val", "./datasets/SHIFT/counts_val.csv")
    compute_per_class_counts("./datasets/SHIFT", "train", "./datasets/SHIFT/counts_train.csv")
