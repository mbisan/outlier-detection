# %%
import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io.image import read_image
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from data.utils import walk_path

# %%
from utils.helper_functions import load_shift_ood, load_shift_segmentation, load_streethazards_ood, load_streethazards_segmentation
from data.shift_dataset import LabelFilter

# %%
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from nets.wrapper import Wrapper

# %%
# ds_train, ds_val, ds_test = load_shift_segmentation(
#     "./datasets/SHIFT", 512, 6, LabelFilter("4", -1, 0), LabelFilter("4", -1, 0), "ood_pedestrian", 8, .05)
# print(len(ds_train.dataset), len(ds_val.dataset), len(ds_test.dataset))

# %%
# ds_train, ds_val, ds_test = load_shift_ood(
#     "./datasets/SHIFT", 512,
#     "./datasets/COCO2014", 256, 4, LabelFilter("4", -1, 0), LabelFilter("4", 10000, 15000), "ood_pedestrian",
#     .7, .9, True, 3, 1, .05)

# %%
# ds_train, ds_val, ds_test = load_streethazards_ood(
#     "./datasets/StreetHazards", 512,
#     "./datasets/COCO2014", 256, 4, "normal",
#     .7, .9, True, 3, 1)

# %%
ds_train, ds_val, ds_test = load_streethazards_segmentation(
    "./datasets/StreetHazards", 400, 8, "normal", 8)

# %%
model = Wrapper("resnet50", 13, .0001)

# %%
ckpt = ModelCheckpoint(
        monitor="val_miou",
        mode="max",
        save_top_k=2,
        filename='{epoch}-{step}-{val_miou:.4f}'
    )

# %%
tr = Trainer(default_root_dir="./test", accelerator="cuda", callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=5)], max_epochs=2)

# %%
tr.fit(model=model, train_dataloaders=ds_train, val_dataloaders=ds_val)


