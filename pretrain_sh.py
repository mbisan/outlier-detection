'''
Script to pretrain on the StreetHazards dataset for 50 epochs
'''

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from utils.helper_functions import StreetHazardsDataModule
from nets.wrapper import Wrapper

dm = StreetHazardsDataModule(
    "./datasets/StreetHazards", 512, 16, "normal", 8)

model = Wrapper("resnet50", 14, .0001)

ckpt = ModelCheckpoint(
        monitor="val_miou",
        mode="max",
        save_top_k=2,
        save_last=True,
        filename='{epoch}-{step}-{val_miou:.4f}'
    )

tr = Trainer(default_root_dir="./test_sh", accelerator="cuda",
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=1)], max_epochs=50)

tr.fit(model=model, datamodule=dm)
