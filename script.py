from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from utils.helper_functions import StreetHazardsDataModule
# from data.shift_dataset import LabelFilter
from nets.wrapper import Wrapper

dm = StreetHazardsDataModule(
    "./datasets/StreetHazards", 512, 8, "normal", 8)

model = Wrapper("resnet50", 13, .0001)

ckpt = ModelCheckpoint(
        monitor="val_miou",
        mode="max",
        save_top_k=2,
        filename='{epoch}-{step}-{val_miou:.4f}'
    )

tr = Trainer(default_root_dir="./test", accelerator="cuda",
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=5)], max_epochs=20)

tr.fit(model=model, datamodule=dm)
