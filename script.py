from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from utils.helper_functions import ShiftSegmentationDataModule
from data.shift_dataset import LabelFilter
from nets.wrapper import Wrapper

# dm = StreetHazardsDataModule(
#     "./datasets/StreetHazards", 512, 8, "normal", 8)

dm = ShiftSegmentationDataModule(
    "./datasets/SHIFT", 512, 6, LabelFilter("4", -1, 0), LabelFilter("4", -1, 0), "ood_pedestrian", 8, .05)

model = Wrapper("resnet50", 22, .0001)

ckpt = ModelCheckpoint(
        monitor="val_miou",
        mode="max",
        save_top_k=4,
        filename='{epoch}-{step}-{val_miou:.4f}'
    )

tr = Trainer(default_root_dir="./test_shift", accelerator="cuda",
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=1)], max_epochs=50)

tr.fit(model=model, datamodule=dm)
