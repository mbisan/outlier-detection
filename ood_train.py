from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from utils.helper_functions import ShiftOODDataModule
from data.shift_dataset import pedestrian_filter_10_15k, no_pedestrian_filter
from nets.wrapperood import WrapperOod

dm = ShiftOODDataModule(
    "./datasets/SHIFT", 512, "./datasets/COCO2014", 256, 8, no_pedestrian_filter, pedestrian_filter_10_15k, "ood_pedestrian",
    horizon=0,
    alpha_blend=0,
    histogram_matching=False,
    num_workers=8, val_amount=.05)

# pylint: disable=no-value-for-parameter
model = WrapperOod.load_from_checkpoint(
    checkpoint_path="test_shift/lightning_logs/version_0/checkpoints/epoch=49-step=68350-val_miou=0.8422.ckpt",
    backbone = "resnet50",
    num_classes = 22,
    lr = .00001,
    beta = .01
)

ckpt = ModelCheckpoint(
        monitor="val_miou",
        mode="max",
        save_top_k=2,
        save_last=True,
        filename='{epoch}-{step}-{val_miou:.4f}'
    )

tr = Trainer(default_root_dir="./test_shift_ood", accelerator="cuda",
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=1)], max_epochs=5, check_val_every_n_epoch=10)

tr.fit(model=model, datamodule=dm)
tr.test(model=model, datamodule=dm)
