import os
from dataclasses import dataclass
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from utils.helper_functions import ShiftOODDataModule, StreetHazardsOODDataModule
from data.shift_dataset import pedestrian_filter_10_15k, no_pedestrian_filter
from nets.wrapperood import WrapperOod

@dataclass
class Arguments:
    dataset_dir: str = "./datasets"
    dataset: str = ""
    horizon: float = 1.0
    alpha: float = 1.0
    blur: int = 0
    histogram: bool = False
    lr: float = .00001
    beta: float = .01
    beta2: float = .001
    checkpoint: str = "./"
    epochs: int = 5

def get_argparse():
    parser = ArgumentParser()

    for argument, typehint in Arguments.__annotations__.items():
        if typehint == bool:
            parser.add_argument(f"--{argument}",
                action="store_false" if Arguments.__dict__[argument] else "store_true")
        else:
            parser.add_argument(f"--{argument}",
                type=typehint, default=Arguments.__dict__[argument])

    return parser

def load_dataset(
        dataset_name, dataset_dir="./datasets",
        horizon=0, alpha_blend=1, histogram_matching=False, blur=0):
    if dataset_name == "SHIFT":
        return ShiftOODDataModule(
            os.path.join(dataset_dir, "SHIFT"), 512,
            os.path.join(dataset_dir, "COCO2014"), 352, 16,
            no_pedestrian_filter, pedestrian_filter_10_15k,
            "ood_pedestrian",
            horizon=horizon,
            alpha_blend=alpha_blend,
            histogram_matching=histogram_matching,
            blur=blur,
            num_workers=8, val_amount=.05
        )
    elif dataset_name == "StreetHazards":
        return StreetHazardsOODDataModule(
            os.path.join(dataset_dir, "StreetHazards"), 512,
            os.path.join(dataset_dir, "COCO2014"), 352, 16,
            "normal",
            horizon=horizon,
            alpha_blend=alpha_blend,
            histogram_matching=histogram_matching,
            blur=blur,
            num_workers=8
        )

    return None

def load_pretrained(dataset_name, lr=.00001, beta=.01, beta2=.001):
    # pylint: disable=no-value-for-parameter
    if dataset_name == "SHIFT":
        model = WrapperOod(backbone="resnet50", num_classes=21, lr=lr, beta=beta, beta2=beta2)
        model.load_from_checkpoint(checkpoint_path="pretrained/shift_weights.ckpt")
        return model
    elif dataset_name == "StreetHazards":
        model = WrapperOod(backbone="resnet50", num_classes=14, lr=lr, beta=beta, beta2=beta2)
        model.load_from_checkpoint(checkpoint_path="pretrained/sh_weights.ckpt")
        return model
    return None

def main(args: Arguments):

    dm = load_dataset(
        args.dataset, args.dataset_dir, args.horizon, args.alpha, args.histogram, args.blur)
    model = load_pretrained(args.dataset, args.lr, args.beta, args.beta2)

    tr = Trainer(
        default_root_dir=args.checkpoint, accelerator="cuda",
        callbacks=[
            ModelCheckpoint(save_last=True, filename='{epoch}-{step}'),
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=2)
        ], max_epochs=args.epochs, check_val_every_n_epoch=10)

    tr.fit(model=model, datamodule=dm)
    out = tr.test(model=model, datamodule=dm)
    print(out)

if __name__ == "__main__":
    p = get_argparse()
    arg = p.parse_args()

    print(arg)
    main(arg)
