import os
import json
from dataclasses import dataclass

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from utils.helper_functions import load_dataset
from utils.arguments import get_parser
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

def load_pretrained(dataset_name, lr=.00001, beta=.01, beta2=.001):
    # pylint: disable=no-value-for-parameter
    if dataset_name == "SHIFT":
        model = WrapperOod(backbone="resnet50", num_classes=21, lr=lr, beta=beta, beta2=beta2)
        model.load_state_dict(state_dict=torch.load("pretrained/shift_weights.ckpt"))
        return model
    elif dataset_name == "StreetHazards":
        model = WrapperOod(backbone="resnet50", num_classes=14, lr=lr, beta=beta, beta2=beta2)
        model.load_state_dict(state_dict=torch.load("pretrained/sh_weights.ckpt"))
        return model
    return None

def main(args: Arguments):

    dm = load_dataset(
        args.dataset, args.dataset_dir, args.horizon, args.alpha, args.histogram, args.blur)
    model = load_pretrained(args.dataset, args.lr, args.beta, args.beta2)

    tr = Trainer(
        default_root_dir=args.checkpoint, accelerator="cuda",
        callbacks=[
            ModelCheckpoint(),
            TQDMProgressBar(refresh_rate=2)
        ], max_epochs=args.epochs, check_val_every_n_epoch=100)

    tr.fit(model=model, datamodule=dm)

    model.ood_scores = []
    model.ood_masks = []

    out = tr.test(model=model, datamodule=dm)
    print(out)

    with open(os.path.join(args.checkpoint, "result.json"), "w") as f:
        json.dump({"results": out, "args": args.__dict__}, f)

if __name__ == "__main__":
    p = get_parser(Arguments)
    arg = p.parse_args()

    print(arg)
    main(arg)
