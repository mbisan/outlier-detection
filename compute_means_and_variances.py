import os
import random
from dataclasses import dataclass

import torch
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from utils.helper_functions import ShiftSegmentationDataModule, StreetHazardsDataModule
from utils.arguments import get_parser
from nets.wrapper import Wrapper
from data.shift_dataset import LabelFilter

@dataclass
class Arguments:
    dataset_dir: str = "./datasets"
    dataset: str = ""
    checkpoint: str = "./"
    ood_scores: str = "max_logits"

def load_model(checkpoint_dir, ood_scores = "max_logits"):
    # pylint: disable=no-value-for-parameter

    loaded = torch.load(checkpoint_dir, map_location="cuda" if torch.cuda.is_available() else "cpu")
    num_classes = loaded["model.classifier.classifier.3.weight"].shape[0]
    model = Wrapper("resnet50", num_classes, ood_scores=ood_scores)
    model.load_state_dict(loaded)

    return model, num_classes

def main(args: Arguments):
    if args.dataset == "SHIFT":
        # this dataset contains no pedestrians in test
        dm = ShiftSegmentationDataModule(
            "./datasets/SHIFT", 512, 2, LabelFilter("4", -1, 0), LabelFilter("4", -1, 0), "ood_pedestrian", 8, .05)
        random.seed(42)
        dm.shift_test.files = random.choices(dm.shift_test.files, k=1000)
    elif args.dataset == "StreetHazards":
        # this dataset contains no OOD in Val
        dm = StreetHazardsDataModule(
            "./datasets/StreetHazards", 512, 2, "normal", 8)

    model, num_classes = load_model(args.checkpoint, args.ood_scores)
    model.save_predictions = True

    tr = Trainer(
        default_root_dir=os.path.dirname(args.checkpoint), accelerator="auto",
        callbacks=[TQDMProgressBar(refresh_rate=2)])

    if args.dataset == "SHIFT":
        out = tr.test(model=model, datamodule=dm)
    if args.dataset == "StreetHazards":
        out = tr.validate(model=model, datamodule=dm)

    model.predictions = np.concatenate(model.predictions)
    model.ood_scores = np.concatenate(model.ood_scores)

    # compute means and variances
    means = np.zeros(num_classes, dtype=np.float64)
    variances = np.zeros(num_classes, dtype=np.float64)
    for i in range(num_classes):
        class_mask = model.predictions == i
        print(np.sum(class_mask))
        if np.sum(class_mask) > 1:
            means[i] = np.mean(model.ood_scores[class_mask])
            variances[i] = np.var(model.ood_scores[class_mask])
        else:
            means[i] = 0
            variances[i] = 1.0

    with open(args.checkpoint.replace(".ckpt", f"{args.ood_scores}.means"), "wb") as f:
        np.save(f, means)
    with open(args.checkpoint.replace(".ckpt", f"{args.ood_scores}.var"), "wb") as f:
        np.save(f, variances)

    print(means)
    print(variances)

    print(out)

if __name__ == "__main__":
    p = get_parser(Arguments)
    arg = p.parse_args()

    print(arg)
    main(arg)
