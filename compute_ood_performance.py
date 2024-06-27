import os
import json
from dataclasses import dataclass

import torch
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from utils.helper_functions import load_dataset, ShiftSegmentationDataModule, StreetHazardsDataModule
from utils.arguments import get_parser
from nets.wrapperood import WrapperOod
from nets.sml import max_logits, unnormalized_likelihood, SMLWithPostProcessing
from data.shift_dataset import LabelFilter

@dataclass
class Arguments:
    dataset_dir: str = "./datasets"
    dataset: str = ""
    checkpoint_dir: str = "./"
    ood_scores: str = "max_logits"
    boundary_suppression: bool = True
    boundary_width: int = 4
    boundary_iteration: int = 4
    dilated_smoothing: bool = True
    kernel_size: int = 7
    dilation: int = 6

def load_model(checkpoint_dir):
    # pylint: disable=no-value-for-parameter

    loaded = torch.load(checkpoint_dir, map_location="cuda" if torch.cuda.is_available() else "cpu")
    num_classes = loaded["model.classifier.classifier.3.weight"].shape[0]
    model = WrapperOod("resnet50", num_classes)
    model.load_state_dict(loaded)

    return model, num_classes

def main(args: Arguments):
    if args.finetuned:
        dm = load_dataset(
            args.dataset, args.dataset_dir, 0, 1, False, 0)
    else:
        if args.dataset == "SHIFT":
            dm = ShiftSegmentationDataModule(
                "./datasets/SHIFT", 512, 16, LabelFilter("4", -1, 0), LabelFilter("4", -1, 0), "ood_pedestrian", 8, .05)
        elif args.dataset == "StreetHazards":
            dm = StreetHazardsDataModule(
                "./datasets/StreetHazards", 512, 16, "normal", 8)

    model = load_model(args.checkpoint)

    if args.ood_scores == "max_logits":
        model.compute_ood_scores = max_logits
    elif args.ood_scores == "unnormalized_likelihood":
        model.compute_ood_scores = unnormalized_likelihood
    elif args.ood_scores == "sml_ml":
        means = np.load(args.checkpoint_dir.replace(".ckpt", "max_logits.means"))
        variances = np.load(args.checkpoint_dir.replace(".ckpt", "max_logits.var"))
        model.compute_ood_scores = SMLWithPostProcessing(
            means, np.sqrt(variances), "max_logits",
            args.boundary_suppression, args.boundary_width, args.boundary_iteration,
            args.dilated_smoothing, args.kernel_size, args.dilation)
    elif args.ood_scores == "sml_ul":
        means = np.load(args.checkpoint_dir.replace(".ckpt", "unnormalized_likelihood.means"))
        variances = np.load(args.checkpoint_dir.replace(".ckpt", "unnormalized_likelihood.var"))
        model.compute_ood_scores = SMLWithPostProcessing(
            means, np.sqrt(variances), "unnormalized_likelihood",
            args.boundary_suppression, args.boundary_width, args.boundary_iteration,
            args.dilated_smoothing, args.kernel_size, args.dilation)

    tr = Trainer(
        default_root_dir=args.checkpoint, accelerator="cuda",
        callbacks=[TQDMProgressBar(refresh_rate=2)])

    out = tr.test(model=model, datamodule=dm)
    print(out)

    with open(os.path.join(os.path.dirname(args.checkpoint_dir), "result.json"), "w") as f:
        json.dump({"results": out, "args": args.__dict__}, f)

if __name__ == "__main__":
    p = get_parser(Arguments)
    arg = p.parse_args()

    print(arg)
    main(arg)
