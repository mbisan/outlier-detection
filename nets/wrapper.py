import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
import torchmetrics as tm

from nets.vainfDLV3p.modeling import (
    deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_hrnetv2_32,
    deeplabv3plus_mobilenet, deeplabv3plus_resnet34
)
from nets.sml import unnormalized_likelihood, max_logits

class Wrapper(LightningModule):

    def __init__(
            self,
            backbone: str = "resnet50",
            num_classes: int = 2,
            lr: float = .001,
            ood_scores = "max_logits",
            **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)

        if backbone=="resnet50":
            self.model = deeplabv3plus_resnet50(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="resnet34":
            self.model = deeplabv3plus_resnet34(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="resnet101":
            self.model = deeplabv3plus_resnet101(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="hrnetv2_32":
            self.model = deeplabv3plus_hrnetv2_32(num_classes=num_classes, pretrained_backbone=False)
        elif backbone=="mobilenet":
            self.model = deeplabv3plus_mobilenet(num_classes=num_classes, pretrained_backbone=True)

        self.jaccard_index = tm.JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=100)
        self.auroc = tm.AUROC(
            task="multiclass", num_classes=num_classes, thresholds=100,
            average="macro", ignore_index=100)
        self.average_precision = tm.AveragePrecision(
            task="multiclass", num_classes=num_classes, thresholds=100,
            average="macro", ignore_index=100)

        self.lr = lr

        self.ood_scores = []
        self.predictions = []
        self.save_predictions = False

        self.additional_loss = False

        if ood_scores == "max_logits":
            self.compute_ood_scores = max_logits
        elif ood_scores == "unnormalized_likelihood":
            self.compute_ood_scores = unnormalized_likelihood

    def training_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        img, label = batch
        # labels can have 101 label, or ood label, this wrapper ignores those labels
        label[label == 101] = 100 # change ood labels to ignore

        out = self.model(img.float()/255.0)

        loss = nn.functional.cross_entropy(out, label, ignore_index=100)

        self.log(
            "train_loss", loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)

        if self.additional_loss:
            logits_nwhc = out.permute((0, 2, 3, 1))
            n, c, w, h = out.shape
            logits_incorrect_nwhc = torch.ones_like(logits_nwhc, dtype=bool)
            logits_incorrect_nwhc.scatter_(3, label.unsqueeze(-1), 0)
            logits_incorrect = logits_nwhc[logits_incorrect_nwhc].reshape((n, w, h, c-1))

            mean_negatives = (logits_incorrect - logits_incorrect.mean(-1, keepdim=True)).square().sum(-1).mean()

            return loss.float() + 0.001 * mean_negatives

        return loss.float()

    def validation_test_step(self, batch) -> torch.Tensor:
        # pylint: disable=arguments-differ
        img, label = batch
        # labels can have 101 label, or ood label, this wrapper ignores those labels
        label[label == 101] = 100 # change ood labels to ignore

        out = self.model(img.float()/255.0)
        loss = nn.functional.cross_entropy(out, label, ignore_index=100)

        out_proba = out.softmax(dim=1)

        _, pred = out.max(dim=1)
        ood_scores = self.compute_ood_scores(out)

        self.jaccard_index.update(out_proba, label)
        # self.auroc.update(out_proba, label)
        # self.average_precision.update(out_proba, label)

        if self.save_predictions:
            self.ood_scores.append(ood_scores.detach().cpu().numpy())
            self.predictions.append(pred.detach().cpu().type(torch.uint8).numpy())

        return loss.float()

    def validation_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        loss = self.validation_test_step(batch)

        self.log(
            "val_loss", loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)
        self.log(
            "val_miou", self.jaccard_index, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        loss = self.validation_test_step(batch)

        self.log(
            "test_loss", loss, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "test_miou", self.jaccard_index, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        """ Configure the optimizers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 
                    mode="max", factor=np.sqrt(0.1), patience=2, min_lr=1e-10),
                "interval": "epoch",
                "monitor": "val_miou",
                "frequency": 1
            },
        }
