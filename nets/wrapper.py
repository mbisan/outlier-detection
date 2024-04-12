import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
import torchmetrics as tm

from nets.vainfDLV3p.modeling import (
    deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_hrnetv2_32,
    deeplabv3plus_mobilenet
)

class Wrapper(LightningModule):

    def __init__(
            self,
            backbone: str = "resnet50",
            num_classes: int = 2,
            lr: float = .001,
            **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)

        if backbone=="resnet50":
            self.model = deeplabv3plus_resnet50(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="resnet101":
            self.model = deeplabv3plus_resnet101(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="hrnetv2_32":
            self.model = deeplabv3plus_hrnetv2_32(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="mobilenet":
            self.model = deeplabv3plus_mobilenet(num_classes=num_classes, pretrained_backbone=True)

        self.jaccard_index = tm.JaccardIndex(
            task="multiclass", num_classes=num_classes, ignore_index=100)
        self.auroc = tm.AUROC(
            task="multiclass", num_classes=num_classes, thresholds=200,
            average="macro", ignore_index=100)
        self.average_precision = tm.AveragePrecision(
            task="multiclass", num_classes=num_classes, thresholds=200,
            average="macro", ignore_index=100)

        self.lr = lr

    def training_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        img, label = batch
        # labels can have 101 label, or ood label, this wrapper ignores those labels
        label[label == 101] = 100 # change ood labels to ignore

        out = self.model(img.float()/255.0)

        loss = nn.functional.cross_entropy(out, label, ignore_index=100)

        self.jaccard_index.update(out, label)
        self.log(
            "train_loss", loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_test_step(self, batch) -> torch.Tensor:
        # pylint: disable=arguments-differ
        img, label = batch
        # labels can have 101 label, or ood label, this wrapper ignores those labels
        label[label == 101] = 100 # change ood labels to ignore

        out = self.model(img.float()/255.0)
        loss = nn.functional.cross_entropy(out, label, ignore_index=100)

        self.jaccard_index.update(out, label)
        self.auroc.update(out, label)
        self.average_precision.update(out, label)

        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        loss = self.validation_test_step(batch)

        self.log(
            "val_loss", loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)
        self.log(
            "val_miou", self.jaccard_index, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "val_auroc", self.auroc, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "val_ap", self.average_precision, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        loss = self.validation_test_step(batch)

        self.log(
            "test_loss", loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)
        self.log(
            "test_miou", self.jaccard_index, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "test_auroc", self.auroc, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "test_ap", self.average_precision, on_epoch=True,
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
