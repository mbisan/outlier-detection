import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics as tm
from sklearn.metrics import roc_curve, average_precision_score, auc

from nets.vainfDLV3p.modeling import (
    deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_hrnetv2_32,
    deeplabv3plus_mobilenet, deeplabv3plus_resnet34
)

def calculate_auroc(gt, conf):
    fpr, tpr, threshold = roc_curve(gt, conf)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    k = -float("inf")
    for i, j, k in zip(tpr, fpr, threshold):
        if i > 0.95:
            fpr_best = j
            break
    return roc_auc, fpr_best, k

class WrapperOod(LightningModule):

    def __init__(
            self,
            backbone: str = "resnet50",
            num_classes: int = 2,
            lr: float = .001,
            beta: float = .0001,
            **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

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
        self.seg_auroc = tm.AUROC(
            task="multiclass", num_classes=num_classes, thresholds=200,
            average="macro", ignore_index=100)
        self.auroc = tm.AUROC(
            task="binary", num_classes=num_classes, thresholds=200,
            average="macro", ignore_index=100)
        self.average_precision = tm.AveragePrecision(
            task="binary", num_classes=num_classes, thresholds=200,
            average="macro", ignore_index=100)

        self.lr = lr
        self.beta = beta

        self.ood_scores = []
        self.ood_masks = []

    def ood_loss(self, logits: torch.Tensor, ood_mask: torch.Tensor):
        lse = torch.logsumexp(logits, dim=1) - logits.mean(dim=1).detach()
        loss_ood = lse[ood_mask].sum() / ood_mask.sum()
        loss_ind = lse[~ood_mask].sum() / (~ood_mask).sum()
        ood_loss = self.beta * loss_ood - self.beta / 2 * loss_ind

        return ood_loss.float()

    def compute_ood_scores(self, logits: torch.Tensor):
        return -logits.max(dim=1)[0]

    def training_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        img, label = batch
        # labels can have 101 label, or ood label, this wrapper ignores those labels
        ood_mask = label == 101
        label[ood_mask] = 100 # change ood labels to ignore

        out: torch.Tensor = self.model(img.float()/255.0) # logits
        seg_loss = nn.functional.cross_entropy(out, label, ignore_index=100)

        ood_loss = self.ood_loss(out, ood_mask)

        out_proba = out.softmax(dim=1)
        self.jaccard_index.update(out_proba, label)
        self.log(
            "train_loss", seg_loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)
        self.log(
            "train_loss_ood", ood_loss, on_epoch=True,
            on_step=True, prog_bar=True, logger=True)

        return seg_loss.float() + ood_loss.float()

    def validation_test_step(self, batch) -> torch.Tensor:
        # pylint: disable=arguments-differ
        img, label = batch
        # labels can have 101 label, or ood label, this wrapper ignores those labels
        ood_mask = label == 101
        label[ood_mask] = 100 # change ood labels to ignore

        out = self.model(img.float()/255.0)
        loss = nn.functional.cross_entropy(out, label, ignore_index=100)

        scores = self.compute_ood_scores(out)
        self.ood_scores.append(scores.cpu().detach().numpy())
        self.ood_masks.append(ood_mask.cpu().detach().numpy())

        out_proba = out.softmax(dim=1)

        self.jaccard_index.update(out_proba, label)

        return loss.float()

    def log_metrics(self):
        # get concatenated outputs
        print(self.ood_scores[0].shape)

        all_ood_scores = np.concatenate([x.reshape(-1) for x in self.ood_scores])
        self.ood_scores = []
        all_ood_masks = np.concatenate([x.reshape(-1) for x in self.ood_masks])
        self.ood_masks = []

        auroc, fpr95, _ = calculate_auroc(all_ood_masks, all_ood_scores)
        ap = average_precision_score(all_ood_masks, all_ood_scores)

        self.log(
            "val_auroc", auroc, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "val_fpr95", fpr95, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)
        self.log(
            "val_ap", ap, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)

    def validation_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        loss = self.validation_test_step(batch)

        self.log(
            "val_miou", self.jaccard_index, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        loss = self.validation_test_step(batch)

        self.log(
            "test_miou", self.jaccard_index, on_epoch=True,
            on_step=False, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_metrics()
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.log_metrics()
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        """ Configure the optimizers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "interval": "epoch",
                "monitor": "val_miou",
                "frequency": 1
            },
        }
