import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics as tm

from nets.vainfDLV3p.modeling import deeplabv3plus_resnet50, deeplabv3plus_resnet101

class Wrapper(LightningModule):

    def __init__(
            self,
            backbone: str = "resnet50",
            num_classes: int = 2,
            **kwargs: torch.Any) -> None:
        super().__init__(**kwargs)

        if backbone=="resnet50":
            self.model = deeplabv3plus_resnet50(num_classes=num_classes, pretrained_backbone=True)
        elif backbone=="resnet101":
            self.model = deeplabv3plus_resnet101(num_classes=num_classes, pretrained_backbone=True)

        self.cm = tm.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, _) -> torch.Tensor:
        # pylint: disable=arguments-differ
        out = self.model(batch["img"])

        loss = nn.functional.cross_entropy(out, batch["semseg"], ignore_index=100)

        return loss
