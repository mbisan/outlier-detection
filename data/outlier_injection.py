import random

import numpy as np
import torch
from torch import nn
import torchvision.transforms.v2 as T

def match_histogram(source, target):
    '''
    Returns the source image with the histograms matching those of target

    target and source images are of uint8, with size (w, h) and (w', h')
    '''
    _, source_lookup, source_counts = torch.unique(
        source, return_counts=True, return_inverse=True)
    target_values, target_counts = torch.unique(target, return_counts=True)

    cpf_source = torch.cumsum(source_counts, 0) / source.numel()
    cpf_target = torch.cumsum(target_counts, 0) / target.numel()

    interp_val = torch.tensor(
        np.interp(
            cpf_source.cpu().numpy(),
            cpf_target.cpu().numpy(),
            target_values.cpu().numpy()), dtype=torch.uint8)

    return interp_val[source_lookup]


class OutlierInjection(nn.Module):

    def __init__(
            self,
            alpha_blend: float = 1,
            histogram_matching: bool = False,
            blur: int = 0
            ) -> None:
        super().__init__()

        self.alpha_blend = alpha_blend
        self.histogram_matching = histogram_matching
        self.blur = blur

    def forward(
            self,
            image: torch.Tensor,
            label: torch.Tensor,
            outlier: torch.Tensor,
            mask: torch.Tensor,
            horizon: float):
        '''
            Image of shape (3, w, h)
            Label of shape (w, h)
            Outlier of shape (3, w', h')
            outlier_mask of shape (w', h')
            Horizon: int

            Puts outlier at a random position on the image
                pasting is done with:
                    alpha blending
                    histogram matching
                    if possible, pasting is done below the "horizon" line
            Sets the labels at the position of the mask to the ignore label

            returns the images with the synthetic outliers, and modified labels
        '''
        if self.histogram_matching:
            for c in range(image.shape[0]):
                outlier[c] = match_histogram(outlier[c], image[c])

        # paste
        x_ood, y_ood = outlier.shape[1], outlier.shape[2]
        x_shape, y_shape = image.shape[1], image.shape[2]
        x_index = random.randint(min(max(horizon, 0), x_shape - x_ood), x_shape - x_ood)
        y_index = random.randint(0, y_shape - y_ood)

        # update labels and blur the outlier mask for pasting
        label[x_index:x_index+x_ood, y_index:y_index+y_ood][mask.bool()] = 101
        if self.blur>0:
            mask = T.functional.gaussian_blur(mask.unsqueeze(0), self.blur)[0]

        # paste
        image[:, x_index:x_index+x_ood, y_index:y_index+y_ood] = \
            image[:, x_index:x_index+x_ood, y_index:y_index+y_ood] - self.alpha_blend*mask.unsqueeze(0)*image[:, x_index:x_index+x_ood, y_index:y_index+y_ood] + \
                self.alpha_blend * outlier * mask.unsqueeze(0)

        return image, label
