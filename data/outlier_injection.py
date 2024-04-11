import numpy as np
import torch
from torch import nn
import torchvision.transforms.v2 as T

def match_histogram(source, target, zero_count = 0):
    '''
    Returns the source image with the histograms matching those of target

    target and source images are of uint8, with size (w, h) and (w', h')
    '''
    _, source_lookup, source_counts = torch.unique(
        source, return_counts=True, return_inverse=True)
    source_counts[0] -= zero_count
    target_values, target_counts = torch.unique(target, return_counts=True)

    cpf_source = torch.cumsum(source_counts, 0) / (source.numel()-zero_count)
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
            mask: torch.Tensor):
        '''
            Image of shape (n, 3, w, h)
            Outlier of shape (n, 3, w, h)
            Mask of shape (n, w, h)
            Label of shape (n, w, h)

            Puts outlier at the position in the mask on the image
                pasting is done with:
                    alpha blending
                    histogram matching
            Sets the labels at the position of the mask to the ignore label

            returns the images with the synthetic outliers, and modified labels
            outlier and mask tensors are not used
        '''
        if self.histogram_matching:
            for n in range(image.shape[0]):
                for c in range(image.shape[1]):
                    outlier[n, c] = match_histogram(
                        outlier[n, c], image[n, c], (outlier[n, c] == 0).sum())

        label[mask.bool()] = 100
        if self.blur>0:
            mask = T.functional.gaussian_blur(mask, self.blur)

        image = image -self.alpha_blend*mask.unsqueeze(1)*image + self.alpha_blend * outlier
        image = image.type(torch.uint8)

        return image, label
