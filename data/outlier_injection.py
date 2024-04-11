import numpy as np
import torch
from torch import nn

def match_histogram(source, target):
    '''
    Returns the source image with the histograms matching those of target

    target and source images are of uint8, with size (w, h) and (w', h')
    '''
    _, source_counts = torch.unique(source, return_counts=True)
    target_values, target_counts = torch.unique(target, return_counts=True)

    cpf_source = torch.cumsum(source_counts, 0) / source.numel()
    cpf_target = torch.cumsum(target_counts, 0) / target.numel()

    interp_val = torch.tensor(
        np.interp(
            cpf_source.cpu().numpy(),
            cpf_target.cpu().numpy(),
            target_values.cpu().numpy()), dtype=torch.int64)

    return interp_val[source.long()]


class OutlierInjection(nn.Module):

    def __init__(
            self,
            alpha_blend: float = 1,
            histogram_matching: bool = False,
            ) -> None:
        super().__init__()

        self.alpha_blend = alpha_blend
        histogram_matching = histogram_matching

    def forward(self, image, label, outlier, mask):
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

        return image, label
