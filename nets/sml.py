import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from kornia import morphology as km
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

FULL_3X3_KERNEL = torch.ones((3, 3), dtype=torch.float32, device=device)
CROSS_3X3_KERNEL = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=device)

def get_dilation_kernel(kernel: torch.Tensor, k):
    if kernel.sum()==0:
        kernel[0, 0, kernel.shape[-2]//2, kernel.shape[-1]//2] = 1

    if k==0:
        return kernel.squeeze(0).squeeze(0)

    return get_dilation_kernel(km.dilation(kernel, CROSS_3X3_KERNEL), k-1)

DILATION_KERNEL = [
    get_dilation_kernel(torch.zeros((1, 1, 2 * i + 1, 2 * i + 1), device=device), i) for i in range(10)
]

def get_2d_gaussian_kernel(n, std, normalised=False) -> np.ndarray:
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian_1D = cv2.getGaussianKernel(n, std)
    gaussian_2D = np.outer(gaussian_1D, gaussian_1D)
    if normalised:
        gaussian_2D /= (2*np.pi*(std**2))
    return gaussian_2D

def find_boundaries(label):
    """
    Calculate boundary mask by getting diff of dilated and eroded prediction maps
    """
    assert len(label.shape) == 4
    boundaries = (km.dilation(label.float(), CROSS_3X3_KERNEL) != km.erosion(label.float(), FULL_3X3_KERNEL)).float()

    return boundaries

def expand_boundaries(boundaries, r=0):
    """
    Expand boundary maps with the rate of r
    """
    if r == 0:
        return boundaries
    expanded_boundaries = km.dilation(boundaries, DILATION_KERNEL[r])

    return expanded_boundaries

def max_logits(logits: torch.Tensor) -> torch.Tensor:
    return -logits.max(dim=1)[0]

def unnormalized_likelihood(logits: torch.Tensor) -> torch.Tensor:
    return -logits.logsumexp(dim=1)

class SMLWithPostProcessing(nn.Module):

    def __init__(
            self,
            means: torch.Tensor,
            std: torch.Tensor,
            base: str = "max_logits",
            boundary_suppression=True,
            boundary_width=4,
            boundary_iteration=4,
            dilated_smoothing=True,
            kernel_size=7,
            dilation=6
            ):
        '''
        Arguments:
            means: tensor of shape (c,) with per-class means
            sts: tensor of shape (c,) with per-class variances
        '''

        super().__init__()

        assert means.shape[0] == std.shape[0]
        self.num_classes = means.shape[0]

        self.register_buffer("means", means.type(torch.float32))
        self.register_buffer("std", std.type(torch.float32))

        self.boundary_suppression = boundary_suppression
        self.boundary_width = boundary_width
        self.boundary_iteration = boundary_iteration
        self.dilated_smoothing = dilated_smoothing
        self.kernel_size = kernel_size
        self.dilation = dilation

        diff = self.boundary_width // self.boundary_iteration
        self.boundary_per_iteration = [self.boundary_width - diff * i - 1 for i in range(self.boundary_iteration-1)] + [0]

        self.register_buffer("gaussian_kernel",
            torch.from_numpy(get_2d_gaussian_kernel(self.kernel_size, 1.0)).type(torch.float32).unsqueeze(0).unsqueeze(0))
        
        self.replication_pad_1 = nn.ReplicationPad2d(1)
        self.replication_pad_dilation = nn.ReplicationPad2d(3*self.dilation)

        if base == "max_logits":
            self.base_ood_scores = max_logits
        elif base == "unnormalized_likelihood":
            self.base_ood_scores = unnormalized_likelihood

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: logits of shape (n, c, w, h)
        '''

        ood_scores = -self.base_ood_scores(x).unsqueeze(1)
        _, pred = x.max(1, keepdims=True) # both of shape (n, 1, w, d)

        sml = (ood_scores - self.means[pred]) / self.std[pred] # shape (n, 1, w, d)

        if self.boundary_suppression:
            boundaries = find_boundaries(pred)

            for boundary_width in self.boundary_per_iteration:
                # expand boundaries, find what sml values are not inside boundaries
                # update the sml values by dividing the number of values inside boundaries
                expanded_boundaries = expand_boundaries(boundaries, r=boundary_width)
                non_boundary_mask = 1. * (expanded_boundaries == 0)
                sml_masked = sml * non_boundary_mask

                sml_non_boundary = F.conv2d(
                    self.replication_pad_1(sml_masked), FULL_3X3_KERNEL.unsqueeze(0).unsqueeze(0), padding="valid")
                non_boundary_count = F.conv2d(
                    self.replication_pad_1(non_boundary_mask), FULL_3X3_KERNEL.unsqueeze(0).unsqueeze(0), padding="valid").long()

                sml_avg = torch.where(non_boundary_count == 0, sml, sml_non_boundary/non_boundary_count)

                sml = torch.where(non_boundary_mask == 0, sml_avg, sml) # inside boundaries the new value is put

        if self.dilated_smoothing:
            sml = F.conv2d(self.replication_pad_dilation(sml_masked), self.gaussian_kernel, padding="valid", dilation=self.dilation)

        return -sml
