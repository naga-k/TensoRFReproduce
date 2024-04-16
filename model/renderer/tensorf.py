import torch
from torch import nn


class TensoRFRenderer(nn.Module):
    def __init__(self):
        super(TensoRFRenderer, self).__init__()
        pass

    def forward(self, rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
        pass