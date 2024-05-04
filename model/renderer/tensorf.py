import torch
from torch import nn


class TensoRFRenderer(nn.Module):
    def __init__(self):
        super(TensoRFRenderer, self).__init__()
        pass

    def forward(self, rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

        rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
        N_rays_all = rays.shape[0]
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

            rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

            rgbs.append(rgb_map)
            depth_maps.append(depth_map)

        return torch.cat(rgbs), None, torch.cat(depth_maps), None, None