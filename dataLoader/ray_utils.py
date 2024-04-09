import torch
from kornia import create_meshgrid


'''
https://github.com/apchenstu/TensoRF/blob/main/dataLoader/ray_utils.py
'''

def get_ray_directions(H, W, focal, center = None):
    '''
    Get the ray directions for all pixels in the image plane.
    '''

    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i,j = grid.unbind(-1)
    
    cent = center if center is not None else (W/2, H/2)
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)
    return directions

def get_rays(directions, c2w):
    '''
    Get the rays for all pixels in the image plane.
    '''
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d

if __name__ == "__main__":
    t1 = get_ray_directions(10, 10, (1, 1))