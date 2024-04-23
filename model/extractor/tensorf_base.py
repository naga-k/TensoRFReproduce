import torch
from torch.nn import functional as F

def positional_encoding(positions, freqs):
    #NOTE: Need renvonvidivision
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe = 6, feape = 6, featureC = 128):
        super(MLPRender_Fea, self).__init__()

        #NOTE: Did not understand why exactly the following is done
        #TODO: Revise the paper when the renderer is called
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3)) # 3 for RGB
        
        torch.nn.init.constant_(self.mlp[-1].weight, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]

        if self.feape > 0:
            indata.append(positional_encoding(pts, self.feape))
        if self.viewpe > 0:
            indata.append(positional_encoding(pts, self.viewpe))

        mlp_in = torch.cat(indata, dim=-1)
        return torch.sigmoid(self.mlp(mlp_in)) #RGB
    


class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp=8, appearance_n_comp=24, app_dim=27,
                 shadingMode='MLP_PE', alphaMask=None, near_far=[2.0, 6.0],
                 density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                 pos_pe=6, view_pe=6, fea_pe=6, featureC=128, step_ratio=2.0,
                 fea2denseAct='softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        #NOTE: What are the following parameters?

        #NOTE: For more on feature2density, refer to self.feature2density
        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.updateStepSize(gridSize)

        
        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC

        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def updateStepSize(self, gridSize):
        #NOTE: I think this is for ray sampling
        self.aabbSize = self.aabb[1] - self.aabb[0]
        #NOTE: Why is the inverse 2/x? is it because ndc is -1 to 1?
        self.invaabbSize = 2.0/self.aabbSize[0]
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize/(self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        #NOTE: Presumably the length of the ray is the length of the diagnoal
        self.aabbDiag = torch.norm(self.aabbSize)
        #NOTE: I think this is the number of samples per ray
        self.nSamples = int((self.aabbDiag/ self.stepSize).max())


    def init_svd_volume(self, res, device):
        pass

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        pass
        if shadingMode == "MLP_Fea":
            self.renderingModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        #TODO: Implement other shading modes
        else:
            raise NotImplementedError("Shading mode not implemented")
    def compute_features(self, xyz_sampled):
        pass

    def compute_densityfeature(self, xyz_sampled):
        pass

    def compute_appfeature(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize': self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }
    
    def save(self, path):
        pass

    def load(self, ckpt):
        pass

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        #NOTE: NDC is pretty straight forward. It must be the straigt forward way to sample rays
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox
    
    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[0] - rays_o) / vec
        rate_b = (self.aabb[1] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples).to(rays_o)
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])

        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox
    
    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        else:
            raise NotImplementedError("Activation function not implemented")

        


    #TODO: Implement the forward prop for the model
    def forward(self, rays_chunk, white_bg = True, is_train=True, ndc_ray = False, N_samples = -1):
        
        viewdirs = rays_chunk[..., 3:6]

        if ndc_ray:
            #This is accumuated transmittance
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                 N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm

        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,
                                                                 N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)

        viewdirs = viewdirs.view(-1,1, 3).expand(xyz_sampled.shape)

        #TODO: ADD CODE FOR ALPHA MASKING SO WE DONT COMPUTE IN EMPTY SPACE
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])

        sigma = torch.zeros(xyz_sampled.shape[:-1]).to(xyz_sampled.device)
        rgb = torch.zeros(*xyz_sampled.shape[:2] , 3).to(xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled[ray_valid])
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

if __name__ == "__main__":
    import sys
    sys.path.append('/project/jacobcha/nk643/TensoRFReproduce')
    from data.blender import BlenderDataset
    device = "cuda"
    train_dataset = BlenderDataset("/project/jacobcha/nk643/data_src/nerf_synthetic/lego", split='train', downsample=1.0, is_stack=False)
    aabb = train_dataset.scene_bbox.to(device)
    from runner.trainer.tensorf_utils import N_to_reso
    reso_cur = N_to_reso(2097156, aabb)
    model = TensorBase(aabb, reso_cur, device, density_n_comp=8, appearance_n_comp=24, app_dim=27,
                    shadingMode='MLP_Fea', alphaMask=None, near_far=[2.0, 6.0],
                    density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe=6, view_pe=6, fea_pe=6, featureC=128, step_ratio=2.0,
                    fea2denseAct='softplus')
    
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    print(type(allrays))
    print(allrays[:4096].shape)
    rays = allrays[:4096]
    rays = rays.to(device)
    model(rays, white_bg = True, is_train=True, ndc_ray = False, N_samples = -1)
