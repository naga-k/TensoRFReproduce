import torch

def positional_encoding(positions, freqs):
    #NOTE: Need revvision
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

        #NOTE: What are these?
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
        print(self.gridSize - 1)
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
    
    #TODO: Implement the forward prop for the model
    def forward():
        pass

