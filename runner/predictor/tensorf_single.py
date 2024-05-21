from omegaconf import DictConfig, OmegaConf
import os
from tqdm.auto import tqdm

import json, random
from data.blender import BlenderDataset
from model.extractor.tensorf import TensorVMSplit
from model.renderer.tensorf import TensoRFRenderer
from .tensorf_utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import imageio


@torch.no_grad()
def evaluation_one_image(test_dataset,tensorf, args, renderer, savePath=None, test_img_idx=0, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    # img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    # idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    idxs = [test_img_idx]
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[test_img_idx:test_img_idx+1]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))
    return PSNRs

# @torch.no_grad()
# def get_valid_normal_idx(test_dataset, img_idx, device = 'cuda'):
#     rays = test_dataset.all_rays[img_idx].view(-1,test_dataset.all_rays.shape[-1])
#     print(rays.shape)
#     viewdirs = rays[..., 3:6]
#     print(viewdirs[10])
#     all_normals = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1], [-1, 0, 0], [0, -1, 0]], device=device)

#     return None

class TensoRFPredictorSingle:
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.renderer = TensoRFRenderer()

    @torch.no_grad()
    def test(self):
        args = self.args
        device = self.device
        renderer = self.renderer
        test_img_idx = args.test_img_idx

        # init dataset
        test_dataset = BlenderDataset(self.args.datadir, split='test', downsample=self.args.downsample_train, is_stack=True)
        white_bg = test_dataset.white_bg
        ndc_ray = args.ndc_ray

        # valid_normal_idx = get_valid_normal_idx(test_dataset, test_img_idx, device=device)

        if not os.path.exists(args.ckpt):
            print('the ckpt path does not exists!!')
            return

        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)

        logfolder = os.path.dirname(args.ckpt)

        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_single', exist_ok=True)
        evaluation_one_image(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_single/',test_img_idx = test_img_idx, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)