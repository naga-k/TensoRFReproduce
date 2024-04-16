from omegaconf import DictConfig, OmegaConf
import os
from tqdm.auto import tqdm

import json, random
from data.blender import BlenderDataset
from model.extractor.tensorf import TensorVMSplit
from model.renderer.tensorf import TensoRFRenderer
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import imageio
import torch
from .tensorf_utils import *

# from dataLoader import BlenderDataset

class TensoRFTrainer():

    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.renderer = TensoRFRenderer()

    def train(self):
        args = self.args
        device = self.device
        renderer = self.renderer

        # init dataset
        train_dataset = BlenderDataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
        test_dataset = BlenderDataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)

        white_bg = train_dataset.white_bg
        near_far = train_dataset.near_far

        ndc_ray = args.ndc_ray

        # init resolution
        upsamp_list = args.upsamp_list
        update_AlphaMask_list = args.update_AlphaMask_list
        n_lamb_sigma = args.n_lamb_sigma
        n_lamb_sh = args.n_lamb_sh

        if args.add_timestamp:
            logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        else:
            logfolder = f'{args.basedir}/{args.expname}'

        #init log file
        os.makedirs(logfolder, exist_ok=True)
        os.makedirs(f'{logfolder}/img_vis', exist_ok = True)
        os.makedirs(f'{logfolder}/imgs_rgba', exist_ok = True)
        os.makedirs(f'{logfolder}/rgba', exist_ok = True)

        summary_writer = SummaryWriter(logfolder)

        #init parameters
        aabb = train_dataset.scene_bbox.to(device)
        reso_cur = N_to_reso(args.N_voxel_init, aabb)
        # print("runner","trainer","tensorf.py", "args.N_voxel_init", args.N_voxel_init, sep = " -> ")
        # print("runner","trainer","tensorf.py", "reso_cur", reso_cur, sep = " -> ")
        # nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
        # print("runner","trainer","tensorf.py", "nSamples", nSamples, sep = " -> ")

        if args.ckpt is not None:
            ckpt = torch.load(args.ckpt, map_location=device)
            kwargs = ckpt['kwargs']
            kwargs.update({'device': device})
            tensorf = eval(args.model_name)(**kwargs)
            tensorf.load(ckpt)
        else:
            tensorf = eval(args.model_name)(aabb, reso_cur, device,
                                            density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                                            app_dim=args.data_dim_color, near_far=near_far,
                                            shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                            density_shift=args.density_shift, distance_scale=args.distance_scale,
                                            pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                            featureC=args.featureC, step_ratio=args.step_ratio,
                                            fea2denseAct=args.fea2denseAct)

        

if __name__ == "__main__":
    print("trainer main")
