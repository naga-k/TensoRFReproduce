import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as T
from PIL import Image

from .ray_utils import get_ray_directions, get_rays

class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):
        '''
        Initialize the BlenderDataset object.

        Args:
            datadir (str): Path to the dataset directory.
            split (str): Split of the dataset to use. Default is 'train'.
            downsample (float): Fraction of the dataset to use. Default is 1.0.
            is_stack (bool): Whether to stack the images along the channel dimension. Default is False.
            N_vis (int): Number of visualizations to use. Default is -1.
        '''

        self.datadir = datadir
        self.split = split
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.downsample = downsample

        #TODO: 800 is hardcoded and we can rather load it in the config file
        self.img_wh = (int(800/downsample), int(800/downsample))

        self.define_transforms()

        # Define the bounding box of the scene
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        
        # Define the transformation matrix to convert from Blender coordinate system to OpenCV coordinate system
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.load_and_process_data()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0, 6.0]

        self.center = torch.mean(self.scene_bbox, axis = 0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.scene_bbox[0]).float().view(1, 1, 3)

    def load_and_process_data(self):
        '''
        Read the metadata file.
        '''
        metafile = os.path.join(self.datadir, f"transforms_{self.split}.json")
        with open(metafile, 'r') as f:
            self.meta = json.load(f)

        #logic to load data from self.meta
        
        w,h = self.img_wh
        camera_angle_x = float(self.meta['camera_angle_x'])
        self.focal = .5 * w / np.tan(.5 * camera_angle_x)

        #NOTE: Not sure exactly why we are doing this
        self.directions = get_ray_directions(h,w,(self.focal,self.focal))
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()
        
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []

        img_eval_interval = 1 if self.N_vis == -1 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for idx in idxs:
            frame = self.meta['frames'][idx]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            img_path = os.path.join(self.datadir, f"{frame['file_path']}.png")
            self.image_paths += [img_path]
            img = Image.open(img_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)

            img = self.transform(img)
            img = img.view(4, -1).permute(1, 0)
            #NOTE: How are we doing the alpha blending here? as in why does this work?
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]

        self.poses = torch.stack(self.poses)

        if not self.is_stack:
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_rays = torch.cat(self.all_rays, 0)

        else:
            self.all_rgbs = torch.stack(self.all_rgbs, 0)
            self.all_rays = torch.stack(self.all_rays, 0).reshape(-1, *self.img_wh[::-1], 3)
    
    def define_transforms(self):
        '''
        Define the transformations to be applied to the images.
        '''
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        '''
        Define the projection matrix.
        '''
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def __len__(self):
        return len(self.all_rgbs)
    
    def __getitem__(self, idx):
        if self.split == 'train':
            return {'rays': self.all_rays[idx],
                    'rgbs': self.all_rgbs[idx]}
        else:
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]

            sample = {'rays': rays,
                      'rgbs': img,
                      'masks': mask}
            return sample   
            

if __name__ == "__main__":
    dataset = BlenderDataset(datadir='/project/jacobcha/nk643/data_src/nerf_synthetic/lego', split='train', downsample=1.0, is_stack=False, N_vis=1)
    n = len(dataset)
    print(n)
    start = (390 * 800) + 297
    for i in range(start,start+100):
        sample = dataset[i]
        print("index: ", i)
        print(f"sample rays: {sample['rays'].shape} ")
        print(f"sample rays data {sample['rays']}")
        print(f"sample rgbs: {sample['rgbs'].shape} ")
        print(f"sample rgbs data {sample['rgbs']}")
        if 'masks' in sample:
            print(f"sample masks: {sample['masks'].shape} ")
            print(f"sample masks data {sample['masks']}")
        print()

if __name__ == "dataLoader.blender":
    print("__name__ : ", __name__)