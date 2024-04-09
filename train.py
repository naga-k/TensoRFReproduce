from tqdm.auto import tqdm
import torch
# import numpy as np
import sys
# from opt import config_parser
# from dataLoader import dataset_dict
from hydra.utils import instantiate

# from runner.trainer.tensorf import TensoRFTrainer


from omegaconf import DictConfig, OmegaConf
import hydra


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reconstruction(args):

    dataset = dataset_dict[args.dataset](args)
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack = False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_b = train_dataset.white_bg
    near_far = train_dataset.near_far

    #NOTE: ndc_ray initution https://github.com/bmild/nerf/issues/18
    ndc_ray = args.ndc_ray

    pbar = tqdm(range(args.n_iters), miniters = args.progress_refresh_rate, file = sys.stdout)

    for iteration in pbar:


        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Reconstruction Iteration {iteration}"
                )
            
        # Print data dimentions
        

# if __name__ == "__main__":

#     # torch.set_default_dtype(torch.float32)
#     # torch.manual_seed(20211202)
#     # np.random.seed(20211202)

#     args = config_parser()
#     print(args)

#     reconstruction(args=args)



@hydra.main(version_base=None, config_path='config', config_name='lego_train')
def main(args: DictConfig):
    # print(OmegaConf.to_yaml(args))
    trainer_cfg = args.trainer
    dataset_cfg = args.dataset
    trainer = instantiate(trainer_cfg)
    dataset = instantiate(dataset_cfg)

    return

if __name__ == "__main__":
    main()