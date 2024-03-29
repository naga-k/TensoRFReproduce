from tqdm.auto import tqdm
import torch
import numpy as np
import sys
from opt import config_parser


def reconstruction(args):


    pbar = tqdm(range(args.n_iters), minters = args.progress_refresh_rate, file = sys.stdout)

    for iteration in pbar:


        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f"Reconstruction Iteration {iteration}"
                )
            
        # Print data dimentions
        

if __name__ == "__main__":

    # torch.set_default_dtype(torch.float32)
    # torch.manual_seed(20211202)
    # np.random.seed(20211202)

    args = config_parser()
    print(args)

    reconstruction(args=args)