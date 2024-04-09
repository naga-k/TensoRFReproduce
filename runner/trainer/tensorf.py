from omegaconf import DictConfig, OmegaConf
import torch
from hydra.utils import instantiate

# from dataLoader import BlenderDataset

class TensoRFTrainer():

    def __init__(self, dataset:DictConfig):
        super().__init__()
        # self.args = args
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        print(dataset)
        print(type(dataset))
        print("initatiating TensoRFTrainer")
        

if __name__ == "__main__":
    print("trainer main")
