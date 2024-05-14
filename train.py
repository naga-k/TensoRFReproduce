from omegaconf import DictConfig, OmegaConf
import hydra
import runner.trainer as trainer
import numpy as np
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path='config', config_name='tensorf_lego_train')
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    trainer_obj = getattr(trainer, args.trainer.class_name)(args)
    trainer_obj.train()
    return


if __name__ == '__main__':
    main()