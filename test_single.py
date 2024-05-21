from omegaconf import DictConfig, OmegaConf
import hydra
import runner.predictor as predictor
import numpy as np
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path='config', config_name='tensorf_lego_test_single')
def main(args: DictConfig):
    print(OmegaConf.to_yaml(args))
    predictor_obj = getattr(predictor, args.predictor.class_name)(args)
    predictor_obj.test()
    return


if __name__ == '__main__':
    main()