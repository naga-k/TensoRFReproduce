import configargparse

def config_parser():


    parser = configargparse.ArgumentParser()
    
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    
    return parser.parse_args()