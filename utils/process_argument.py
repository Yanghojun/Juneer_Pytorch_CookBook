import argparse

def get_args():
    parser = argparse.ArgumentParser(description=__file__)
    parser.add_argument(
        '-c', '--config',
        help = 'The configuration file',
        default= './cfg/Denoising_cfg.yaml',
        required=False
	)
    
    args = parser.parse_args()
    return args