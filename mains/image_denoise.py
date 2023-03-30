import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.process_argument import get_args
from utils.process_configuration import ConfigurationParameters
from data_loader.las_data_loader import LasDataLoader
from models.autoencoder import AutoEncoder
from tqdm import tqdm

def main():
    # try:
    # capture the command line arguments from the interface script
    args = get_args()

    # pares the configuration parameters for the autoencoder model
    config = ConfigurationParameters(args)

    # except:
    #     print('Missing or Invalid arguments! ')
    #     exit(0)

    train_dataloader = LasDataLoader(config, train=True).build()
    test_dataloader = LasDataLoader(config, train=False).build()
    
    model = AutoEncoder(config, train_dataloader, test_dataloader)
    model.fit_model()

    # model.predict()
    # model.generate_data()

if __name__ == '__main__':
    main()