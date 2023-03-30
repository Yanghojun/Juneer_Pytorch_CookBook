# pytorch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# etc..
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))        # 상위 폴더인 Juneer_deeplearning_cookbook을 append
from base.loader_base import BaseLoader
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image

class LasDataLoader(BaseLoader):
	def __init__(self, config, train=True):
		self.dataset = LasDataSet(config, train)

	def build(self):
		return DataLoader(self.dataset, batch_size=self.dataset.batch_size, shuffle=True)


# Inner class LasDataSet
class LasDataSet(Dataset):
	def __init__(self, config, train=True):
		self.config = config
		self.batch_size = self.config.config_namespace.BATCH_SIZE
		self.transform = transforms.ToTensor()

		if train:
			dir_path = self.config.config_namespace.TRAIN_DATA_PATH
			# self.img_paths = glob(os.path.join(self.config.config_namespace.TRAIN_DATA_PATH, '*.png'))
		else:
			dir_path = self.config.config_namespace.TEST_DATA_PATH
			# self.img_paths = glob(os.path.join(self.config.config_namespace.TEST_DATA_PATH, '*.png'))

		file_list = os.listdir(dir_path)
		self.x_paths = [os.path.join(dir_path, file) for file in file_list if '_denoised.png' not in file]
		self.y_paths = [os.path.join(dir_path, file) for file in file_list if '_denoised.png' in file]
		self.x_paths.sort(), self.y_paths.sort()

	def __len__(self):
		# torch는 batchsize로 따로 조절해줄 필요 없을듯. torch Loader 클래스에 batch argument 존재
		if self.config.config_namespace.PYTORCH:
			return len(self.x_paths)

	def __getitem__(self, idx):
		x_path, y_path = self.x_paths[idx], self.y_paths[idx]
		img = Image.open(x_path)
		label = Image.open(y_path)

		img, label = self.transform(img), self.transform(label)
		return img, label