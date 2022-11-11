import numpy as np
import os
from torch.utils.data import Dataset
import h5py

class CommentaryAndRatings(Dataset):
	SPLIT = {'train': 0.8, 'val': 0.1, 'test': 0.1}
	
	"""
	Returns data as such:
	[player, rating, commentary_len, padded_commentary_embedding]
	"""

	def __init__(self, processed_dataset_path, mode='train'):
		"""
		Initialization process. Reads in data and runs processing.
		If the processed data already exists, load that file directly.
		"""
		self.mode = mode
		self.dataset_path = os.path.join(os.environ['DATA_DIR'], processed_dataset_path)
		self.dataset = {}
		with h5py.File(self.dataset_path, 'r') as f:
			for k in f:
				self.dataset[k] = np.array(f[k])
	
		#shuffle the dataset and split in train/val
		np.random.seed(0)
		indices = np.random.permutation(self.__len__())
		for k in self.dataset:
			self.dataset[k] = self.dataset[k][indices]

		if self.mode == 'train':
			for k in self.dataset:
				self.dataset[k] = self.dataset[k][:int(self.dataset[k].shape[0]*self.SPLIT['train'])]
		elif self.mode == 'val':
			for k in self.dataset:
				self.dataset[k] = self.dataset[k][int(self.dataset[k].shape[0]*self.SPLIT['train']): int(self.dataset[k].shape[0]*(self.SPLIT['train']+self.SPLIT['val']))]
		elif self.mode == 'test':
			for k in self.dataset:
				self.dataset[k] = self.dataset[k][int(self.dataset[k].shape[0]*(self.SPLIT['train'] + self.SPLIT['val'])): ]

	def __len__(self):
		return self.dataset['rating'].shape[0]

	def __getitem__(self, idx):
		return {k: self.dataset[k][idx] for k in self.dataset} #'player', 'rating', 'commentary_len', 'padded_commentary_embedding'
		

if __name__=='__main__':
	from torch.utils.data import DataLoader
	data = CommentaryAndRatings(processed_dataset_path='processed_data_bert.h5', mode='val')

	loader = DataLoader(data, batch_size=64, shuffle=True)
	for batch in loader:
		for k in batch:
			print(k, batch[k].shape)
