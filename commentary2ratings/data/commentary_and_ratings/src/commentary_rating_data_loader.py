import numpy as np
import os
from torch.utils.data import Dataset
import h5py

class CommentaryAndRatings(Dataset):
	"""
	Returns data as such:
	[fixture_id, player, player_rating, list_of_commentaries]
	"""

	def __init__(self, processed_dataset_path):
		"""
		Initialization process. Reads in data and runs processing.
		If the processed data already exists, load that file directly.
		"""

		dataset_path = os.path.join(os.environ['DATA_DIR'], processed_dataset_path)
		self.dataset = {}
		with h5py.File(dataset_path, 'r') as f:
			for k in f:
				self.dataset[k] = np.array(f[k])

	def __len__(self):
		return self.dataset['rating'].shape[0]

	def __getitem__(self, idx):
		return {k: self.dataset[k][idx] for k in self.dataset} #'fixture_id', 'player', 'rating', 'comments'
		

if __name__=='__main__':
	from torch.utils.data import DataLoader
	data = CommentaryAndRatings(processed_dataset_path='processed_data.h5')

	loader = DataLoader(data, batch_size=10, shuffle=True)
	for batch in loader:
		for k in batch:
			print(k, batch[k].shape)