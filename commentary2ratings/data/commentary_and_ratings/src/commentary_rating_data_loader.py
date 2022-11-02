from datetime import datetime
import numpy as np
import json
import os
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
import torch
import torch.nn.functional as F
from commentary2ratings.lm_embeddings.bert import BERTHelper
from commentary2ratings.lm_embeddings.xlnet import XLNetHelper

class CommentaryAndRatings(Dataset):
	"""
	Returns data as such:
	[fixture_id, player, player_rating, list_of_commentaries]
	"""

	def __init__(self, fixture_csv=None, ratings_csv=None, commentary_folder=None, embed_class=BERTHelper):
		"""
		Initialization process. Reads in data and runs processing.
		If the processed data already exists, load that file directly.
		"""
		# Load BERT or XLNet
		self.embed_model = embed_class()

		# Check for the pre-processed data:
		preprocessed_path = os.path.join(os.environ['DATA_DIR'], 'player_comments_ratings.csv')
		if not os.path.exists(preprocessed_path):
			assert (fixture_csv is not None) and (ratings_csv is not None) and (commentary_folder is not None)
			fixtures_df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], fixture_csv))
			ratings_df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], ratings_csv))
			
			self.commentary_rating = self.parseCommentary(os.path.join(os.environ['DATA_DIR'], commentary_folder), ratings_df, fixtures_df)
		else:
			self.commentary_rating = pd.read_csv(preprocessed_path, converters={'comments': literal_eval})

		self.dataset = self.process_for_learning()
		# Remove the indices since we don't want them when we return items for training
		self.commentary_rating = self.commentary_rating.reset_index(drop=True).values.tolist()

	def __len__(self):
		return self.dataset['rating'].shape[0]

	def __getitem__(self, idx):
		return {k: self.dataset[k][idx] for k in self.dataset}#self.commentary_rating[idx, :3] #'fixture_id', 'player', 'rating', 'comments'

	def parseCommentary(self, commentary_folder, ratings_df, fixtures_df):
		"""
		Parses through the folder of commentary alongside the ratings and fixture
		data.
		"""

		self.createPlayerList(ratings_df)
		
		# Create a new DataFrame with consolidated player names and their associated comments
		player_comments = pd.DataFrame(columns=['fixture_id', 'player', 'comments'])

		# Run through all of the commentary files and search for players within the commentary
		for filename in os.listdir(commentary_folder):
			fixture_id = os.path.splitext(filename)[0]
			with open(os.path.join(commentary_folder, filename), 'r', encoding='utf-8') as f:
				commentary_data = json.load(f)['data']
				for data in commentary_data:

					comment = data['comment']
					for player in self.players:
						if player in comment:
							new_comment_df = pd.DataFrame([[fixture_id, player, comment]], columns=['fixture_id', 'player','comments'])
							player_comments = pd.concat([player_comments, new_comment_df])

		# Group each comment with the same player name as a list
		player_comments=player_comments.groupby(['fixture_id', 'player'], sort=False)['comments'].apply(list).reset_index()

		player_rating_comment = pd.DataFrame(columns=['fixture_id', 'player', 'rating', 'comments'])

		# Now go through each player and add a rating from the ratings dataset
		for index, player_comment_row in player_comments.iterrows():

			fixture_id = int(player_comment_row['fixture_id'])
			player = player_comment_row['player']
			comments = player_comment_row['comments']

			# Grab the row in fixtures that has this id
			fixture_row = fixtures_df.loc[fixtures_df['id'] == fixture_id]

			if fixture_row.empty:
				print("Fixture ", fixture_id, " missing from fixtures.csv")
				continue

			# Find the matching date in the fixtures data
			date = fixture_row['time_starting_at_date'].values[0]

			# Convert the date in fixtures to the one used in the player stats file
			stat_date = datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%Y')
			ratings_for_id = ratings_df.loc[ratings_df['date'] == stat_date]

			if ratings_for_id.empty:
				print("Fixture ", fixture_id, " missing from player statistics")
				continue

			# We have the stat rows, so now we look for the matching stats
			player_avg = 0
			rating_count = 0
			for rating_idx, rating_row in ratings_for_id.iterrows():
				
				if rating_row['player'] == player_comment_row['player'] and rating_row['rater'] != 'Kicker':
					player_avg += rating_row['original_rating']
					rating_count += 1

			if rating_count != 0:
				player_avg /= rating_count
				new_comment_df = pd.DataFrame([[fixture_id, player, player_avg, comments]], columns=['fixture_id', 'player', 'rating', 'comments'])
				player_rating_comment = pd.concat([player_rating_comment, new_comment_df])

		# # Write this out to file so we can skip this in the future
		player_rating_comment.to_csv(os.path.join(os.environ['DATA_DIR'], 'player_comments_ratings.csv'))
		return player_rating_comment
			
	def createPlayerList(self, ratings_df):
		"""
		Create a unique player list.
		"""
		# Hack? Deal with the player who is just named William from the ratings dataset with no last name
		# to disambiguate from other players with William in their name
		ratings_df['player'] = ratings_df['player'][ratings_df['player'] != 'William']
		ratings_df['player'] = ratings_df['player'].dropna()
		self.players = ratings_df['player'].drop_duplicates().dropna()

	def player_one_hot(self, player):
		one_hot = torch.zeros(len(self.player2idx))
		one_hot[self.player2idx[player]] = 1
		return one_hot

	def process_for_learning(self):
		n_samples = len(self.commentary_rating)
		self.player2idx = {p: i for i, p in enumerate(sorted(self.commentary_rating['player'].unique()))}
		
		dataset = {
					'player': torch.zeros((n_samples, len(self.player2idx)), dtype=torch.float32),
					'rating': torch.zeros(n_samples, dtype=torch.float32),
					'padded_commentary_embeddings': [],
					'commentary_len': torch.zeros(n_samples, dtype=torch.float32)
				}
		for idx, row in self.commentary_rating.iterrows():
			print(idx)
			dataset['player'][idx, self.player2idx[row['player']]] = 1
			dataset['rating'][idx] = row['rating']

			dataset['commentary_len'][idx] = len(row['comments'])
			dataset['padded_commentary_embeddings'].append(torch.tensor(self.embed_model.embed_commentaries(row['comments']))[:, :3])
		
		max_len = torch.max(dataset['commentary_len'])
		dataset['padded_commentary_embeddings'] = torch.stack([F.pad(comments, (0, 0, 0, int(max_len-n_comments))) 
													for comments, n_comments in zip(dataset['padded_commentary_embeddings'], dataset['commentary_len'])])

		_, indices = torch.sort(dataset['commentary_len'], descending=True)
		dataset = {k : dataset[k][indices] for k in dataset}

		return dataset
		

if __name__=='__main__':
	import random
	from commentary2ratings.lm_embeddings.bert import BERTHelper
	from commentary2ratings.lm_embeddings.xlnet import XLNetHelper
	from torch.utils.data import DataLoader
	
	data = CommentaryAndRatings('fixtures.csv', 'data_football_ratings.csv', 'commentary')
	
	# model = BERTHelper()
	# for i in range(10):
	# 	comments = data[random.randrange(0, len(data))]
	# 	if len(comments[4])>0:
	# 		embeddings = model.embed_commentaries(comments[4])
	# 		print(embeddings.shape)

	# loader = DataLoader(data, batch_size=10, shuffle=True)
	# for batch in loader:
	# 	print(batch['player'].shape, batch['rating'].shape)
	# 	print(batch)