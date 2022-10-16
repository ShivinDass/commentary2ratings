from datetime import datetime
import json
import os
import pandas as pd
from torch.utils.data import Dataset

class CommentaryAndRatings(Dataset):
	"""
	Returns data as such:
	[fixture_id, player, player_rating, list_of_commentaries]
	"""

	def __init__(self, fixture_csv, ratings_csv, commentary_folder):
		"""
		Initialization process. Reads in data and runs processing.
		If the processed data already exists, load that file directly.
		"""

		# Check for the pre-processed data:
		preprocessed_path = 'data_files/player_comments_ratings.csv'
		if not os.path.exists(preprocessed_path):
			fixtures_df = pd.read_csv(fixture_csv)
			ratings_df = pd.read_csv(ratings_csv)
			
			self.commentary_rating = self.parseCommentary(commentary_folder, ratings_df, fixtures_df)
		else:
			self.commentary_rating = pd.read_csv(preprocessed_path)
		
		# Remove the indices since we don't want them when we return items for training
		self.commentary_rating = self.commentary_rating.reset_index(drop=True).values.tolist()

	def __len__(self):
		return len(self.commentary_rating)

	def __getitem__(self, idx):
		return self.commentary_rating[idx]

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
			with open(os.path.join(commentary_folder, filename), 'r') as f:
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

		# Write this out to file so we can skip this in the future
		player_rating_comment.to_csv('data_files/player_comments_ratings.csv')
		return player_rating_comment
			
	def createPlayerList(self, ratings_df):
		"""
		Create a unique player list.
		"""
		# Hack? Deal with the player who is just named William from the ratings dataset with no last name
		# to disambiguate from other players with William in their name
		ratings_df['player'] = ratings_df['player'][ratings_df['player'] != 'William']
		ratings_df['player'] = ratings_df['player'].dropna()
		self.players = ratings_df['player'].drop_duplicates()