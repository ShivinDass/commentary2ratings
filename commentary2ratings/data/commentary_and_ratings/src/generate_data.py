import os
import pandas as pd
import json
import torch
import torch.nn.functional as F
from datetime import datetime
from ast import literal_eval
import h5py
from tqdm import tqdm

from commentary2ratings.lm_embeddings.bert import BERTHelper
from commentary2ratings.lm_embeddings.xlnet import XLNetHelper

class GenerateData:

    def __init__(self, fixture_csv=None, ratings_csv=None, commentary_folder=None, processed_dataset_path=None, embed_class=BERTHelper):
        # Load BERT or XLNet
        self.embed_model = embed_class()

        preprocessed_path = os.path.join(os.environ['DATA_DIR'], 'player_comments_ratings.csv')
        if not os.path.exists(preprocessed_path):
            assert (fixture_csv is not None) and (ratings_csv is not None) and (commentary_folder is not None)
            fixtures_df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], fixture_csv))
            ratings_df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], ratings_csv))
            
            self.commentary_rating = self.parseCommentary(os.path.join(os.environ['DATA_DIR'], commentary_folder), ratings_df, fixtures_df)
        else:
            self.commentary_rating = pd.read_csv(preprocessed_path, converters={'comments': literal_eval})

        dataset_path = os.path.join(os.environ['DATA_DIR'], processed_dataset_path)
        #if not os.path.exists(dataset_path):
            #self.process_for_learning(self.commentary_rating, dataset_path)

    def parseCommentary(self, commentary_folder, ratings_df, fixtures_df):
        """
        Parses through the folder of commentary alongside the ratings and fixture
        data.
        """

        # Hack? Deal with the player who is just named William from the ratings dataset with no last name
        # to disambiguate from other players with William in their name
        ratings_df['player'] = ratings_df['player'][ratings_df['player'] != 'William']
        ratings_df['player'] = ratings_df['player'].dropna()
        players = ratings_df['player'].drop_duplicates().dropna()

        # We don't want to use the Kicker ratings, so remove them:
        ratings_df = ratings_df[ratings_df['rater'] != 'Kicker']

        # Average the ratings
        ratings_df['rating'] = ratings_df.groupby(['date', 'player'])['original_rating'].transform('mean')

        # Remove the original rating since we don't need it now and de-duplicate
        ratings_df = ratings_df.drop(['original_rating', 'rater', 'is_human'], axis=1)
        ratings_df = ratings_df.drop_duplicates()
        
        # Create a new DataFrame with consolidated player names and their associated comments
        player_comments = pd.DataFrame(columns=['fixture_id', 'player', 'comments'])

        # Run through all of the commentary files and search for players within the commentary
        for filename in os.listdir(commentary_folder):
            fixture_id = os.path.splitext(filename)[0]
            with open(os.path.join(commentary_folder, filename), 'r', encoding='utf-8') as f:
                commentary_data = json.load(f)['data']
                for data in reversed(commentary_data):

                    comment = data['comment']
                    for player in players:
                        if player in comment:
                            new_comment_df = pd.DataFrame([[fixture_id, player, comment]], columns=['fixture_id', 'player','comments'])
                            player_comments = pd.concat([player_comments, new_comment_df])

        # Group each comment with the same player name as a list
        player_comments=player_comments.groupby(['fixture_id', 'player'], sort=False)['comments'].apply(list).reset_index()

        # Define which columns to grab from rankings
        rating_data_column_names = list(ratings_df.columns.values)
        column_names_to_copy = rating_data_column_names[7:49] + rating_data_column_names[55:57] + rating_data_column_names[58:59] + rating_data_column_names[60:61]

        player_rating_comment_columns = ['fixture_id', 'player', 'comments'] + column_names_to_copy
        player_rating_comment = pd.DataFrame(columns=player_rating_comment_columns)

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
            for rating_idx, rating_row in ratings_for_id.iterrows():
                
                if rating_row['player'] == player_comment_row['player']:
                    to_copy = rating_row[column_names_to_copy].tolist()
                    new_df_list = [fixture_id, player, comments] + to_copy
                    new_comment_df = pd.DataFrame([new_df_list], columns=player_rating_comment_columns)
                    player_rating_comment = pd.concat([player_rating_comment, new_comment_df])
            
        # # Write this out to file so we can skip this in the future
        player_rating_comment.to_csv(os.path.join(os.environ['DATA_DIR'], 'player_comments_ratings.csv'))
        return player_rating_comment
        
    def process_for_learning(self, commentary_rating, dataset_path):
        n_samples = len(commentary_rating)
        self.player2idx = {p: i for i, p in enumerate(sorted(commentary_rating['player'].unique()))}
        
        dataset = {
                    'player': torch.zeros((n_samples, len(self.player2idx)), dtype=torch.float32, requires_grad=False),
                    'rating': torch.zeros(n_samples, dtype=torch.float32, requires_grad=False),
                    'padded_commentary_embedding': [],
                    'commentary_len': torch.zeros(n_samples, dtype=torch.float32, requires_grad=False)
                }
        for idx, row in tqdm(self.commentary_rating.iterrows()):
            if idx>=n_samples:
                break
            dataset['player'][idx, self.player2idx[row['player']]] = 1
            dataset['rating'][idx] = row['rating']

            dataset['commentary_len'][idx] = len(row['comments'])
            dataset['padded_commentary_embedding'].append(torch.tensor(self.embed_model.embed_commentaries(row['comments']), requires_grad=False))

        max_len = torch.max(dataset['commentary_len'])
        dataset['padded_commentary_embedding'] = torch.stack([F.pad(comments, (0, 0, 0, int(max_len-n_comments))) 
                                                    for comments, n_comments in zip(dataset['padded_commentary_embedding'], dataset['commentary_len'])])

        _, indices = torch.sort(dataset['commentary_len'], descending=True)
        dataset = {k : dataset[k][indices].detach().cpu().numpy() for k in dataset}

        with h5py.File(dataset_path, 'w') as f:
            for k in dataset:
                f.create_dataset(k, data=dataset[k])

        return dataset

if __name__=='__main__':
    data_gen = GenerateData(fixture_csv='fixtures.csv', ratings_csv='data_football_ratings.csv',commentary_folder='commentary', processed_dataset_path='processed_data_bert.h5', embed_class=BERTHelper)
    