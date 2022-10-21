## How to use the commentary_rating_data_loader

Here is an example of usage, assuming that you are running the python file from the repo folder:

```
from commentary2ratings.data.commentary_and_ratings.src.commentary_rating_data_loader import CommentaryAndRatings
from torch.utils.data import DataLoader

if __name__ == "__main__":

	fixture_path = "data_files/fixtures.csv"
	commentary_path = "data_files/commentary"
	ratings_path = "data_files/data_football_ratings.csv"

	dataset = CommentaryAndRatings(fixture_path, ratings_path, commentary_path)

	dataloader = DataLoader(dataset)
```

The class processes the raw data and generates a player_comments_ratings.csv file that it saves to disk. If that file is available, it saves time by skipping the processing stage.