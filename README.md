# football_commentary2data
Using football commentaries to predict player ratings in match.

## Requirements

- python 3.8

## Installation Instructions

1. Go into project folder
2. Create and activate virtualenv with python 3.8: https://docs.python.org/3/library/venv.html
3. Clone this repository
4. Install requirements and package,
```
cd commentary2ratings
pip3 install -r requirements.txt
pip3 install -e .
```
5. Create data directory in the project repository,
```
mkdir ./data_files
export DATA_DIR=./data_files
```

## Download Data

Download data from [here](https://drive.google.com/drive/folders/1W76B70aN-adoJcYoX7mwDY1A8YQSNvHL?usp=sharing) and add it to the data_files folder created in step 5 above

data description
- fixtures.csv : contains information about the fixtures. Important fields are date and team ids to uniquely identify a match
- data_football_ratings.csv : contains player ratings. Using data and team names, identify the corresponding players from the game and fetch their ratings
- commentary : folder containing 380 commentary files named by their unique fixture_id
