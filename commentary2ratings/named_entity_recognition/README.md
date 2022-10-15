##NER

Requires Spacy.
I have also uploaded the "./content" folder as well as the commentary.csv folder so if you just want to check the file then you can directly comment out everything else in the main except NER()

Otheriwse you can ideally call the functions from any other file if needed (although I haven't tried)

```python

NER(path, store)

# This is the main function for the NER. Currently it accepts a path to csv and whether or not the output should be stored in a csv.
# It reads a csv file(it uses column named "commentary") and runs the ner on each comment in the file. if store is True, then the extracted players and teams are stored in a csv along with the original comment else it is printed to terminal

#It reads the NER model from "./content". If content directory does not exist then run train_ner() first.

#---------------------#

train_ner()

#It uses the training data in train_data.py for training. If additional training data is to be added then it should be added in train_data.py in the format

#("Comment", {"entities": [(start_char, end_char, tag("ORG"|"PERSON")), ...]})

#It trains the spacy ner model on a few rows from our dataset. It has to be done to teach it patterns of our dataset. Stores the model in "./content"

#---------------------#

get_sample(path, n)

#path is the path to the csv of all the commentaries. If doesnt exist then run json_to_csv() first for the commentary folder to get the csv.

#It just stores 'n' random samples from our dataset and stores it in a txt file. Needed this to generate the training data but it's not necessary otherwise.

#---------------------#

json_to_csv()

#takes input from the "./commentary/commentary/" folder which contains all the commentary jsons from commentary.zip and stores them in a csv called commentary.csv.

#commentary.csv has one column "commentary".
```