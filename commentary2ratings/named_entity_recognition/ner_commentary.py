# Using NER on commentary data to tag the associated players and clubs
import re
import spacy
from spacy.tokens import Span
import os
import pandas as pd
import json
from train_data import TRAIN_DATA
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from pathlib import Path



def train_ner():
    output_dir = Path('./content/')
    nlp = spacy.load("en_core_web_sm")
    ner=nlp.get_pipe("ner")

    
    #print(doc.text)
    #print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]



    with nlp.disable_pipes(*unaffected_pipes):

    # Training for 30 iterations
        for iteration in range(30):

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 65.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)

                example = []
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                nlp.update(
                            example,
                            drop=0.5,  # dropout - make it harder to memorise data
                            losses=losses,
                        )
                print("Losses", losses)

    #doc = nlp("Federico Fernández (Swansea City) wins a free kick in the defensive half.")
    #print(doc)
    #print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

    
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

def get_sample(path, n):
    df = pd.read_csv(path, usecols= ['commentary'])

    new_df = df.sample(n)
    print(len(new_df))

    #with open('training_vals.txt', 'a') as f:
    #    dfAsString = df.to_string(header=False, index=False)
    #    f.write(dfAsString)
    new_df.to_csv(r'training_vals.txt', header=None, index=None, sep=' ', mode='a')
    #print(new_df.head(n))



def json_to_csv():
    dfs = []
    for filename in os.listdir("commentary/commentary"):
        with open(os.path.join("./commentary/commentary/", filename), 'r', encoding="utf-8") as f:
        
            file = json.load(f)
            
            dfs.append(pd.DataFrame([row["comment"] for row in file["data"]], columns=['commentary']))
        #dfs.append(pd.read_json(os.path.join("commentary/commentary", filename)))
    appended_df = pd.concat(dfs)
    appended_df.to_csv("commentary.csv",index=False)
        #print(text)


def NER(ip_file_path, op_file_path, store):
    output_dir = Path('./content/')
    #os.environ['CLASSPATH'] = "E:/USC/CSCI 544 - NLP/Project/stanford-ner-2020-11-17/stanford-ner.jar"
    #os.environ['STANFORD_MODELS'] = 'E:/USC/CSCI 544 - NLP/Project/stanford-corenlp-4.4.0-models-english/edu/stanford/nlp/models/ner'
    #os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk-12.0.2/bin/java.exe'

    
    print("Loading from", output_dir)
    nlp_updated = spacy.load(output_dir)

    df = pd.read_csv(ip_file_path, usecols= ['commentary'])

    #new_df = df.sample(100)

    players = []
    teams = []
    for comment in df['commentary'].values:
        
        doc = nlp_updated(comment)
        extracted_players = []
        extracted_teams = []

        for t in doc.ents:
            if t.label_ == "ORG":
                extracted_teams.append(t)

            elif t.label_ == "PERSON":
                extracted_players.append(t)
        players.append(extracted_players)
        teams.append(extracted_teams)

    

    if store:
        df["players"] = players
        df['teams'] = teams
        df.to_csv(op_file_path, index=False)

    else:
        print(players)
        print(teams)

def NER_from_string(comment):
    output_dir = Path('./content/')
    #os.environ['CLASSPATH'] = "E:/USC/CSCI 544 - NLP/Project/stanford-ner-2020-11-17/stanford-ner.jar"
    #os.environ['STANFORD_MODELS'] = 'E:/USC/CSCI 544 - NLP/Project/stanford-corenlp-4.4.0-models-english/edu/stanford/nlp/models/ner'
    #os.environ['JAVAHOME'] = 'C:/Program Files/Java/jdk-12.0.2/bin/java.exe'

    
    print("Loading from", output_dir)
    nlp_updated = spacy.load(output_dir)

        
    doc = nlp_updated(comment)
    extracted_players = []
    extracted_teams = []

    for t in doc.ents:
        if t.label_ == "ORG":
            extracted_teams.append(t)

        elif t.label_ == "PERSON":
            extracted_players.append(t)

    
    return extracted_players, extracted_teams


if __name__ == "__main__":
    #json_to_csv()
    #get_sample(100, path)
    #train_ner()
    #NER('commentary.csv', 'test.csv', True)
    print(NER_from_string("Federico Fernández (Swansea City) wins a free kick in the defensive half."))