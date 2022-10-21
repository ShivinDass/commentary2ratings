''' 
Load BERT from hugging face and generate embeddings for commentaries.
Visualize commentaries using t-SNE plot.
'''
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F
import torch
import numpy as np
from visualization import obtain_visualization
import pandas as pd
from ast import literal_eval

class BERTHelper():

	def __init__(self):
		self.tokenizer = None
		self.model = None
	def load_model(self):
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

	'''
	inputs:
		data. List of strings. Each string is a different commentary.
	outputs:
		text_embeddings. It is a numpy array with the sentence embeddings. One
						 for each string in data.
	'''
	def embed_commentaries(self, data):
		text_embeddings = []
		for text in data:
			tokenization = self.tokenizer.tokenize(text)
			indexes = self.tokenizer.convert_tokens_to_ids(tokenization)
			segments = [1] * len(tokenization)
			tokenization_tensor = torch.tensor([indexes])
			segments_tensor = torch.tensor([segments])
			
			self.model.eval()
			with torch.no_grad():
				outputs = self.model(tokenization_tensor, segments_tensor)
				hidden = outputs[2]
				sentence_embedding = outputs[1]
				print(sentence_embedding.shape)
			text_embeddings.append(sentence_embedding.numpy())
		return text_embeddings

def load_data():
	df = pd.read_csv("../../data_files/player_comments_ratings.csv")
	comments = df["comments"].values
	TRAIN_DATA = [x for l in comments for x in literal_eval(l)]
	LABELS = obtain_labels(TRAIN_DATA)
	ind = np.where(LABELS == "None")
	LABELS = np.delete(LABELS, ind)
	TRAIN_DATA = np.delete(TRAIN_DATA, ind)
	return TRAIN_DATA, LABELS

def obtain_labels(data):
  labels = []
  for item in data:
    low = item.lower()
    if "substitu" in low:
      labels.append("Substitution")
    elif "yellow card" in low:
      labels.append("Yellow card")
    elif "second half" in low:
      labels.append("Second half commentary")
    elif "blocked attack" in low:
      labels.append("Blocked attack")
    elif "attack" in low:
      labels.append("Attack")
    elif "offside" in low:
      labels.append("Offside")
    elif "corner" in low:
      labels.append("Corner")
    elif "goal" in low:
      labels.append("Goal")
    elif "free kick" in low:
      labels.append("Free kick")
    elif "foul" in low:
      labels.append("Foul")
    elif "hand" in low:
      labels.append("Hand ball")
    elif "delay" in low:
      labels.append("Game delay")
    elif "red card" in low:
      labels.append("Red card")
    else:
      labels.append("None")
  return np.asarray(labels)

def visualize_commentaries(embeddings, ground_truth=np.asarray([0,1,2,3,4,0,5,6,0,3,7,6,8,9,4,0,0,8,3,0,7,8,8,10,9,5,8,9,0,11,8])):
	obtain_visualization(embeddings, ground_truth)

def main():
	#Import data
	data, labels = load_data()

	#Create BERT Helper object
	bert = BERTHelper()
	#Initialize tokenizer and model
	bert.load_model()
	#Obtain embeddings
	embeddings = bert.embed_commentaries(data)
	
	#Remove dimensions equal to 1
	embeddings = np.asarray(embeddings).squeeze()
	#Visualize t-SNE plot
	visualize_commentaries(embeddings, labels, title="BERT_embedding_perplexity_100.png")

if __name__ == "__main__":
	main()