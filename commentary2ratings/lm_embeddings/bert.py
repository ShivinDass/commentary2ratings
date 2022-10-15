''' 
Load BERT from hugging face and generate embeddings for commentaries.
Visualize commentaries using t-SNE plot.
'''
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F
import torch
import numpy as np
from train_data import TRAIN_DATA
from visualization import obtain_visualization

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
			text_embeddings.append(sentence_embedding.numpy())
		return text_embeddings
			
def load_data(data):
	return [x[0] for x in data]

def visualize_commentaries(embeddings, ground_truth=np.asarray([0,1,2,3,4,0,5,6,0,3,7,6,8,9,4,0,0,8,3,0,7,8,8,10,9,5,8,9,0,11,8])):
	obtain_visualization(embeddings, ground_truth)

def main():
	#Import data
	data = load_data(TRAIN_DATA)
	#Create BERT Helper object
	bert = BERTHelper()
	#Initialize tokenizer and model
	bert.load_model()
	#Obtain embeddings
	embeddings = bert.embed_commentaries(data)
	#Remove dimensions equal to 1
	embeddings = np.asarray(embeddings).squeeze()
	#Visualize t-SNE plot
	visualize_commentaries(embeddings)

if __name__ == "__main__":
	main()