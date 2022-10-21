''' 
Load XL net from hugging face and generate embeddings for commentaries.
Visualize commentaries using t-SNE plot.
'''
import re
from send2trash import send2trash
from transformers import XLNetTokenizer, XLNetModel,AutoTokenizer
from torch.nn import functional as F
import torch
import numpy as np
from train_data import TRAIN_DATA
from visualization import obtain_visualization


class XLNetHelper():


	def __init__(self):
		self.tokenizer = None
		self.model = None
	def load_model(self):
		self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
		self.model = XLNetModel.from_pretrained('xlnet-base-cased', output_hidden_states = True)

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
			indexes = self.tokenizer._convert_token_to_id(tokenization)
			segments = [1] * len(tokenization)
			tokenization_tensor = torch.tensor([indexes])
			segments_tensor = torch.tensor([segments])
			
			self.model.eval()
			with torch.no_grad():
				outputs = self.model(tokenization_tensor, segments_tensor)
				sentence_embedding=outputs.last_hidden_state
				sentence_embedding=sentence_embedding[0]
				##Convert it into 1-dimensional data using sum
				sentence_embedding=torch.sum(sentence_embedding,dim=0)
				
			text_embeddings.append(sentence_embedding.numpy())
		return text_embeddings
			
def load_data(data):
	return [x[0] for x in data]

def visualize_commentaries(embeddings, ground_truth=np.asarray([0,1,2,3,4,0,5,6,0,3,7,6,8,9,4,0,0,8,3,0,7,8,8,10,9,5,8,9,0,11,8]),title=None):
    if title:
        obtain_visualization(embeddings, ground_truth,title)
    else:
        obtain_visualization(embeddings, ground_truth)

def main():
	#Import data
	data = load_data(TRAIN_DATA)
	#Create Xlnet Helper object
	xlnet = XLNetHelper()
	#Initialize tokenizer and model
	xlnet.load_model()
	#Obtain embeddings
	embeddings = xlnet.embed_commentaries(data)	
	embeddings = np.asarray(embeddings).squeeze()
	#Visualize t-SNE plot
	visualize_commentaries(embeddings,title="xlnet_tsne.png")


if __name__ == "__main__":
	main()