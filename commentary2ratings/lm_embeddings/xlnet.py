''' 
Load XL net from hugging face and generate embeddings for commentaries.
Visualize commentaries using t-SNE plot.
'''
from transformers import XLNetTokenizer, XLNetModel
import torch
import numpy as np
from commentary2ratings.utils.lm_utils import load_commentaries_with_tags, tsne_visualize

class XLNetHelper():

	def __init__(self):
		self.tokenizer = None
		self.model = None
		self.load_model()

	def load_model(self):
		self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
		self.model = XLNetModel.from_pretrained('xlnet-base-cased', output_hidden_states = True)

	def embed_commentaries(self, data):
		'''
		inputs:
			data. List of strings. Each string is a different commentary.
		outputs:
			text_embeddings. It is a numpy array with the sentence embeddings. One
							for each string in data.
		'''
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
		return np.asarray(text_embeddings)

def main():
	# Import data
	TRAIN_DATA, LABELS = load_commentaries_with_tags()
	
	# Create Xlnet Helper object
	xlnet = XLNetHelper()
	
	# Obtain embeddings
	embeddings = xlnet.embed_commentaries(TRAIN_DATA)
	
	# Visualize t-SNE plot
	tsne_visualize(embeddings, LABELS, perplexity=5, title="xlnet_tsne.png")


if __name__ == "__main__":
	main()