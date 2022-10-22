''' 
Load BERT from hugging face and generate embeddings for commentaries.
Visualize commentaries using t-SNE plot.
'''
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from commentary2ratings.utils.lm_utils import load_commentaries_with_tags, tsne_visualize

class BERTHelper():

	def __init__(self):
		self.tokenizer = None
		self.model = None
		self.load_model()

	def load_model(self):
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

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
			indexes = self.tokenizer.convert_tokens_to_ids(tokenization)
			segments = [1] * len(tokenization)
			tokenization_tensor = torch.tensor([indexes])
			segments_tensor = torch.tensor([segments])
			
			self.model.eval()
			with torch.no_grad():
				outputs = self.model(tokenization_tensor, segments_tensor)
				sentence_embedding = outputs[1]

			text_embeddings.append(sentence_embedding.numpy())
		return np.asarray(text_embeddings)[:, 0, :]

def main():
	# Import data
	TRAIN_DATA, LABELS = load_commentaries_with_tags()

	# Create BERT Helper object
	bert = BERTHelper()
	
	# Obtain embeddings
	embeddings = bert.embed_commentaries(TRAIN_DATA)
	
	# Visualize t-SNE plot
	tsne_visualize(embeddings, LABELS, perplexity=5, title="bert_tsne.png")

if __name__ == "__main__":
	main()