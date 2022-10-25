import os
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

EVENT_KEYS = {
    'substitu' : 'Substitution',
    'yellow card': 'Yellow Card',
    'second half': 'Second Half Commentary',
    'blocked attack': 'Blocked Attack',
    'attack': 'Attack',
    'offside': 'Offside',
    'corner': 'Corner',
    'goal': 'Goal',
    'free kick': 'Free Kick',
    'foul': 'Foul',
    'hand': 'Hand Ball',
    'delay': 'Game delay',
    'red card': 'Red Card'
}

def load_commentaries_with_tags():
	def obtain_labels(data):
		labels = []
		for item in data:
			labels.append('None')
			low = item.lower()

			for k in EVENT_KEYS:
				if k in low:
					labels[-1] = EVENT_KEYS[k]
					break
		return np.asarray(labels)

	df = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'player_comments_ratings.csv'), converters={'comments': literal_eval})
	TRAIN_DATA = [x for l in df["comments"].values for x in l]#[:10]
	LABELS = obtain_labels(TRAIN_DATA)
	
	# Delete labels with value None
	ind = np.where(LABELS == 'None')
	LABELS = np.delete(LABELS, ind)
	TRAIN_DATA = np.delete(TRAIN_DATA, ind)
	return TRAIN_DATA, LABELS

def tsne_visualize(embeddings, ground_truth, perplexity=100, title=None):
	X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity).fit_transform(embeddings)
	tsne_result_df = pd.DataFrame({'tsne_1': X_embedded[:,0], 'tsne_2': X_embedded[:,1], 'label': ground_truth})
	fig, ax = plt.subplots(1)
	sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120, palette="Set2")
	lim = (X_embedded.min()-5, X_embedded.max()+5)
	ax.set_xlim(lim)
	ax.set_ylim(lim)
	ax.set_aspect('equal')
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	if title:
		exp_dir = os.path.join(os.environ['EXP_DIR'], 'tsne_plots')
		if not os.path.exists(exp_dir):
			os.makedirs(exp_dir)
		plt.savefig(os.path.join(exp_dir, title))