from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def obtain_visualization(embeddings, ground_truth,title=None):
	X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(embeddings)
	tsne_result_df = pd.DataFrame({'tsne_1': X_embedded[:,0], 'tsne_2': X_embedded[:,1], 'label': ground_truth})
	fig, ax = plt.subplots(1)
	sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
	lim = (X_embedded.min()-5, X_embedded.max()+5)
	ax.set_xlim(lim)
	ax.set_ylim(lim)
	ax.set_aspect('equal')
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	if title:
		plt.savefig(title)