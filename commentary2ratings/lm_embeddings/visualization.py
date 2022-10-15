from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def obtain_visualization(embeddings, ground_truth):
	tsne_result_df = pd.DataFrame({'tsne_1': embeddings[:,0], 'tsne_2': embeddings[:,1], 'label': ground_truth})
	fig, ax = plt.subplots(1)
	sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
	lim = (embeddings.min()-5, embeddings.max()+5)
	ax.set_xlim(lim)
	ax.set_ylim(lim)
	ax.set_aspect('equal')
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)