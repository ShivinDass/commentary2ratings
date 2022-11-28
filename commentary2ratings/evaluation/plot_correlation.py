import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from commentary2ratings.data.commentary_and_ratings.src.commentary_rating_data_loader import CommentaryAndRatings
from commentary2ratings.models import *

class PlotCorrelation:

    def __init__(self, data, model_class, model_weights_path):
        self.data = data
        self.model = model_class()
        self.model_weights_path = model_weights_path

        min_loss_epoch = self.test_loss_over_models()
        print("Min loss epoch: {}".format(min_loss_epoch))
        self.plot_correlation(min_loss_epoch)
    
    def test_loss(self, model):
        total_loss = 0
        SSR = 0
        SST = 0
        loader = DataLoader(self.data, batch_size=64)

        all_data = []
        pred_data = []
        for batch in loader:
            with torch.no_grad():
                self.model.eval()
                output = model(batch)
                mu = output[:,0]
                total_loss += model.loss(output, batch).detach()/len(loader)

                pred_data.append(mu.detach().cpu().numpy())
                all_data.append(batch['rating'].detach().cpu().numpy())

        pred_data = np.concatenate(pred_data)
        all_data = np.concatenate(all_data)
        
        SSR = np.sum(np.square(all_data-pred_data))
        SST = np.sum(np.square(all_data-np.mean(all_data)))

        return total_loss, 1-(SSR/SST), np.mean(np.square(all_data-pred_data))

    def test_loss_over_models(self):
        min_loss_epoch = 0
        min_loss = np.inf
        for epoch in range(0, 66):
            if not self.model.load_weights(epoch, self.model_weights_path):
                continue
            loss, r_squared, mse = self.test_loss(self.model)
            print("==> epoch{}\nTest error:{}\nR-squared:{}".format(epoch, loss, r_squared))

            if min_loss > loss:
                min_loss = loss
                min_loss_epoch = epoch
        return min_loss_epoch

    def plot_correlation(self, epoch):
        self.model.load_weights(epoch, self.model_weights_path)
        pred_ratings = []
        true_ratings = []
        for batch in DataLoader(self.data, batch_size=64):
            with torch.no_grad():
                self.model.eval()
                output = self.model(batch).squeeze().detach().cpu().numpy()
                pred_ratings.append(output[:, 0])
                true_ratings.append(batch['rating'].detach().cpu().numpy())
        
        pred_ratings = np.concatenate(pred_ratings)
        true_ratings = np.concatenate(true_ratings)

        if self.data.normalize:
            pred_ratings = pred_ratings*self.data.rating_mean + self.data.rating_stddev
            true_ratings = true_ratings*self.data.rating_mean + self.data.rating_stddev

        loss, r_squared, mse = self.test_loss(self.model)
        # import pickle
        # with open('ours.pkl', 'wb') as f:
        #     pickle.dump([pred_ratings, true_ratings], f)
        plt.figure()
        plt.scatter(true_ratings, pred_ratings, c=((0.5, 0.5, 1)), alpha=0.5)
        plt.xlabel('true ratings')
        plt.ylabel('predicted ratings')
        plt.plot((4,9), (4,9), 'r--')
        plt.text(4, 9, 'MSE: {:.3f}'.format(mse), fontsize=10)
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, help="path to weights directory")
    parser.add_argument("--normalize", default=False, type=int, help="whether to normalize training data or not")
    parser.add_argument("--min_comments", default=None, type=int, help="parameter to filter data by minimum commentaries")
    args = parser.parse_args()
    # Example command: python commentary2ratings/evaluation/plot_correlation.py --weights_dir=experiments/rating_predictor/SeqC2R/SeqC2R_nll/weights --normalize=0
    
    eval = PlotCorrelation(
                    data=CommentaryAndRatings('processed_data_xlnet.h5', mode='test', normalize=args.normalize, min_comments=args.min_comments),
                    model_class=SeqC2R,
                    model_weights_path=args.weights_dir
                )