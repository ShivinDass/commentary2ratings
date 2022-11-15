import os
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
        loader = DataLoader(self.data, batch_size=64)
        for batch in loader:
            with torch.no_grad():
                self.model.eval()
                total_loss += model.loss(model(batch), batch).detach()/len(loader)

        return total_loss

    def test_loss_over_models(self):
        min_loss_epoch = 0
        min_loss = np.inf
        for epoch in range(0, 66):
            if not self.model.load_weights(epoch, self.model_weights_path):
                continue
            loss = self.test_loss(self.model)
            print("==> Test error for epoch{}: {}".format(epoch, loss))

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
                pred_ratings.append(output[:])
                true_ratings.append(batch['rating'].detach().cpu().numpy())
        
        pred_ratings = np.concatenate(pred_ratings)
        true_ratings = np.concatenate(true_ratings)

        if self.data.normalize:
            pred_ratings = pred_ratings*self.data.rating_mean + self.data.rating_stddev
            true_ratings = true_ratings*self.data.rating_stddev + self.data.rating_stddev

        plt.figure()
        plt.scatter(true_ratings, pred_ratings)
        plt.xlabel('true ratings')
        plt.ylabel('predicted ratings')
        plt.plot((4,9), (4,9), 'r--')
        plt.show()

if __name__=='__main__':
    eval = PlotCorrelation(
                    data=CommentaryAndRatings('processed_data_xlnet.h5', mode='test', normalize=False, min_comments=1),
                    model_class=SeqC2R,
                    model_weights_path=os.path.join(os.environ['EXP_DIR'], 'rating_predictor/SeqC2R/SeqC2R_l2/weights')
                )