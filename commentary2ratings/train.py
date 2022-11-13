import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, exp_name, model_class, train_dataset, val_dataset, batch_size=64):

        '''
            exp_name: unique name of the current run(include details of the architecture. eg. SimpleC2R_64x3_relu_run1)
            model_class: class of the pytorch model being used (eg. SimpleC2R)
            train_dataset: dataset to be trained
            val_dataset: dataset to be validated
        '''
        self.exp_name = exp_name
        self.model_class = model_class
        self.model = model_class()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        self.create_log()

    def train(self, n_epoch):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        for epoch in range(n_epoch):
            total_loss = 0
            for batch in loader:
                inputs = batch
                #print(inputs['padded_commentary_embedding'].shape)
                #print(inputs['player'][0])
                #break
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                losses = self.model.loss(outputs, inputs)
                losses.backward()
                optimizer.step()

                total_loss += float(losses.detach())/len(loader)

            val_loss = None
            if epoch%3==0:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = 0
                    for batch in val_loader:
                        inputs = batch
                        outputs = self.model(inputs)
                        val_loss += float(self.model.loss(outputs, inputs))/len(val_loader)
                    
                    print("epoch{} loss:".format(epoch), total_loss)
                    print("==> epoch{} val loss:".format(epoch), val_loss)
                    if epoch%5:
                        self.save_checkpoint(epoch)
            
            self.log_outputs(epoch, total_loss, val_loss)
        self.logger.close()

    def create_log(self):
        self.log_path = os.path.join(os.environ['EXP_DIR'], 'rating_predictor', self.model_class.__name__, self.exp_name)
        self.logger = SummaryWriter(self.log_path)
        self.logger.add_text('dataset', self.train_dataset.dataset_path, 0)
    
    def log_outputs(self, epoch, train_loss, val_loss):
        self.logger.add_scalar('loss/train', train_loss, epoch)
        if val_loss is not None:
            self.logger.add_scalar('loss/val', val_loss, epoch)
        
        self.model.log_params(epoch, self.logger)

    def save_checkpoint(self, epoch):
        self.model.save_weights(epoch, self.log_path)

if __name__=='__main__':
    import sys
    from commentary2ratings.models import TestC2R, SimpleC2R, ProjC2R, SeqC2R
    from commentary2ratings.data.commentary_and_ratings.src.commentary_rating_data_loader import CommentaryAndRatings

    if len(sys.argv)<2:
        raise Exception('Enter the unique name of the current run(include details of the architecture. eg. SimpleC2R_64x3_relu_run1)')
    Trainer(
                exp_name=sys.argv[1],
                model_class = SeqC2R, 
                train_dataset = CommentaryAndRatings(processed_dataset_path='processed_data_bert.h5', mode='train'),
                val_dataset = CommentaryAndRatings(processed_dataset_path='processed_data_bert.h5', mode='val')
            ).train(n_epoch=51)
