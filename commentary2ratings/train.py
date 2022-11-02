import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, exp_name, model_class, train_dataset, val_dataset, batch_size=64):
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
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                losses = self.model.loss(outputs, inputs)
                losses.backward()
                optimizer.step()

                total_loss += float(losses.detach())/len(loader)

            val_loss = None
            if epoch%5==0:
                with torch.no_grad():
                    val_loss = 0
                    for batch in val_loader:
                        inputs = batch
                        outputs = self.model(inputs)
                        val_loss += float(self.model.loss(outputs, inputs))/len(val_loader)
                    
                    print("epoch{} loss:".format(epoch), total_loss)
                    print("==> epoch{} val loss:".format(epoch), val_loss)
                    self.save_checkpoint(epoch)
            
            self.log_outputs(epoch, total_loss, val_loss)
        self.logger.close()

    def create_log(self):
        self.log_path = os.path.join(os.environ['EXP_DIR'], 'rating_predictor', self.model_class.__name__, self.exp_name)
        self.logger = SummaryWriter(self.log_path)
    
    def log_outputs(self, epoch, train_loss, val_loss):
        self.logger.add_scalar('loss/train', train_loss, epoch)
        if val_loss is not None:
            self.logger.add_scalar('loss/val', val_loss, epoch)
        
        self.model.log_params(epoch, self.logger)

    def save_checkpoint(self, epoch):
        self.model.save_weights(epoch, self.log_path)

if __name__=='__main__':
    # import numpy as np
    # from commentary2ratings.models.test_c2r import TestC2R
    # class Hello(Dataset):
    #     def __init__(self):
    #         self.W = np.asarray(np.random.randn(3,5), dtype=np.float32)

    #     def __getitem__(self, index):
    #         inp = np.asarray(np.random.randn(5), dtype=np.float32)
    #         return {'inp': inp, 'target': self.W @ inp}
        
    #     def __len__(self):
    #         return 100

    # data = Hello()
    # trainer = Trainer('run1', TestC2R,data,data,32).train(51)

    # fix dataset
    # run param
    Trainer('run1', model, train_data, val_data).train(51)
