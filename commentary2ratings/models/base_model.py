import os
import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        raise NotImplementedError("Need to implement this function in the subclass!")
    
    def loss(self, outputs, inputs):
        raise NotImplementedError("Need to implement this function in the subclass!")

    def log_params(self, epoch, logger):
        self.log_gradients(epoch, logger)
        self.log_weights(epoch, logger)

    def log_gradients(self, epoch, logger):
        grad_norms = list([torch.norm(p.grad.data) for p in self.parameters() if p.grad is not None])
        if len(grad_norms) == 0:
            return
        grad_norms = torch.stack(grad_norms)

        logger.add_scalar('gradients/mean_norm', grad_norms.mean(), epoch)
        logger.add_scalar('gradients/max_norm', grad_norms.abs().max(), epoch)

    def log_weights(self, epoch, logger):
        weights = list([torch.norm(p.data) for p in self.parameters() if p.grad is not None])
        if len(weights) == 0:
            return
        weights = torch.stack(weights)

        logger.add_scalar('weights/mean_norm', weights.mean(), epoch)
        logger.add_scalar('weights/max_norm', weights.abs().max(), epoch)
    
    def save_weights(self, epoch, path):
        path = os.path.join(path, 'weights')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, "weights_ep{}.pth".format(epoch))
        torch.save(self.state_dict(), path)
    
    def load_weights(self, epoch, path):
        path = os.path.join(path, "weights_ep{}.pth".format(epoch))
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            return True
        else:
            print("File not found: {}".format(path))
            return False