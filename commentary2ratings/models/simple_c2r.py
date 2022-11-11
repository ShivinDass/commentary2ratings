import torch.nn as nn
from commentary2ratings.models.base_model import BaseModel

class SimpleC2R(BaseModel):

    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        pass
    
    def loss(self, outputs, inputs):
        pass