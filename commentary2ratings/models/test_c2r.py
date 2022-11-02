import torch.nn as nn
from commentary2ratings.models.base_model import BaseModel

class TestC2R(BaseModel):

    def __init__(self):
        super().__init__()
        self.weights = nn.Linear(5, 3)
    
    def forward(self, inputs):
        return self.weights(inputs['inp'])
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['target'])