import torch.nn as nn
import torch.nn.functional as F
from commentary2ratings.models.base_model import BaseModel

class SimpleC2R(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768+452, 256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 1)
                    )
    
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        embedding_and_player = torch.cat((embeddings, inputs['player']), dim=-1)
        return self.model(embedding_and_player)
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])