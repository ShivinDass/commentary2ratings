import torch
import torch.nn as nn
import torch.nn.functional as F
from commentary2ratings.models.base_model import BaseModel
import math

class SimpleC2R(BaseModel):

    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(
        #         nn.Linear(46, 128),
        #         nn.BatchNorm1d(128),
        #         nn.LeakyReLU(0.2),
        #         nn.Linear(128, 64),
        #         nn.BatchNorm1d(64),
        #         nn.LeakyReLU(0.2),
        #         nn.Linear(64, 32),
        #         nn.BatchNorm1d(32),
        #         nn.LeakyReLU(0.2),
        #         nn.Linear(32, 2)
        #     )
        self.model = nn.Sequential(
                        nn.Linear(768, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 2)
                    )
    
    def forward(self, inputs):
        # embeddings = inputs['player_stats']
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        # embeddings = torch.cat((embeddings, inputs['player']), dim=-1)
        return self.model(embeddings)
    
    def loss(self, outputs, inputs):
        # return nn.MSELoss()(outputs, inputs['rating'])
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)
