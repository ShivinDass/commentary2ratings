import torch
import math
import torch.nn as nn
from commentary2ratings.models.base_model import BaseModel

class ProjC2R(BaseModel):

    def __init__(self):
        super().__init__()
        self.hidden_size = 128

        self.encoder_layer = nn.Sequential(
                        nn.Linear(768, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.2)
                    )

        self.regression_layer = nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(self.hidden_size, 2))
    
    def forward(self, inputs):
        batch, seq = inputs['padded_commentary_embedding'].shape[:2]

        # conc_inp = torch.cat((inputs['padded_commentary_embedding'].view(batch*seq, -1), torch.repeat_interleave(inputs['player'], seq, dim=0)), dim=-1)
        projections = self.encoder_layer(inputs['padded_commentary_embedding'].view(batch*seq, -1)).reshape(batch, seq, -1)
        mask = torch.norm(inputs['padded_commentary_embedding'], dim=-1) > 0
        projections = torch.sum(projections*(mask[..., None]), dim=1)/(inputs['commentary_len'][:, None])
        # projections = torch.cat((projections, inputs['player_stats']), dim=-1)
        return self.regression_layer(projections)
        
        
    
    def loss(self, outputs, inputs):
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)
