import torch.nn as nn
from commentary2ratings.models.base_model import BaseModel

class ProjC2R(BaseModel):

    def __init__(self):
        super().__init__()
        self.encoder_layer=nn.Sequential(nn.Linear(768, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128,64),
                        nn.LeakyReLU(0.2))
        self.regression_layer=nn.Linear(64+46,2)
    
    def forward(self, inputs):
        batch, seq = inputs['padded_commentary_embedding'].shape[:2]
        inp_sizes = inputs['commentary_len'].type(torch.long)
        encoded = self.encoder_layer(inputs['padded_commentary_embedding'].view(batch*seq, -1)).reshape(batch, seq, -1)
        embeddings = torch.sum(encoded, dim=1)/inp_sizes[:,None]
        embedding_and_player = torch.cat((embeddings, inputs['player_stats']), dim=-1)
        return self.regression_layer(embedding_and_player)
        
    
    def loss(self, outputs, inputs):
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)
