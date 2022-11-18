import torch.nn as nn
import torch.nn.functional as F
from commentary2ratings.models.base_model import BaseModel

class SimpleC2R_MSE(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768, 128),#+452 /// 256
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 1)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        return self.model(embeddings)
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])

class SimpleC2R_MSE_Stats(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768+452, 128),#+452 /// 256
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 1)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        embedding_and_player = torch.cat((embeddings, inputs['player_stats']), dim=-1)
        return self.model(embedding_and_player)
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])   


class SimpleC2R_MSE_BN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768, 128),#+452 /// 256
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 1)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        return self.model(embeddings)
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])  

class SimpleC2R_MSE_BN_Stats(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768+452, 128),#+452 /// 256
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 1)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        embedding_and_player = torch.cat((embeddings, inputs['player_stats']), dim=-1)
        return self.model(embedding_and_player)
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])   

class SimpleC2R_NLL(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768, 128),#+452 /// 256
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 2)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        #embedding_and_player = torch.cat((embeddings, inputs['player_stats']), dim=-1)
        return self.model(embeddings)
        #return self.model(embedding_and_player)
    
    def loss(self, outputs, inputs):
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)

class SimpleC2R_NLL_Stats(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768+452, 128),#+452 /// 256
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 2)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        embedding_and_player = torch.cat((embeddings, inputs['player_stats']), dim=-1)
        return self.model(embedding_and_player)
    
    def loss(self, outputs, inputs):
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)  


class SimpleC2R_NLL_BN(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768, 128),#+452 /// 256
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 2)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        return self.model(embeddings)
    
    def loss(self, outputs, inputs):
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)

class SimpleC2R_NLL_BN_Stats(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768+452, 128),#+452 /// 256
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 2)
                    )
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        embedding_and_player = torch.cat((embeddings, inputs['player_stats']), dim=-1)
        return self.model(embedding_and_player)
    
    def loss(self, outputs, inputs):
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)