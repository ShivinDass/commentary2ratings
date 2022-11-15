import torch
import torch.nn as nn
from commentary2ratings.models.base_model import BaseModel

class TestC2R(BaseModel):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(768+452+46, 512),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(0.2),
                        nn.Linear(128, 1)
                    )
                        
    def forward(self, inputs):
        embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
        embedding_and_player = torch.cat((embeddings, inputs['player'], inputs['player_stats']), dim=-1)
        return self.model(embedding_and_player)

    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])

# class TestC2R(BaseModel):

#     def __init__(self):
#         super().__init__()
#         self.projection_mlp = nn.Sequential(
#                         nn.Linear(768+452, 512),
#                         nn.LeakyReLU(0.2),
#                         nn.Linear(512, 256),
#                         nn.LeakyReLU(0.2),
#                         nn.Linear(256, 128),
#                         nn.LeakyReLU(0.2)
#                     )
        
#         self.final_mlp = nn.Sequential(
#                         nn.Linear(128, 128),
#                         nn.LeakyReLU(0.2),
#                         nn.Linear(128, 128),
#                         nn.Linear(128, 1)
#                     )
                        
#     def forward(self, inputs):
#         batch, seq = inputs['padded_commentary_embedding'].shape[:2]

#         conc_inp = torch.cat((inputs['padded_commentary_embedding'].view(batch*seq, -1), torch.repeat_interleave(inputs['player'], seq, dim=0)), dim=-1)
#         projections = self.projection_mlp(conc_inp).reshape(batch, seq, -1)

#         mask = torch.norm(inputs['padded_commentary_embedding'], dim=-1) > 0
#         projections = torch.sum(projections*(mask[..., None]), dim=1)/(inputs['commentary_len'][:, None])

#         # embeddings = torch.mm(projections, (torch.sum(inputs['padded_commentart_embedding'], dim=1)))
#         # embeddings = torch.sum(inputs['padded_commentary_embedding'], dim=1)/inputs['commentary_len'][:, None]
#         # embedding_and_player = torch.cat((embeddings, inputs['player']), dim=-1)
#         return self.final_mlp(projections)
    
#     def loss(self, outputs, inputs):
#         return nn.MSELoss()(outputs, inputs['rating'])