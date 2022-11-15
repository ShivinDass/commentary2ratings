import torch
import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from commentary2ratings.models.base_model import BaseModel

class SeqC2R(BaseModel):

    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.encoder = nn.Sequential(
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

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first = True)
        
        self.final_mlp = nn.Sequential(
                        nn.Linear(self.hidden_size+46, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.LeakyReLU(0.2),
                        nn.Linear(self.hidden_size, 2)
                    )

    def forward(self, inputs):
        batch, seq = inputs['padded_commentary_embedding'].shape[:2]

        inp_sizes = inputs['commentary_len'].type(torch.long)
        encoded = self.encoder(inputs['padded_commentary_embedding'].view(batch*seq, -1)).reshape(batch, seq, -1)
        
        packed = pack_padded_sequence(encoded, inp_sizes, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        unpacked = pad_packed_sequence(lstm_out, batch_first=True)[0][torch.arange(batch), inp_sizes-1]

        return self.final_mlp(torch.cat((unpacked, inputs['player_stats']), dim=-1))
    
    def loss(self, outputs, inputs):
        # return nn.MSELoss()(outputs, inputs['rating'])
        mu, log_sigma = outputs[:, 0], outputs[:, 1]
        sigma = log_sigma.exp()
        loss = -1*(-1*((inputs['rating'] - mu) ** 2) / (2 * sigma**2) - log_sigma - math.log(math.sqrt(2*math.pi)))
        return torch.mean(loss)