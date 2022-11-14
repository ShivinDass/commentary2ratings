import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from commentary2ratings.models.base_model import BaseModel

class SeqC2R(BaseModel):

    def __init__(self):
        super().__init__()
        #452

        #                                                                        Fully Connected
        #                     LSTM       (take values of final output)           #
        #   ###########     ###########           ###########                    #     #
        #   #         # --> #         # ..... --> #         # --> embedding -->  # --> # --> #
        #   ###########     ###########           ###########                    #     #
        #                                                                        #
        self.hidden_size = 64
        self.lstm = nn.LSTM(768, self.hidden_size, 2, batch_first = True, dropout = 0.2)
        self.fc1 = nn.Linear(self.hidden_size,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        self.relu = nn.ReLU(0.2)

    
    def forward(self, inputs):

        ####without concat (make sure lstm above does not have +452)
        #h0 = torch.randn(1, inputs['padded_commentary_embedding'].shape[0], 27)
        #c0 = torch.randn(1, inputs['padded_commentary_embedding'].shape[0], 27)
        #print(inputs['padded_commentary_embedding'].shape)
        #h0 = torch.randn(2, inputs['padded_commentary_embedding'].shape[0], 64)
        #c0 = torch.randn(2, inputs['padded_commentary_embedding'].shape[0], 64)
        #print(inputs['padded_commentary_embedding'].shape)
        #print(inputs["commentary_len"][:5])
        ip = inputs['padded_commentary_embedding'].flip((1))  #since emma mentioned that its in the exact opposite order, i've flipped the commentaries
        
        
        ip = pack_padded_sequence(ip, inputs["commentary_len"], batch_first=True, enforce_sorted=False)
        #print(ip)
        
        x,(hn,cn) = self.lstm(ip)
        #print(x.shape)
        
        ###TO CONCAT####
        #I dont think concat can be used when using pad_packed_sequences
        #(make sure lstm above has  +452)

        ##get player##
        #player_inputs = inputs['player']

        ##expand the player array to a third dimension to concat
        #player_inputs.unsqueeze_(-1)
        #player_inputs = player_inputs.transpose(2,1)
        #player_inputs = player_inputs.expand(inputs['player'].shape[0], 27, inputs['player'].shape[1])
        #print(player_inputs.shape)

        #get embeddings
        #embeds = torch.cat((inputs['padded_commentary_embedding'], player_inputs), dim=-1)
        #print(embeds.shape)
        #x,_ = self.lstm(embeds)
        
        x,_ = pad_packed_sequence(x, batch_first=True)
        
        #x = x.view(-1, self.hidden_size)
        #x = x.squeeze()[-1, :]
        #print(x.shape)
        #print(hn.shape)
        #print(cn.shape)
        #print(x.shape)
        
        x = x[:,-1,:] #get last output
        x = self.relu(x)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        y = self.fc3(x)

        #Tried adding a relu at the end, not quite ure if it will help but its not harming it either
        y = self.relu(y)
        #print(y.shape)
        #y = y.view(inputs['padded_commentary_embedding'].shape[0], -1)
        #y = y[:,-1]
        return y
        
    
    def loss(self, outputs, inputs):
        #print(inputs['rating'].shape)
        return nn.MSELoss()(outputs, inputs['rating'])

if __name__ == "__main__":
    from commentary2ratings.commentary2ratings.train import Trainer