import torch
import torch.nn as nn
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
        hidden_size = 27
        self.lstm = nn.LSTM(768, hidden_size, 3, batch_first = True, dropout = 0.2)
        self.fc1 = nn.Linear(hidden_size,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,1)

        self.relu = nn.LeakyReLU(0.2)

    
    def forward(self, inputs):

        ####without concat (make sure lstm above does not have +452)
        x,_ = self.lstm(inputs['padded_commentary_embedding'])
        
        ###TO concat####
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
        
        x = x[:,-1,:]
        x = self.relu(x)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        y = self.fc3(x)
        #print(y.shape)
        return y
        
    
    def loss(self, outputs, inputs):
        return nn.MSELoss()(outputs, inputs['rating'])