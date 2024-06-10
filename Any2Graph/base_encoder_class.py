import torch
from torch import nn

class Encoder(nn.Module):
    '''
    Base Encoder Class
    '''
    def __init__(self,
                 config:dict):
        super().__init__()
        self.model_dim = config['model_dim']
        self.N_features = 5
    
    def forward(self,inputs):
        B = len(inputs)
        x = torch.zeros(B,self.N_features,self.model_dim,device=inputs.device)
        masks = torch.zeros(B,self.N_features,device=inputs.device) 
        pos_embed = torch.zeros(B,self.N_features,self.model_dim,device=inputs.device)
        return x,masks,pos_embed