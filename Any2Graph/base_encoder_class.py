import torch
from torch import nn

class Encoder(nn.Module):
    '''
    Base Encoder Class
    '''
    def __init__(self,
                 config:dict):
        self.model_dim = config['model_dim']
    
    def forward(self,inputs):
        B = len(inputs)
        x = torch.zeros(B,self.Mmax,self.model_dim,device=inputs.device)
        masks = torch.zeros(B,self.Mmax,self.Mmax,device=inputs.device) 
        pos_embed = torch.zeros(B,self.Mmax,self.model_dim,device=inputs.device)
        return x,masks,pos_embed