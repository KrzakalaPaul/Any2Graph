from.base_dataset_class import Dataset
from .base_encoder_class import Encoder
import torch
from .utils import NoamOpt, batched_pairwise_KL, batched_pairwise_L2
from .graphs.custom_graphs_classes import BatchedContinuousGraphs_from_list, ContinuousGraphs_from_padding

class Task():
    '''
    You must subclass this class in order to use Any2Graph.
    All methods must be implemented.
    '''
    
    def __init__(self,config) -> None:
        self.config = config
    

    def get_dataset(self,config,split):
        '''
        Get Dataset
        '''
        return Dataset(config,split)
    
    def get_encoder(self):
        '''
        Get Encoder
        '''
        return Encoder(self.config)
    
    def get_loss_fn(self):
        '''
        Get Loss Function
        '''
        return torch.nn.MSELoss()
    
    def F_from_logits(self,F_logits):
        '''
        Get F from logits
        Default: softmax (change to identity if F is not one hot encoded)
        '''
        #return torch.softmax(F_logits,dim=-1)
        return F_logits
    
    def F_fd_from_logits(self,F_fd_logits):
        '''
        Get F from logits
        Default: identity (diffused feature are not one hot encoded)
        '''
        return F_fd_logits
    
    def F_cost(self,F_logits,F):
        '''
        Get F cost
        Default: KL divergence
        i.e. M_kij = KL( \hat{F}_{ki}, F_{kj})
        '''
        M = batched_pairwise_L2(F_logits,F)
        return M

    def F_fd_cost(self,F_fd_logits,F_fd):
        '''
        Get F_fd cost
        Default: L2 distance
        i.e. M_kij =  ||\hat{F}_{ki} - F_{kj}||_2^2
        '''
        M = batched_pairwise_L2(F_fd_logits,F_fd)
        return M
    
    def get_optimizer(self,model):
        '''
        Get Optimizer
        '''
        return NoamOpt(self.config['lr'],self.config['warmup'],torch.optim.Adam(model.parameters(),lr=0,betas=(0.9, 0.98),eps=1e-9))
    
    def collate_fn(self,batch):
        '''
        Collate Function
        '''
        inputs = []
        padded_targets = []
        for x,y in batch:
            inputs.append(x)
            padded_targets.append(ContinuousGraphs_from_padding(y['F'],y['A'],self.config['Mmax']))
        inputs = torch.stack(inputs,dim=0)
        padded_targets = BatchedContinuousGraphs_from_list(padded_targets)
        return inputs, padded_targets

    def inputs_to_device(self,inputs,device):
        '''
        Send inputs to device
        Ex for text: 
        tokens, mask = inputs
        return tokens.to(device), mask.to(device)
        '''
        return inputs.to(device)
        
        