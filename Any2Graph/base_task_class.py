from.base_dataset_class import Dataset
from .base_encoder_class import Encoder
import torch
from .utils import NoamOpt, batched_pairwise_KL, batched_pairwise_L2
from .graphs.custom_graphs_classes import BatchedContinuousGraphs_from_list, ContinuousGraphs_from_padding
import numpy as np

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
    
    def F_from_logits(self,F_logits):
        '''
        Get F from logits
        Default: softmax (change to identity if F is not one hot encoded)
        '''
        #return torch.softmax(F_logits,dim=-1)
        return F_logits
    
    def F_fd_from_logits(self,F_fd_logits):
        '''
        Get F_fd from logits
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
        indices = []
        for x,y,idx in batch:
            inputs.append(x)
            indices.append(idx)
            padded_targets.append(ContinuousGraphs_from_padding(y['F'],y['A'],self.config['Mmax']))
        inputs = torch.stack(inputs,dim=0)
        padded_targets = BatchedContinuousGraphs_from_list(padded_targets)
        return inputs, padded_targets, indices

    def inputs_to_device(self,inputs,device):
        '''
        Send inputs to device
        Ex for text: 
        tokens, mask = inputs
        return tokens.to(device), mask.to(device)
        '''
        return inputs.to(device)
        
    def is_same_feature(self,F1,F2):
        '''
        F1 of size nxd is the feature matrix of graph 1
        F2 of size nxd is the feature matrix of graph 2
        return a boolean vector of size n where True means that the feature is the same in both graphs
        '''
        treshold = 0.1
        return np.where(np.linalg.norm(F1 -F2,ord=2,axis=1)< treshold, 1,0)
    
    def is_same_feature_matrix(self,F1,F2):
        '''
        F1 of size nxd is the feature matrix of graph 1
        F2 of size mxd is the feature matrix of graph 2
        return a boolean matrix of size nxm, where M_ij = 1 means that the feature i of graph 1 is the same as the feature j of graph 2
        '''
        treshold = 0.1
        return np.where(np.linalg.norm(F1[None,:,:] - F2[:,None,:],ord=2,axis=-1)< treshold, 1,0)
        