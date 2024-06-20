from Any2Graph.base_task_class import Task
from .Coloring import Coloring
from .Img2Graph_Encoder import EncoderCNN
from Any2Graph.utils import batched_pairwise_L2, batched_pairwise_KL
from Any2Graph.graphs.custom_graphs_classes import BatchedContinuousGraphs_from_list, ContinuousGraphs_from_padding
import torch 

class Img2Graph(Task):
    
    def __init__(self,config) -> None:
        self.config = config
        
    def get_dataset(self,config,split):
        '''
        Get Dataset
        '''
        return Coloring(config,split)
    
    def get_encoder(self):
        '''
        Get Encoder
        '''
        return EncoderCNN(self.config)
    
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
        return torch.softmax(F_logits,dim=-1)
    
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
        M = batched_pairwise_KL(F_logits,F)
        return M

    def F_fd_cost(self,F_fd_logits,F_fd):
        '''
        Get F_fd cost
        Default: L2 distance
        i.e. M_kij =  ||\hat{F}_{ki} - F_{kj}||_2^2
        '''
        M = batched_pairwise_L2(F_fd_logits,F_fd)
        return M
    
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
        
        
