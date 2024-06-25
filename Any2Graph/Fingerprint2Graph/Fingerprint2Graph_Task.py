from Any2Graph.base_task_class import Task
from .QM9.QM9_dataset import QM9_Dataset
from .GDB13.GDB13_dataset import GDB13_Dataset
from .Fingerprint2Graph_Encoder import TokenEncoder
from Any2Graph.utils import batched_pairwise_L2, batched_pairwise_KL
import torch 
import numpy as np
from torchtext.functional import to_tensor
from Any2Graph.graphs.custom_graphs_classes import BatchedContinuousGraphs_from_list, ContinuousGraphs_from_padding

class Fingerprint2Graph(Task):
    
    def __init__(self,config) -> None:
        self.config = config
        
    def get_dataset(self,config,split):
        '''
        Get Dataset
        '''
        dataset = config['dataset']
        
        if dataset == 'QM9':
            return QM9_Dataset(split=split,dataset_size=config['dataset_size'])
        elif dataset == 'GDB13':
            return GDB13_Dataset(split=split,dataset_size=config['dataset_size'])
        else:
            raise ValueError('Unknown dataset')
    
    def get_encoder(self):
        '''
        Get Encoder
        '''

        return TokenEncoder(model_dim=self.config['model_dim'], vocab_len = 2048 + 2, pos_embed= True) # 2048 + 2 for ECFP4 + <sos> token + <unk> token
    
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
    
    def collate_fn(self,data):   
        token_batch = []
        padded_targets = []
        mask_batch = []
        idx_batch = []
        for tokens,graph,idx in data:
            token_batch.append(tokens)
            padded_targets.append(ContinuousGraphs_from_padding(graph['F'],graph['A'],self.config['Mmax']))
            mask_batch.append([0]*len(tokens))
            idx_batch.append(idx)
        token_batch = to_tensor(token_batch,padding_value=0,dtype=torch.long)
        mask_batch = to_tensor(mask_batch,padding_value=1,dtype=torch.bool)
        inputs = {'tokens':token_batch,'masks':mask_batch}
        padded_targets = BatchedContinuousGraphs_from_list(padded_targets)
        return inputs,padded_targets,idx_batch
    

    def inputs_to_device(self,inputs,device):
        '''
        Send inputs to device
        Ex for text: 
        tokens, mask = inputs
        return tokens.to(device), mask.to(device)
        '''
        token_batch = inputs['tokens'].to(device)
        mask_batch = inputs['masks'].to(device)
        inputs = {'tokens':token_batch,'masks':mask_batch}
        return inputs

    def is_same_feature(self,F1,F2):
        '''
        F1 of size nxd is the feature matrix of graph 1
        F2 of size nxd is the feature matrix of graph 2
        return a boolean vector of size n where True means that the feature is the same in both graphs
        '''
        return np.argmax(F1,-1) == np.argmax(F2,-1)
    
    def is_same_feature_matrix(self,F1,F2):
        '''
        F1 of size nxd is the feature matrix of graph 1
        F2 of size mxd is the feature matrix of graph 2
        return a boolean matrix of size nxm, where M_ij = 1 means that the feature i of graph 1 is the same as the feature j of graph 2
        '''
        return np.where(np.argmax(F1,-1)[None,:] == np.argmax(F2,-1)[:,None], 1, 0)