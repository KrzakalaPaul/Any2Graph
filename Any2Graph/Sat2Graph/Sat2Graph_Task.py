from Any2Graph.base_task_class import Task
from .USCities import USCities_Dataset
from .TOULOUSE import TOULOUSE_Dataset
from .Sat2Graph_Encoder import EncoderTOULOUSE, EncoderUSCities
from Any2Graph.utils import batched_pairwise_L2, batched_pairwise_KL
from Any2Graph.graphs.custom_graphs_classes import BatchedContinuousGraphs_from_list, ContinuousGraphs_from_padding
import numpy as np

class Sat2Graph(Task):
    
    def __init__(self,config) -> None:
        self.config = config

    def get_dataset(self,config,split):
        '''
        Get Dataset
        '''
        dataset = config['dataset']
        
        if dataset == 'TOULOUSE':
            return TOULOUSE_Dataset(split=split,augment_data=config['augment_data'],dataset_size=config['dataset_size'])
        elif dataset == 'USCities':
            return USCities_Dataset(split=split,augment_data=config['augment_data'],dataset_size=config['dataset_size'])
        else:
            raise ValueError('Unknown dataset')
    
    def get_encoder(self):
        '''
        Get Encoder
        '''
        dataset = self.config['dataset']
        
        if dataset == 'TOULOUSE':
            return EncoderTOULOUSE(model_dim=self.config['model_dim'],positionnal_encodings='learned')
        elif dataset == 'USCities':
            return EncoderUSCities(model_dim=self.config['model_dim'],positionnal_encodings='learned')
        else:
            raise ValueError('Unknown dataset')
    
    def F_from_logits(self,F_logits):
        '''
        Get F from logits
        Default: softmax (change to identity if F is not one hot encoded)
        '''
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