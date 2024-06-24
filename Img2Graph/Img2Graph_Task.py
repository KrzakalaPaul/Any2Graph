from Any2Graph.base_task_class import Task
from .Coloring.Coloring_Dataset import ColoringDataset
from .Img2Graph_Encoder import TroncatedResNet
from Any2Graph.utils import batched_pairwise_L2, batched_pairwise_KL
from Any2Graph.graphs.custom_graphs_classes import BatchedContinuousGraphs_from_list, ContinuousGraphs_from_padding
import torch 
import numpy as np

class Img2Graph(Task):
    
    def __init__(self,config) -> None:
        self.config = config
        
        dataset = config['dataset']
        
        if dataset == 'ColoringSmall':
            self.subset = 'small'
            self.input_shape = (3,32,32)
        elif dataset == 'ColoringMedium':
            self.subset = 'medium'
            self.input_shape = (3,32,32)
        elif dataset == 'ColoringLarge':
            self.subset = 'large'
            self.input_shape = (3,32,32)
        elif dataset == 'ColoringMicro':
            self.subset = 'micro'
            self.input_shape = (3,32,32)
        
    def get_dataset(self,config,split):
        '''
        Get Dataset
        '''
        return ColoringDataset(root_path='Img2Graph/Coloring/data/',subset=self.subset,split=split,augment_data=config['augment_data'],dataset_size=config['dataset_size'])
    
    def get_encoder(self):
        '''
        Get Encoder
        '''

        return TroncatedResNet(model_dim=self.config['model_dim'],input_shape=self.input_shape,positionnal_encodings='learned')
    
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