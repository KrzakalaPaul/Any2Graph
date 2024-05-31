import torch.nn as nn
import torch
from .utils_ import Transformer, MLP
from Any2Graph.graphs.custom_graphs_classes import BatchedContinuousGraphs
from Any2Graph.base_task_class import Task

class Decoder(nn.Module):
    def __init__(self,
                 config:dict):
        
        super().__init__()
        
        model_dim = config['model_dim']
        dropout = config['dropout']
        node_feature_dim = config['node_feature_dim']
        MLP_layers = config['MLP_layers']
        self.virtual_node = config['virtual_node']
        self.Mmax = config['Mmax']
        self.FD = config['FD']
        
        self.head_h = MLP(input_dim=model_dim,hidden_dim=4*model_dim,output_dim=1,num_layers=MLP_layers,dropout=dropout)
        self.head_F = MLP(input_dim=model_dim,hidden_dim=4*model_dim,output_dim=node_feature_dim,num_layers=MLP_layers,dropout=dropout)
        
        input_dim1 = 2*model_dim if self.virtual_node else model_dim
        self.head_A_1 = MLP(input_dim=input_dim1,hidden_dim=4*model_dim,output_dim=model_dim,num_layers=MLP_layers,dropout=dropout)
        input_dim2 = model_dim 
        self.head_A_2 = MLP(input_dim=input_dim2,hidden_dim=4*model_dim,output_dim=1,num_layers=MLP_layers,dropout=dropout)         
        
        if self.FD is not None:
            self.head_F_fd = MLP(input_dim=model_dim,hidden_dim=4*model_dim,output_dim=node_feature_dim,num_layers=MLP_layers,dropout=dropout)

    def forward(self,x:torch.Tensor):

        virtual_node = x[:,0,:]
        true_nodes = x[:,1:,:]
        
        h_logits = self.NodeWeihead_hghtHead(true_nodes).squeeze()
        F_logits = self.head_h(true_nodes)
        
        if self.virtual_node:
            x = torch.concat([true_nodes,virtual_node.unsqueeze(1).repeat(1,self.Mmax,1)],dim=2)
        else:
            x = true_nodes
        x = self.head_A_1(x)
        A_logits = self.head_A_2(x[:,None,:,:] + x[:,:,None,:])
        
        A_logits = A_logits.squeeze()
            
        mask = ~torch.eye(self.Mmax,dtype=bool,device=x.device)
        A_logits = A_logits*mask[None,:,:] - torch.eye(self.Mmax,dtype=bool,device=x.device) 

        if self.FD:
            F_fd_logits = self.head_F_fd(true_nodes)
        else:
            F_fd_logits = None
            
        return h_logits,F_logits,A_logits,F_fd_logits
    
    
def get_transformer(config:dict):
    model_dim = config['model_dim']
    transformer_heads = config['transformer_heads']
    transformer_layers = config['transformer_layers']
    dropout = config['transformer_dropout']
    Mmax = config['Mmax']
    pre_norm = config['pre_norm']
    return Transformer(Mmax=Mmax,model_dim=model_dim,num_layer=transformer_layers,nhead=transformer_heads,dropout=dropout,normalize_before=pre_norm)

class Any2Graph_Model(nn.Module):

    def __init__(self,
                 task:Task,
                 config:dict):
        
        super().__init__()
        
        self.task = task
        self.encoder = task.get_encoder()
        self.transformer = get_transformer(config)
        self.decoder = Decoder(config)


    def forward(self,inputs,logits=False):
        
        set_of_features,mask,pos_embed = self.encoder(inputs)
        set_of_nodes = self.transformer(set_of_features,mask,pos_embed)
        h_logits,F_logits,A_logits,F_fd_logits = self.decoder(set_of_nodes)
        
        if logits:
            continuous_predictions = BatchedContinuousGraphs(h = h_logits, F = F_logits, A = A_logits, F_fd = F_fd_logits)
        else:
            h = torch.sigmoid(h_logits)  
            F = self.task.F_from_logits(F_logits)
            A = torch.sigmoid(A_logits)
            mask = ~torch.eye(self.decoder.Mmax,dtype=bool,device=A.device)
            A = A*mask[None,:,:]
            F_fd = self.task.F_fd_from_logits(F_fd_logits)

            continuous_predictions = BatchedContinuousGraphs(h = h, F = F, A = A, F_fd = F_fd)

        return continuous_predictions
