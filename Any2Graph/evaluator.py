import torch 
import numpy as np
from torch.utils.data import DataLoader
from Any2Graph import Task, Dataset
from Any2Graph.PMFGW import PMFGW
from scipy.optimize import linear_sum_assignment
import pandas as pd

class Evaluator():
    
    def __init__(self, task:Task, dataset:Dataset, config:dict):
        self.task = task
        self.dataset = dataset
        self.config = config
        self.loss_fn = PMFGW(task, config)
        
    def eval_single(self, h_pred, F_pred, A_pred, h_trgt, F_trgt, A_trgt):

        metrics = {}
        
        # Tresholding
        
        m = int(np.sum(h_trgt))
        
        Mmax = h_pred.shape[0]
        
        if m>Mmax:
            h_trgt = h_trgt[:Mmax]
            F_trgt = F_trgt[:Mmax]
            A_trgt = A_trgt[:Mmax,:Mmax]
            m = Mmax
            print('Warning: target graph is too large for the model')
        
        # Decoding step
        
        indices = h_pred.argsort()
        indices = indices[-m:]

        F_pred = F_pred[indices]
        A_pred = A_pred[indices][:,indices]
        
        h_pred = np.where(h_pred>0.5,1,0)
        A_pred = np.where(A_pred>0.5,1,0)
        
        # Graph matching 
        T = matching_numpy(F_pred,A_pred,F_trgt,A_trgt,max_iter=200,beta=100,sk_iter=100,treshold_nodes_features=self.task.treshold_nodes_features())
        
        # Reordering
        permutation_trgt, permutation_pred = linear_sum_assignment(-T.T)
        
        F_trgt = F_trgt[permutation_trgt]
        A_trgt = A_trgt[permutation_trgt][:,permutation_trgt]
        
        F_pred = F_pred[permutation_pred]
        A_pred = A_pred[permutation_pred][:,permutation_pred]
        
        edges_pred = np.concatenate([A_pred[k,k+1:] for k in range(m)])
        edges_trgt = np.concatenate([A_trgt[k,k+1:] for k in range(m)])
        
        ## SIZE ACCURACY 
        m_pred = int(np.sum(h_pred))
        metrics['Size Accuracy'] = int(m_pred == m)
        
        ## NODE ACCURACY
        metrics['Nodes Accuracy'] = np.where(np.linalg.norm(F_pred -F_trgt,ord=2,axis=1)< self.task.treshold_nodes_features(), 1,0).sum().item() / m
        
        ## EDGE ACCURACY
        metrics['Edge Precision'] = (np.sum(edges_pred*edges_trgt)/(np.sum(edges_pred)+1e-6)).item()
        metrics['Edge Recall'] = (np.sum(edges_pred*edges_trgt)/(np.sum(edges_trgt)+1e-6)).item()
        
        ## Edit Distance
        metrics['Edit Distance'] = np.sum(edges_pred!=edges_trgt) + np.where(np.linalg.norm(F_pred -F_trgt,ord=2,axis=1)<self.task.treshold_nodes_features() , 0,1).sum()
        metrics['GI Accuracy'] = int(metrics['Edit Distance'] < 1e-6)
        
        return metrics
    
    
    def eval(self, model:torch.nn.Module, n_samples = 1000, batchsize = 128):
        
        dataloader = DataLoader(self.dataset, 
                                batch_size=batchsize, 
                                shuffle=False, 
                                collate_fn=self.task.collate_fn)
        
        device = self.config['device']
        model = model.to(device)
        model.eval()
        
        size = 0
        
        metrics = {'Edit Distance': [],
                   'GI Accuracy': [],
                   'PMFGW': [],
                   'PMFGW h': [],
                   'PMFGW F': [],
                   'PMFGW F_fd': [],
                   'PMFGW A': [],
                   'Edge Precision': [],
                   'Edge Recall': [],
                   'Nodes Accuracy': [],
                   'Size Accuracy': [],
                   }
        
        for inputs, padded_targets, indices in dataloader:
            
            # To device
            inputs = self.task.inputs_to_device(inputs,device)
            padded_targets = padded_targets.to(device)
            
            # Forward 
            continuous_predictions = model(inputs,logits=True)
            
            # PMGW Loss
            _, log_loss = self.loss_fn(continuous_predictions, padded_targets)
            
            metrics['PMFGW h']+=list(log_loss['loss h (batch)'])
            metrics['PMFGW F']+=list(log_loss['loss F (batch)'])
            metrics['PMFGW F_fd']+=list(log_loss['loss Fdiff (batch)'])
            metrics['PMFGW A']+=list(log_loss['loss A (batch)'])
            metrics['PMFGW']+=list(log_loss['loss (batch)'])
            
            # Post Process Logits
            
            continuous_predictions.h = torch.sigmoid(continuous_predictions.h )  
            continuous_predictions.F = self.task.F_from_logits(continuous_predictions.F)
            A = torch.sigmoid(continuous_predictions.A)
            mask = ~torch.eye(self.decoder.Mmax,dtype=bool,device=A.device)
            continuous_predictions.A = A*mask[None,:,:]
            if continuous_predictions.F_fd is not None:
                continuous_predictions.F_fd = self.task.F_fd_from_logits(continuous_predictions.F_fd)
            
            # Discrete Metrics
            
            batchsize = len(inputs)
            for i in range(batchsize):
                
                h_pred = continuous_predictions.h[i].detach().cpu().numpy()
                F_pred = continuous_predictions.F[i].detach().cpu().numpy()
                A_pred = continuous_predictions.A[i].detach().cpu().numpy()
                
                h_trgt = padded_targets.h[i].detach().cpu().numpy()
                F_trgt = padded_targets.F[i].detach().cpu().numpy()
                A_trgt = padded_targets.A[i].detach().cpu().numpy()
                
                metrics_single = self.eval_single(h_pred, F_pred, A_pred, h_trgt, F_trgt, A_trgt)
                
                for key in metrics_single.keys():
                    metrics[key].append(metrics_single[key])
                    
            size += batchsize
            if size > n_samples:
                break   
            
        return pd.DataFrame(metrics)
  
    
    def plot_prediction(self, model, save_path, n_samples = 10):
        pass
    
    
import pygmtools as pygm

def matching_numpy(F1,A1,F2,A2,max_iter=100,sk_iter=100,beta=100,treshold_nodes_features=1e-6):
    
    m = len(F1)
    
    S = np.where(A1[None,:,None,:] == A2[:,None,:,None], 1 , 0)/2   
    M = np.where(np.linalg.norm(F1[None,:,:] - F2[:,None,:],ord=2,axis=-1)< treshold_nodes_features, 1,0)

    K = S.reshape(m**2,m**2)
                       
    for i in range(m):
        for k in range(m):
            for l in range(m):
                K[i+m*k,i+m*l] = 0
    for k in range(m):
        for i in range(m):
            for j in range(m):
                K[i+m*k,j+m*k] = 0
                
    np.fill_diagonal(K,M.reshape(-1))

    X = pygm.rrwm(K, m, m, max_iter=max_iter, sk_iter = sk_iter,beta=beta)
    X = pygm.hungarian(X)

    return X

'''
def matching_torch(F1,A1,F2,A2,max_iter=100,sk_iter=100,beta=100,treshold2d=0.1):
    
    m = len(F1)
    
    S = torch.where(A1[None,:,None,:] == A2[:,None,:,None], 1 , 0)/2   
    M = torch.where(torch.linalg.vector_norm(F1[None,:,:] - F2[:,None,:],ord=2,dim=-1)< treshold2d, 1,0)

    K = S.reshape(m**2,m**2)
                       
    for i in range(m):
        for k in range(m):
            for l in range(m):
                K[i+m*k,i+m*l] = 0
    for k in range(m):
        for i in range(m):
            for j in range(m):
                K[i+m*k,j+m*k] = 0
                
    torch.fill_diagonal_(K,M.reshape(-1))

    X = pygm.rrwm(K, m, m, max_iter=max_iter, sk_iter = sk_iter,beta=beta)
    X = pygm.hungarian(X)

    return X
'''