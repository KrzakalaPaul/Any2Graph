from Any2Graph.base_task_class import Task
import torch
from .utils_ import init_matrix_quad_batch, tensor_product_quad_batch, solver_linear_batch, solver_quad_batch

class PMFGW():
    
    def __init__(self,
                 task:Task,
                 config:dict):
        
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.max_iter_inner = config['max_iter_inner']
        self.alpha_h = config['alpha_h']
        self.alpha_F = config['alpha_F']
        self.alpha_F_fd = config['alpha_F_fd']
        self.alpha_A = config['alpha_A']
        self.FD = config['FD']
        self.Mmax = config['Mmax']
        self.Hungarian = config['Hungarian']
        self.mask_self_loops = config['mask_self_loops']
        self.linear_matching = config['linear_matching']
        self.task = task

        
    def __call__(self,continuous_predictions, padded_targets):
        
        # Get predictions and targets
        
        h = padded_targets.h
        F = padded_targets.F
        A = padded_targets.A
        if self.FD:
            padded_targets.F_fd = torch.bmm(F,A)
        
        h_logits = continuous_predictions.h
        F_logits = continuous_predictions.F
        A_logits = continuous_predictions.A
        F_fd_logits = continuous_predictions.F_fd
        
        weights = self.Mmax*h/torch.sum(h,1,keepdim=True)
        
        # Init Cost matrix h
        h_logits_01 = torch.stack([h_logits,torch.zeros_like(h_logits)],-1)
        h_targets_01 = torch.stack([h,1-h],-1)
        M_weight = -torch.log_softmax(h_logits_01,dim=-1)@torch.permute(h_targets_01,(0,2,1)) - torch.sum(torch.special.entr(h_targets_01),-1)[:,None,:]
        M_weight = self.alpha_h*M_weight
        
        # Init Cost matrix F
        M_F = self.alpha_F*self.task.F_cost(F_logits,F)*weights[:,None,:]
        if self.FD:
            M_F_fd = self.alpha_F_fd*self.task.F_fd_cost(F_fd_logits,padded_targets)*weights[:,None,:]
        else:
            M_F_fd = torch.zeros_like(M_F)
            
        M =  M_weight + M_F + M_F_fd
            
        # Init Cost matrix A
        L = init_matrix_quad_batch(A_logits=A_logits,A=A,w=weights,alpha=self.alpha_A,mask_self_loops=self.mask_self_loops)

        # Get OT plan 
        with torch.no_grad():
            if self.linear_matching:
                T,log_solver = solver_linear_batch(M=M,max_iter_inner=self.max_iter_inner,log=True)
            else:
                T,log_solver = solver_quad_batch(M=M,L=L,max_iter=self.max_iter,tol=self.tol,max_iter_inner=self.max_iter_inner,mask_self_loops=self.mask_self_loops,Hungarian=self.Hungarian,log=True)
            T = T.to(device=M.device)
            
        # Forward
        
        losses_h = torch.sum(T*M_weight,dim=(1,2))
        losses_F = torch.sum(T*M_F,dim=(1,2))
        losses_F_fd = torch.sum(T*M_F_fd,dim=(1,2))
        LT = tensor_product_quad_batch(L, T,mask_self_loops=self.mask_self_loops)
        losses_A = torch.sum(T*LT,dim=(1,2))
        losses = losses_h + losses_F + losses_F_fd + losses_A

        loss = torch.mean(losses)
        
        log = {'loss (batch)': loss.item(),
               'loss_F (batch)': torch.sum(losses_F).item()/self.alpha[0],
               'loss_F (batch)': torch.sum(losses_F).item()/self.alpha[0], 
               'loss_F (batch)': torch.sum(losses_F).item()/self.alpha[0],
               'loss_F (batch)': torch.sum(losses_F).item()/self.alpha[0],
               'avg_cg_iter (batch)': log_solver['avg_cg_iter'],
                }
               
        return loss,log
        
        
        