import torch

class ContinuousGraph():
    
    def __init__(self,h,F,A,F_fd=None):
        self.h = h
        self.F = F
        self.A = A
        self.F_fd = F_fd
        
    def to(self,device):
        self.h = self.h.to(device)
        self.F = self.F.to(device)
        self.A = self.A.to(device)
        if self.F_fd is not None:
            self.F_fd = self.F_fd.to(device)
        return self
    
    
def ContinuousGraphs_from_padding(F,A,Mmax):
    m = len(F)
    h = torch.ones(m,device=F.device,dtype=F.dtype)
    h = torch.nn.functional.pad(h,(0,Mmax-m))
    F = torch.nn.functional.pad(F,(0,0,0,Mmax-m))
    A = torch.nn.functional.pad(A,(0,Mmax-m,0,Mmax-m))
    
    return ContinuousGraph(h,F,A)
    
class BatchedContinuousGraphs():
    
    def __init__(self,h,F,A,F_fd=None):
        self.h = h
        self.F = F
        self.A = A
        self.F_fd = F_fd
        
    def to(self,device):
        self.h = self.h.to(device)
        self.F = self.F.to(device)
        self.A = self.A.to(device)
        if self.F_fd is not None:
            self.F_fd = self.F_fd.to(device)
        return self
    
    def __len__(self):
        return len(self.h)
        
        
        
def BatchedContinuousGraphs_from_list(batch,F_fd=None):
    '''
    Collate Function
    '''
    h = torch.stack([g.h for g in batch])
    F = torch.stack([g.F for g in batch])
    A = torch.stack([g.A for g in batch])
    if F_fd:
        F_fd = torch.stack([g.F_fd for g in batch])
    
    return BatchedContinuousGraphs(h,F,A,F_fd=F_fd)
    
    
    