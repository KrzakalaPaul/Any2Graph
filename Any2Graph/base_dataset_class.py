import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    
    def __init__(self) -> None:
        self.x = torch.rand(10)
        F = torch.tensor([[-1,-1],
                          [-1,-1],
                          [0,0],
                          [1,1],
                          [1,1]],dtype=torch.float32)
        A = torch.tensor([[0,0,0,0,1],
                          [0,0,0,0,0],
                          [0,0,0,0,0],
                          [0,0,0,0,0],
                          [1,0,0,0,0]],dtype=torch.float32)
        self.y = {'F': F,'A': A}

    
    def __len__(self):
        return 1000
    
    def __getitem__(self,idx):
        x = self.x
        y = self.y
        return x,y,idx
    
    def plot_img(self,index,ax):
        pass
    
    def plot_trgt(self,index_trgt,ax_trgt):
        pass
    
    def plot_pred(self,F,A,index_trgt,ax_pred):
        pass
