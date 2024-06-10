import torch

class Dataset():
    
    def __init__(self,config,split) -> None:
        self.x = torch.rand(10)
        F = torch.rand(3,2)
        A = torch.rand(3,3)
        A = torch.where((A+A.T)/2>0.5,1,0)
        self.y = {'F': F,'A': A}

    
    def __len__(self):
        return 1000
    
    def __getitem__(self,idx):
        x = self.x
        y = self.y
        return x,y
    