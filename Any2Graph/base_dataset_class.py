import torch

class Dataset():
    
    def __init__(self,config,split) -> None:
        pass
    
    def __len__(self):
        return 10
    
    def __getitem__(self,idx):
        return torch.ones(10)
    