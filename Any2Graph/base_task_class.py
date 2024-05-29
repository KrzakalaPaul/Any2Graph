class Task():
    
    def __init__(self,config) -> None:
        pass
    
    def get_model(self,config):
        '''
        Build Model
        '''
        return None
    
    def get_optimizer(self,model,config):
        '''
        Get Optimizer
        '''
        return None
    
    def collate_fn(self,batch):
        '''
        Collate Function
        '''
        return None
    
    def to_device(self,inputs,targets,device):
        '''
        Send inputs and targets to device
        '''
        return None
        