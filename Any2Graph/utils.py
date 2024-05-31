import torch

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, base_lr,
        warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.base_lr = base_lr
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        if step > self.warmup:
            return self.base_lr * (step/self.warmup)  ** (-0.5)
        return step*self.base_lr/self.warmup
    
def get_std_opt(model,base_lr,warmup):
    return NoamOpt(base_lr, warmup, torch.optim.Adam(model.parameters(),lr=0,betas=(0.9, 0.98),eps=1e-9))

def batched_pairwise_KL(logits,targets):
    return -torch.log_softmax(logits,dim=-1)@torch.permute(targets,(0,2,1)) 

def batched_pairwise_L2(logits,targets):
    XX = torch.sum(logits**2,dim=-1)
    YY = torch.sum(targets**2,dim=-1)
    XY =logits@torch.permute(targets,(0,2,1))
    return XX[:,:,None] + YY[:,None,:] - 2*XY