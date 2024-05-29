import torch 
from torch.utils.data import DataLoader
from numpy import inf
from Any2Graph import Task, Dataset
from time import perf_counter

class Trainer():
    
    def __init__(self,task:Task, dataset_train:Dataset, dataset_test:Dataset, config:dict):
        self.task = task
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.config = config
        self.loss_fn = None
        
    def train(self, model:torch.nn.Module, save_path:str):
        
        device = self.config['device']
        batchsize=self.config['batchsize']
        max_grad_step = self.config['max_grad_step']
        max_grad_norm = self.config['max_grad_norm']
        n_eval_interval  = self.config['n_eval_interval']

        model = model.to(device)

        dataloader_train = DataLoader(self.dataset_train, 
                                      batch_size=batchsize, 
                                      shuffle=True, 
                                      collate_fn=self.task.collate_fn)
        
        dataloader_test = DataLoader(self.dataset_test, 
                                      batch_size=batchsize, 
                                      shuffle=True, 
                                      collate_fn=self.task.collate_fn)
        
        optimizer = self.task.get_optimizer(model,self.config)

        tic_start = perf_counter()

        total_eval_time = 0
        total_loss_time = 0
        best_test_loss = +inf

        grad_step=0

        while grad_step < max_grad_step:
            
            for inputs, padded_targets in dataloader_train:
            
                tic_gradient_step = perf_counter()

                # To device
                tic_to_device = perf_counter()
                inputs,padded_targets = self.task.to_device(inputs,padded_targets,device)
                tac_to_device = perf_counter()
                
                # Forward 
                tic_forward = perf_counter()
                continuous_predictions = model(inputs)
                tac_forward = perf_counter()
                
                # Compute Loss
                tic_loss = perf_counter()
                loss, log_loss = self.loss_fn(continuous_predictions, padded_targets)
                tac_loss = perf_counter()
                
                # Backprop
                tic_backprop = perf_counter()
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                tac_backprop = perf_counter()

                tac_gradient_step = perf_counter()
                
                
                # Logs
                total_loss_time += tac_loss - tic_loss
                total_time = perf_counter() - tic_start
                    
                # Logs
                log =  {'grad_step': grad_step,
                        'lr': optimizer.rate(),
                        'total_time': total_time,
                        'total_loss_time': total_loss_time,
                        'total_eval_time': total_eval_time,
                        'to_device_time': tac_to_device - tic_to_device,
                        'forward_time': tac_forward - tic_forward,
                        'loss_time': tac_loss - tic_loss,   
                        'backprop_time': tac_backprop - tic_backprop,
                        'grad_step_time': tac_gradient_step - tic_gradient_step,
                        }
                
                log = log | log_loss
                
                # Eval
                if grad_step % n_eval_interval == 0 or grad_step == max_grad_step-1:

                    tic_eval = perf_counter()
                    
                    log_test = eval(model,
                                    dataloader_test
                                    )
                    
                    log_train = eval(model,
                                    dataloader_train
                                    )
                    
                    for key in log_test:
                        log[key + ' (test set)'] = log_test[key]
                        log[key + ' (train set)'] = log_train[key]
                        
                    if save_path != None:
                        if log_test['loss'] < best_test_loss:
                            best_test_loss = log_test['loss']
                            torch.save(model.state_dict(),save_path+'/best_model')
                            torch.save(model.state_dict(),save_path+'/latest_model')
                        else:
                            torch.save(model.state_dict(),save_path+'/latest_model')
                        
                    tac_eval = perf_counter()

                    total_eval_time += tac_eval - tic_eval
                
                
                # End Gradient Step
                print(log)
                grad_step+=1
                if grad_step>max_grad_step:
                    break