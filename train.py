from Any2Graph import Trainer, Any2Graph_Model
from Img2Graph import Img2Graph


### ------------------------ LOAD CONFIG ------------------------ ###

config_run = {'dataset': 'ColoringMedium',
               'dataset_size': -1,
               'wandb': True,
               'run_name': None,
               'augment_data': True,
               }

config_optim = {'device':'cuda', 
                'batchsize': 128, 
                'max_grad_step': 100000, 
                'max_grad_norm': 0.1, 
                'n_eval_interval': 1000,
                'lr':3*1e-4,
                'warmup': 8000,
                }

config_model = {'Mmax': 15,
                'model_dim': 256,
                'dropout': 0.,
                'node_feature_dim': 4,
                'MLP_layers': 3,
                'virtual_node': True,
                'transformer_layers': 4,
                'transformer_heads': 4,
                'transformer_dropout': 0.,
                'pre_norm': True,
                }

config_loss = {'max_iter': 20,
               'tol': 1e-5,
               'max_iter_inner': 1000,
               'alpha_h': 1,
               'alpha_F': 1,
               'alpha_F_fd': 1,
               'alpha_A': 1,
               'FD': True,
               'Hungarian': False,
               'mask_self_loops': True,
               'linear_matching': False,
                }                            

config = config_run | config_optim | config_model | config_loss

### ------------------------ LOAD TASK ------------------------ ###

task = Img2Graph(config)

### ------------------------ LOAD DATASET ------------------------ ###

dataset_train = task.get_dataset(config, split = 'train')
print(f'Train set size: {len(dataset_train)}')
dataset_test = task.get_dataset(config, split = 'valid')
print(f'Test set size: {len(dataset_test)}')

### ------------------------ INIT MODEL ------------------------ ###

model = Any2Graph_Model(task, config)

### ------------------------ TRAIN ------------------------ ###

trainer = Trainer(task, dataset_train, dataset_test, config)
trainer.train(model, save_path = None)


'''
from Any2Graph import Trainer, Task
from Any2Graph.model import Any2Graph_Model, Constant_Model

config_optim = {'device':'cuda', 
                'batchsize': 128, 
                'max_grad_step': 100000, 
                'max_grad_norm': 1, 
                'n_eval_interval': 1000,
                'lr':1e-1,
                'warmup': 0,
                }

config_model = {'Mmax': 5,
                'model_dim': 128,
                'dropout': 0.1,
                'node_feature_dim': 2,
                'MLP_layers': 2,
                'virtual_node': True,
                'transformer_layers': 3,
                'transformer_heads': 4,
                'transformer_dropout': 0.1,
                'pre_norm': True,
                }

config_loss = {'max_iter': 20,
               'tol': 1e-5,
               'max_iter_inner': 1000,
               'alpha_h': 5,
               'alpha_F': 5,
               'alpha_F_fd': 1,
               'alpha_A': 1,
               'FD': True,
               'Hungarian': False,
               'mask_self_loops': True,
               'linear_matching': True,
                }                            

config = config_optim | config_model | config_loss

task = Task(config)
dataset_train = task.get_dataset(config, split = 'train')
dataset_test = task.get_dataset(config, split = 'valid')
model = Constant_Model(task, config)
trainer = Trainer(task, dataset_train, dataset_test, config)
trainer.train(model, save_path = None)
'''