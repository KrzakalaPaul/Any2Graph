from Any2Graph import Trainer, Any2Graph_Model
from Sat2Graph import Sat2Graph


### ------------------------ LOAD CONFIG ------------------------ ###

config_run = {'dataset': 'TOULOUSE',
               'dataset_size': -1,
               'wandb': True,
               'run_name': 'TOULOUSE_testrun',
               'augment_data': True,
               }

config_optim = {'device':'cuda', 
                'batchsize': 128, 
                'max_grad_step': 150000, 
                'max_grad_norm': 0.1, 
                'n_eval_interval': 1000,
                'lr':3*1e-4,
                'warmup': 8000,
                }

config_model = {'Mmax': 11,
                'model_dim': 256,
                'dropout': 0.,
                'node_feature_dim': 2,
                'MLP_h_layers': 1,
                'MLP_F_layers': 1,
                'MLP_F_fd_layers': 1,
                'MLP_A_layers': 3,
                'virtual_node': True,
                'transformer_layers': 4,
                'transformer_heads': 8,
                'transformer_dropout': 0.,
                'pre_norm': True,
                }

config_loss = {'max_iter': 20,
               'tol': 1e-5,
               'max_iter_inner': 2000,
               'alpha_h': 1,
               'alpha_F': 5,
               'alpha_F_fd': 0,
               'alpha_A': 1,
               'FD': False,
               'Hungarian': False,
               'mask_self_loops': False,
               'linear_matching': False,
                }                            

config = config_run | config_optim | config_model | config_loss

### ------------------------ LOAD TASK ------------------------ ###

task = Sat2Graph(config)

### ------------------------ LOAD DATASET ------------------------ ###

dataset_train = task.get_dataset(config, split = 'train')
print(f'Train set size: {len(dataset_train)}')
dataset_test = task.get_dataset(config, split = 'valid')
print(f'Test set size: {len(dataset_test)}')

### ------------------------ INIT MODEL ------------------------ ###

model = Any2Graph_Model(task, config)

### ------------------------ TRAIN ------------------------ ###

trainer = Trainer(task, dataset_train, dataset_test, config)
trainer.train(model)

