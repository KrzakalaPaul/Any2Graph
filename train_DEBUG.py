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
