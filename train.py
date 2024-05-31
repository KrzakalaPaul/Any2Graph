from Any2Graph import base_task_class
from Any2Graph.trainer import Trainer
from Any2Graph.model import Any2Graph_Model


config_optim = {'device':'cuda', 
                'batchsize': 128, 
                'max_grad_step': 100000, 
                'max_grad_norm': 1, 
                'n_eval_interval': 1000,
                'lr':3*1e-4,
                'warmup': 8000,
                }

config_model = {'Mmax': 10,
                'model_dim': 128,
                'dropout': 0.1,
                'node_feature_dim': 1,
                'MLP_layers': 2,
                'virtual_node': True,
                'FD': True,
                'transformer_layers': 3,
                'transformer_heads': 4,
                'transformer_dropout': 0.1,
                'pre_norm': True,
                }

config = config_optim | config_model

task = base_task_class(config)
dataset_train = task.get_dataset(name='Coloring', split = 'train')
dataset_test = task.get_dataset(name='Coloring', split = 'valid')
model = Any2Graph_Model(task, config)
trainer = Trainer(task, dataset_train, dataset_test, config)
trainer.train(model, save_path = None)