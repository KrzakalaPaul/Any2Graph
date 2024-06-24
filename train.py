
### ------------------------ LOAD CONFIG ------------------------ ###

import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config',type=str)
parser.add_argument('--run_name',default=None)
args = parser.parse_args()

config_name = args.config

with open(os.path.join('configs/',config_name+'.yaml')) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
config['run_name'] = args.run_name
    
### ------------------------ LOAD TASK ------------------------ ###

if config['task'] == 'Img2Graph':
    from Img2Graph import Img2Graph as Task
    
elif config['task'] == 'Sat2Graph':
    from Sat2Graph import Sat2Graph as Task

else:
    raise ValueError('Task not recognized')

task = Task(config)

### ------------------------ LOAD DATASET ------------------------ ###

from Any2Graph import Trainer, Any2Graph_Model

dataset_train = task.get_dataset(config, split = 'train')
print(f'Train set size: {len(dataset_train)}')
dataset_test = task.get_dataset(config, split = 'valid')
print(f'Test set size: {len(dataset_test)}')

### ------------------------ INIT MODEL ------------------------ ###

model = Any2Graph_Model(task, config)

### ------------------------ TRAIN ------------------------ ###

trainer = Trainer(task, dataset_train, dataset_test, config)
save_path = None if config['run_name'] == None else f'./runs/{config["run_name"]}/'
trainer.train(model, save_path = save_path)

