

### ------------------------ LOAD CONFIG ------------------------ ###
import json
import os

run_name = 'ColoringMedium_testrun'
save_path = f'./runs/{run_name}/'


with open(os.path.join(save_path,'args.txt')) as f:
    config = json.load(f)
    
    
### ------------------------ LOAD MODEL/TASK/DATASET ------------------------ ###
    
from Any2Graph import Any2Graph_Model
from Img2Graph import Img2Graph
import torch

task = Img2Graph(config)
dataset = task.get_dataset(config, split = 'test')
model = Any2Graph_Model(task, config)
model.load_state_dict(torch.load(os.path.join(save_path,'best_model')))


### ------------------------ EVAL ------------------------ ###

from Any2Graph import Evaluator

evalutor = Evaluator(task, dataset, config)

evalutor.eval(model)

