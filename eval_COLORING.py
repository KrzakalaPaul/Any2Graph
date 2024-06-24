### ------------------------ LOAD CONFIG ------------------------ ###
import json
import os

run_name = 'ColoringSmall_testrun'
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
from time import perf_counter
import os

evaluator = Evaluator(task, dataset, config)

tic = perf_counter()
metrics = evaluator.eval(model,n_samples=10)
tac = perf_counter()

metrics.to_csv(save_path+'/metrics.csv')

metrics_avg = metrics.mean()
metrics_avg['eval_time (min)'] = (tac-tic)/60
metrics_avg.to_csv(save_path+'/metrics_avg.csv')


evaluator.plot_prediction(model,save_path,n_samples=10)

