
### ------------------------ LOAD CONFIG ------------------------ ###

import argparse
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--run_name',type=str)
parser.add_argument('--n_samples_eval',type=int, default=0)
parser.add_argument('--n_samples_plot',type=int, default=0)
args = parser.parse_args()

run_name = args.run_name
save_path = f'./runs/{run_name}/'

with open(os.path.join(save_path,'args.yaml')) as f:
    config_train = yaml.load(f, Loader=yaml.FullLoader)

config = config_train
config['augment_data'] = False

    
### ------------------------ LOAD TASK ------------------------ ###

if config['task'] == 'Img2Graph':
    from Any2Graph.Img2Graph import Img2Graph as Task
    
elif config['task'] == 'Sat2Graph':
    from Any2Graph.Sat2Graph import Sat2Graph as Task

else:
    raise ValueError('Task not recognized')

task = Task(config)
    
### ------------------------ LOAD MODEL/DATASET ------------------------ ###
    
from Any2Graph import Any2Graph_Model
import torch

dataset = task.get_dataset(config, split = 'test')
model = Any2Graph_Model(task, config)
model.load_state_dict(torch.load(os.path.join(save_path,'best_model')))


### ------------------------ EVAL ------------------------ ###

from Any2Graph import Evaluator
from time import perf_counter
import os

evaluator = Evaluator(task, dataset, config)

if args.n_samples_eval>0:
    
    tic = perf_counter()
    metrics = evaluator.eval(model,n_samples=args.n_samples_eval)
    tac = perf_counter()

    metrics.to_csv(save_path+'/metrics.csv')

    metrics_avg = metrics.mean()
    metrics_avg['eval_time (min)'] = (tac-tic)/60
    metrics_avg.to_csv(save_path+'/metrics_avg.csv')

if args.n_samples_plot>0:
    evaluator.plot_prediction(model,save_path,n_samples=args.n_samples_plot)
