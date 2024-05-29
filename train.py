from Img2Graph import Img2Graph
from Img2Graph.Datasets import Coloring
from Any2Graph.trainer import Trainer


config = {'device':'cuda', 'batchsize': 128, 'max_grad_step': 100000, 'max_grad_norm': 1, 'n_eval_interval': 1000}

task = Img2Graph(config)
model = task.get_model(config)
dataset_train = Coloring(config, split = 'train')
dataset_test = Coloring(config, split = 'valid')
trainer = Trainer(task, dataset_train, dataset_test, config)
trainer.train(model, save_path = None)