from Img2Graph import TaskA
from Img2Graph.Datasets import DatasetA
from Any2Graph.trainer import Trainer


config = None
task = TaskA(config)
model = task.get_model(config)
dataset = DatasetA(config)

trainer = Trainer(task, dataset, config)
trainer.train(model)