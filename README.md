# Welcome to Any2Graph! 

## What is Any2Graph ?

When you want to train a model for a task where the **input** is graph, you can use a Graph Neural Network. But what about tasks where the **output** is a graph? This is where _Any2Graph_ comes into play!

<p align="middle">
  <img src="/fig/GNNvsAny2Graph.png" width="600" />
</p>

Any2Graph is a framework composed of 1) a continuous graph representation, 2) an architecture, and 3) a loss (PMFGW). See [the paper](https://arxiv.org/pdf/2402.12269) for more details.

<p align="middle">
  <img src="/fig/Any2Graph_Pipeline.png" width="500" />
</p>

## Tasks

*Any2Graph* is a flexible framework that can be used to tackle a variety of **tasks**.  Different tasks have different modalities for instance different types of inputs  (images, text, graphs ...) or output graphs with different properties (no nodes features, discrete node features, continuous node features ...)

The first step to use Any2Graph is always to choose a task. In practice, this means subclassing the base Task class defined in 
```
Any2Graph/
└── base_task_class.py
```
We already provide 3 classes of tasks: Img2Graph, Sat2Graph and Fingerprint2Graph. Feel free to create new ones! 

Your new task will probably be organised like this:
```
Any2Graph/
└── base_task_class.py
└── MyTask/
	└── __init__.py
	└── my_task.py <- Subclass Task here, remember to overwrite all methods
	└── my_encoder.py <- Define your encoder here for the "get_encoder" method
	└── Dataset1/ <- You can define one or several datasets for the "get_dataset" method
	└── Dataset2/ <- 
```

## Download Data 

### Coloring

### Toulouse

### USCities

### QM9

### GDB13


## Training/Evaluating a model 

To train a model, you first need to create a config yaml file with the parameters of the training.
```
configs/
└── my_config.yaml
```
Then you can launch the training with the following command 
```
python train.py --config my_config --run_name a_run
```
To eval a run/model you can run the following command 
```
python eval.py --run_name a_run --n_samples_eval 1000 --n_samples_plot 10
```
Where *n_samples_eval* is the number of samples used to eval the model and  *n_samples_plot*  is the number of samples for which an input/target/prediction figure is plotted. The results will be available in 

```
runs/
└── a_run/
```


