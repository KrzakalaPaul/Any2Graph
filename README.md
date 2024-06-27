# Welcome to Any2Graph! 

## What is Any2Graph ?

When you want to train a model for a task where the **input** is a graph, you can use a Graph Neural Network. But what about tasks where the **output** is a graph? This is where _Any2Graph_ comes into play!

<p align="middle">
  <img src="/fig/GNNvsAny2Graph.png" width="600" />
</p>

Any2Graph is a framework composed of 1) a continuous graph representation, 2) an architecture, and 3) a loss (PMFGW). See [the paper](https://arxiv.org/pdf/2402.12269) for more details.

<p align="middle">
  <img src="/fig/Any2Graph_Pipeline.png" width="500" />
</p>

## Tasks

*Any2Graph* is a flexible framework that can be used to tackle a variety of **tasks**.  Different tasks have different modalities for instance different types of inputs  (images, text, graphs ...) or output graphs with different properties (no nodes features, discrete node features, continuous node features ...)

The first step to use *Any2Graph* is always to choose a task. In practice, this means subclassing the base Task class defined in 
```
Any2Graph/
└── base_task_class.py
```
We already provide 3 classes of tasks: *Img2Graph*, *Sat2Graph* and *Fingerprint2Graph*. Feel free to create new ones! 

Your new task will probably be organized like this:
```
Any2Graph/
└── base_task_class.py
└── MyTask/
	└── __init__.py
	└── my_task.py <- Subclass Task here, remember to overwrite all methods
	└── my_encoder.py <- Define your encoder here for the "get_encoder" method
	└── Dataset1/ <- You can define one or several datasets for the "get_dataset" method
	└── Dataset2/ 
```

## Download Data 

### Coloring

_Coloring_ is a synthetic dataset that we introduce for benchmarking Any2Graph. See the illustration below (first row = input image, second row = target graph).

<p align="middle">
  <img src="/fig/coloring_inputs.png" width="700" />
</p>

To train a model on Coloring you need to add data in _Any2Graph/Img2Graph/Coloring/Data_ as below: 

```
Any2Graph/
└── Img2Graph
	└── Coloring
		└── Data
			└── small
			└── medium

```

The small subset is composed of 100 000 Graph of size 5 to 10. The medium subset is composed of 200 000 Graph of size 10 to 15. They can be downloaded here: [small.zip](https://drive.google.com/file/d/1NcTNsvydlWO9I9BEjGm3ybIA5Wnpq4ga/view?usp=sharing), [medium.zip](https://drive.google.com/file/d/1NcTNsvydlWO9I9BEjGm3ybIA5Wnpq4ga/view?usp=sharing).

You can also create your own Coloring Subset by running Coloring_Generate_Data.py and changing the parameters of ColoringSampler. 


### Toulouse

### USCities

Download data from [20 US Cities dataset](https://github.com/songtaohe/Sat2Graph/tree/master/prepare_dataset). It should be organized as follows

```
Any2Graph/
└── Sat2Graph
	└── USCities
		└── data/
		    └── 20cities/
```

Then execute prepare_USCities.py (located in Any2Graph/Sat2Graph/USCities).


### QM9

Download the smiles string of the molecules in QM9 from [QM9_smiles](https://github.com/songtaohe/Sat2Graph/tree/master/prepare_dataset). It should be organized as follows

```
Any2Graph/
└── Fingerprint2Graph
	└── QM9
		└── data/
		    └── QM9_smiles.csv
```
Then execute split_QM9.py (located in Any2Graph/Fingerprint2Graph/QM9) to split between train/test/validation.

### GDB13

Download the smiles string of the molecules in GDB13 from [GDB13_smiles]([https://github.com/songtaohe/Sat2Graph/tree/master/prepare_dataset](https://zenodo.org/record/5172018/files/GDB13_Subset-ABCDEFGH.smi.gz?download=1)). It should be organized as follows

```
Any2Graph/
└── Fingerprint2Graph
	└── GDB13
		└── data/
		    └── GDB13_Subset-ABCDEFGH.smi
```
Then execute split_GDB13.py (located in Any2Graph/Fingerprint2Graph/GDB13) to split between train/test/validation.

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


