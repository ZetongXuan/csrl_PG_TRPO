# Constrained Policy Optimization with Linear Temporal Logic Objectives
The repo builds on CSRL project (in this [article](https://arxiv.org/abs/1909.07299).) of Duke University and apply Policy Gradient and TRPO method to the grid world case study.
## Dependencies
 - [Python](https://www.python.org/): (>=3.5)
 - [Rabinizer 4](https://www7.in.tum.de/~kretinsk/rabinizer4.html): ```ltl2ldba``` must be in ```PATH``` (```ltl2ldra``` is optional)
 - [NumPy](https://numpy.org/): (>=1.15)
 - [scipy](https://scipy.org/)
 - [dill](https://pypi.org/project/dill/)

The examples in this repository also require the following optional libraries for visualization:
 - [Matplotlib](https://matplotlib.org/): (>=3.03)
 - [JupyterLab](https://jupyter.org/): (>=1.0)
 - [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/): (>=7.5)
 - [tqdm](https://github.com/tqdm/tqdm)

## Installation
To install the current release:
```
git clone https://github.com/ZetongXuan/csrl_PG_TRPO.git
cd csrl
pip3 install .
```
## Basic Usage
The package consists of three main classes ```GridMDP```, ```OmegaAutomaton``` and ```ControlSynthesis```. The class ```GridMDP``` constructs a grid-world MDP using the parameters ```shape```, ```structure``` and ```label```. The class ```OmegaAutomaton``` takes an LTL formula ```ltl``` and translates it into an LDBA. The class ```ControlSynthesis``` can then be used to compose a product MDP of the given ```GridMDP``` and ```OmegaAutomaton``` objects and its method ```q_learning``` can be used to learn a control policy for the given objective. For example,
```shell
$ python
```
```python
>>> from csrl.mdp import GridMDP
>>> from csrl.oa import OmegaAutomaton
>>> from csrl import ControlSynthesis
>>> import numpy as np
>>> 
>>> ltl = '(F G a | F G b) & G !c'  # LTL formula
>>> oa = OmegaAutomaton(ltl)  # LDBA
>>> print('LDBA Size (including the trap state):',oa.shape[1])
LDBA Size (including the trap state): 4
>>> 
>>> shape = (5,4)  # Shape of the grid
>>> structure = np.array([  # E:Empty, T:Trap, B:Obstacle
... ['E',  'E',  'E',  'E'],
... ['E',  'E',  'E',  'T'],
... ['B',  'E',  'E',  'E'],
... ['T',  'E',  'T',  'E'],
... ['E',  'E',  'E',  'E']
... ])
>>> label = np.array([  # Labels
... [(),       (),     ('c',),()],
... [(),       (),     ('a',),('b',)],
... [(),       (),     ('c',),()],
... [('b',),   (),     ('a',),()],
... [(),       ('c',), (),    ('c',)]
... ],dtype=np.object)
>>> grid_mdp = GridMDP(shape=shape,structure=structure,label=label)
>>> grid_mdp.plot()
>>> 
>>> csrl = ControlSynthesis(grid_mdp,oa) # Product MDP
>>> 
>>> Q=csrl.q_learning(T=100,K=100000)  # Learn a control policy
>>> value=np.max(Q,axis=4)
>>> policy=np.argmax(Q,axis=4)
>>> policy[0,0]
array([[1, 3, 0, 2],
       [2, 3, 3, 6],
       [0, 3, 0, 2],
       [6, 0, 5, 0],
       [3, 0, 0, 0]])
``` 

## Examples


Attached an example of PG with 2 discount factor 0.99 0.99999 as Jupter Notebook, (https://github.com/ZetongXuan/csrl_PG_TRPO/blob/master/PG_2discount.ipynb)

Attached an example of TRPO with 2 discount factor 0.99 0.99999 as Jupter Notebook, (https://github.com/ZetongXuan/csrl_PG_TRPO/blob/master/TRPO.ipynb)
They both use dill to import a pregererated grid model, which is the same as generated in the above code, 

These 2 examples' python script is also attached, (https://github.com/ZetongXuan/csrl_PG_TRPO/blob/master/PG.py) (https://github.com/ZetongXuan/csrl_PG_TRPO/blob/master/TRPO_2discount.py)
They both build the grid world from csrl package

I also tried some tunning for PG, like changing step size, batch size etc. With Q learning result as banchmark, the test results are shown in (https://github.com/ZetongXuan/csrl_PG_TRPO/blob/master/Test%20results.pdf)

