# CLVR Implementation Project - Starter Code


## Requirements

- python >= 3.6
- pytorch >= 1.3

## Installation Instructions

Create a virtual environment and install all required packages.
```
cd clvr_RewIR_starter
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

## Sprites Dataset

We provide a dataset that generates sequences of multiple geometric objects that bounce in a 2D frame _on the fly_. An example PyTorch
dataset can be found at the top of [`sprites_datagen/moving_sprites.py`](sprites_datagen/moving_sprites.py). You can wrap it 
in a PyTorch [Dataloader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset)
to allow for efficient data "loading" when training your model.

At the bottom of the file you can find an example `spec` that needs to be passed to the Dataset class, in which you can define
image resolution, number of shapes in the scene etc. It also allows you to define which rewards your dataset should include
(e.g. rewards proportional to x / y position etc). All reward functions are defined in [`sprites_datagen/rewards.py`](sprites_datagen/rewards.py)
and you can define more if you want.


## Sprites Environment

Renders trajectories the same way as the dataset, but allows to control the movement of the agent by passing in a 2D array of 
x,y-velocities. Implements the standard `gym` interface. We provide image and state-based versions of the environment.
All environments are registered via [`sprites_env/__init__.py`](sprites_env/__init__.py), including image and state-based
environment with different number of distractors.

You can create an instance of a particular environment by running:
```
import gym
env = gym.make('Sprites-v1')
```
This will create an image-based environment with one distractor. Change the name to get different environments.
