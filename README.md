# PacMan AI

This project utilizes deep Q-networks to 
train an agent to play Pacman. It uses openAI's
gymnasium environments (specifically 
gymnasium[atari]). The final DQN model does not work very well, as for some
reason, it repeats the same move over and over again (upright).

## Things needed to run the project
Make sure your developer tools are updated
if using MacOS. This gave me a headache because 
it wouldn't download necessary packages for
importing the openAI gymnasium environment.
All other packages needed are in the import
statements.


## Files

### env.py
Initializes the ALE/MsPacman-v5 environment 
from openAI's gymnasium environments.

### dqn.py

This project includes a Deep Q-Network (DQN) implementation using PyTorch, 
designed for reinforcement learning tasks. The DQN model consists of a neural
network with two hidden layers, optimized using the Adam optimizer. 
It uses an epsilon-greedy strategy for action selection, balancing 
exploration and exploitation.

Replay Buffer: Stores past experiences to improving learning stability.

Epsilon Decay: Gradually reduces exploration over time to focus on 
exploiting learned strategies.

Experience Replay: Samples minibatches from the replay buffer to update the
model, using mean squared error loss to minimize the difference between 
predicted and target Q-values.

This implementation is suitable for environments with discrete action spaces 
and is designed to learn optimal policies through interaction with the environment.

### train.py

This is a training script that allows the DQN to train itself over many episodes
in the MsPacman gymnasium environment. It takes hyperparameters from config.yaml
and then uses those values to train the DQN. It saves the model to dqn_model.pth
every 10 episodes. Every episode, the reward is stored in an array, and then after
completing the training loop, creates a line graph to track rewards in
reward_plot.png.

### weights.py

This is a simple code that tests whether or not the saved model and an untrained
model have different weights. This is because I thought the model wasnt' being
saved, but it was.

### test.py

This was originally to be tested on the external Pacman game, but since that
didn't work, it just tests the model on the existing environment.

### config.yaml

Stores epsilon-greedy hyperparameters

### pacman-python file

This is a pygame file from pygame's website. I forked this project off of
https://github.com/greyblue9/pacman-python. I wanted to use it for testing my
agent on a game that isn't the environment, but I unfortunately could not figure
how to adjust the game so that the agent could play on it. So that's a scrap.

## Sources Used

I used ChatGPT and Flint for helping me write the code and understand different
concepts

I got the environment from ALE's Atari Gymnasium. https://ale.farama.org/environments/pacman/

## What I would do if I had more time

If I had more time, I would definitely focus on making my model not repeat the
same move over and over again. I would try retraining the model again, 
adjusting different hyperparameters, or adjusting the model itself. For this, I
want to try adjusting the reward system so that it loses points from running
into ghosts. In the defined reward system from the environment, there is no
reward loss for running into ghosts, so I would implement something to make it
start running away from ghosts.