# PacMan AI

This project utilizes deep Q-networks to 
train an agent to play Pacman. It uses openAI's
gymnasium environments (specifically 
gymnasium[atari]). It uses CNNs to process observations, which are images
that are grayscaled and scaled to 84x84. Then I stack 4 images so that the
DQN can understand direction. Then it trains by using greedy-epsilon,
and it has a  dueling architecture that helps the network separately assess 
the value of a state and the advantage of each action, which helps the model 
learn more efficiently. I also included my own reward function
that results in a -100 reward for dying to a ghost, and I normalize
the rewards to fix abnormally high reward values.

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
reward_plot.png. This was later adjusted for the NNnSPDDDQN.py instead.

### train2.py
Same as train.py but for the PDDDQN.py.

### weights.py

This is a simple code that tests whether or not the saved model and an untrained
model have different weights. This is because I thought the model wasnt' being
saved, but it was.

### test.py

This was originally to be tested on the external Pacman game, but since that
didn't work, it just tests the model on the existing environment. Right now,
it doesn't really work because I changed the architecture of the model
greatly, and since I don't have enough storage space, the model can't
run for enough episodes to save.

### dqncnn.py
This was my first attempt at trying to stack frames. I did not
spend much time on this, and I moved on to other models that
incorporated frame stacking but also other things.

### PDDDQN.py
This was trying to use the website listed below's idea of
using prioritized double dueling. In a nutshell, it makes the agent
learn faster and better from statistically significant events, or
events that make the Q-value abnormally higher or lower. So if 
something happens in training that is significant, it will learn
better from it. The double DQN has two networks, as opposed to a
normal DQN's one. A normal DQN has one network Q-network to 
select actions and evaluate them. A double DQN has a main network
that selects the best action in the next state and a target 
network that evaluates the value of that action. By decoupling
action selection and evaluation, Double DQN reduces overestimation
and it has a more stable and accurate learning process.

### NNnSPDDDQN.py
This uses similar principles as PDDDQN.py, but introduces noisy
networks and n-step returns. The noisy networks basically aim
to get rid of the epsilon-greedy policy. While epsilon-greedy is
good for simpler environments, I wanted my Pacman agent to be able
to decide when and how to explore or exploit, which epsilon-greedy
can't do. It's useful for Pacman, which may have different results
for the same action. It promotes a more structured form of exploration, 
making it easier for the agent to discover useful strategies. It also 
helps avoid overly repetitive actions, improving the model's 
ability to adapt to different scenarios. N-step returns allow the 
model to evaluate rewards over multiple steps, not just one, to 
evaluate over multiple steps, which is helpful in Pacman.


### config.yaml

Stores hyperparameters


## Sources Used

I used ChatGPT and Flint for helping me write the code and understand different
concepts

I got the environment from ALE's Atari Gymnasium. https://ale.farama.org/environments/pacman/

I also used this blog post: https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814
for insights into making my project. It gave my the ideas for using CNNs,
processing stacks of 3 or 4 inputs instead of one to give the model a sense
of direction, using prioritized double dueling DQN, and the Noisy Networks n-Step
Prioritized Double Dueling DQN. They also provided a Github repo, and I used
it to see if I could gain anything there: https://github.com/jakegrigsby/AdvancedPacmanDQNs.

## What I would do if I had more time

If I had more time, I would focus on fine-tuning my model even more. Making
this AI is a painstaking process, and it's far from perfect. I could also
benefit from training it over way more episodes-- somewhere around the 10k
range would be ideal. However, my computer is not strong enough to handle
it. Maybe if I tried using a different input than images, the memory wouldn't
reach it's maximum