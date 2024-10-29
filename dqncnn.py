import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import random
from collections import deque



class FrameStacker:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = []

    def reset(self, first_frame):
        # Initialize stack with the first frame repeated
        self.frames = [first_frame] * self.num_frames
        return np.concatenate(self.frames, axis=0)  # Concatenate along the channel axis

    def add_frame(self, new_frame):
        if len(self.frames) >= self.num_frames:
            self.frames.pop(0)  # Remove the oldest frame
        self.frames.append(new_frame)
        return np.concatenate(self.frames, axis=0)  # Concatenate along the channel axis


def preprocess_observation(observation):
    # Convert the observation to grayscale
    gray_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

    # Resize the observation to 84x84
    resized_observation = cv2.resize(gray_observation, (84, 84))

    # Normalize the pixel values to [0, 1]
    normalized_observation = resized_observation / 255.0

    # Add a channel dimension (1, 84, 84) to match the expected input shape
    return np.expand_dims(normalized_observation, axis=0)

# Define a CNN-based DQN model
class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def remember(self, state, action, reward, next_state, done):
        # Ensure state and next_state are preprocessed and have the correct shape
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        # Convert state to a tensor
        state_tensor = torch.FloatTensor(state)

        # Check if the state tensor needs a batch dimension
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension, shape becomes (1, 4, 84, 84)

        # Ensure state_tensor is defined before using it
        if random.random() < self.epsilon:
            return random.randrange(self.fc2.out_features)  # Random action
        else:
            with torch.no_grad():
                q_values = self.forward(state_tensor)  # Forward pass
            return torch.argmax(q_values).item()  # Return the action with the highest Q-value

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Ensure that the state_batch has the correct shape
        state_batch = torch.FloatTensor(np.array([s[0] for s in minibatch]))
        action_batch = torch.LongTensor(np.array([s[1] for s in minibatch]))
        reward_batch = torch.FloatTensor(np.array([s[2] for s in minibatch]))
        next_state_batch = torch.FloatTensor(np.array([s[3] for s in minibatch]))
        done_batch = torch.FloatTensor(np.array([s[4] for s in minibatch]))


        # Ensure the state batch has the shape [batch_size, num_frames, height, width]
        # Assuming num_frames = 4 (for frame stacking)
        state_batch = state_batch.view(batch_size, 4, 84, 84)
        next_state_batch = next_state_batch.view(batch_size, 4, 84, 84)

        with torch.no_grad():
            next_q_values = self.forward(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (0.99 * next_q_values * (1 - done_batch))

        current_q_values = self.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Example usage in training loop
def train_dqn(env, num_episodes, batch_size):
    input_shape = (1, 84, 84)  # Grayscale image with one channel
    num_actions = env.action_space.n
    dqn = DQNCNN(input_shape, num_actions)

    for episode in range(num_episodes):
        observation = env.reset()
        observation = preprocess_observation(observation)
        done = False
        total_reward = 0

        while not done:
            action = dqn.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            next_observation = preprocess_observation(next_observation)
            dqn.remember(observation, action, reward, next_observation, done)
            dqn.replay(batch_size)
            observation = next_observation
            total_reward += reward

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
