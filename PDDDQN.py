import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import random
from collections import deque


class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 0.0
        self.count = 1.0  # Start with 1 to avoid division by zero initially
        self.epsilon = epsilon  # Small value to prevent division by zero

    def update(self, reward):
        # Incremental update of mean and variance
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.var += delta * (reward - self.mean)

    def normalize(self, reward):
        # Ensure variance is not negative due to floating point issues
        std = np.sqrt(max(self.var / self.count, 0.0) + self.epsilon)
        return (reward - self.mean) / (std + self.epsilon)

    def reset(self):
        # Reset mean, variance, and count to initial values
        self.mean = 0.0
        self.var = 0.0
        self.count = 1.0




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

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, transition):
        max_priority = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class DuelingDQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions, memory_capacity=1000000, gamma=0.99):
        super(DuelingDQNCNN, self).__init__()

        # Store parameters
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = PrioritizedReplayBuffer(capacity=memory_capacity)

        # Define the main network
        self.main_network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),  # Ensure output is flattened here
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # Value and Advantage streams for dueling architecture
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

        # Initialize the target network similarly to the main network
        self.target_network = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # Value and Advantage streams for target network
        self.target_value_stream = nn.Linear(512, 1)
        self.target_advantage_stream = nn.Linear(512, num_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.00025)

    def forward(self, x):
        x = self.main_network(x)

        # Compute Value and Advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get final Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension

        if random.random() < self.epsilon:
            return random.randrange(self.advantage_stream.out_features)  # Random action
        else:
            with torch.no_grad():
                q_values = self.forward(state_tensor)
            return torch.argmax(q_values).item()

    def update_target_network(self, tau=1.0):
        for target_param, param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def replay(self, batch_size, beta=0.4):
        if len(self.memory) < batch_size:
            return None

        minibatch, indices, weights = self.memory.sample(batch_size, beta)

        state_batch = torch.FloatTensor(np.array([s[0] for s in minibatch]))
        action_batch = torch.LongTensor(np.array([s[1] for s in minibatch]))
        reward_batch = torch.FloatTensor(np.array([s[2] for s in minibatch]))
        next_state_batch = torch.FloatTensor(np.array([s[3] for s in minibatch]))
        done_batch = torch.FloatTensor(np.array([s[4] for s in minibatch]))
        weights_batch = torch.FloatTensor(weights)

        state_batch = state_batch.view(batch_size, 4, 84, 84)
        next_state_batch = next_state_batch.view(batch_size, 4, 84, 84)

        with torch.no_grad():
            # Double DQN: Use main network to select the best action, and target network to calculate its value
            next_action = self.forward(next_state_batch).argmax(1)
            next_q_values = self.target_network(next_state_batch).gather(1, next_action.unsqueeze(1)).squeeze()
            target_q_values = reward_batch + (0.99 * next_q_values * (1 - done_batch))

        current_q_values = self.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        loss = (weights_batch * (current_q_values - target_q_values.detach()).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
        self.optimizer.step()

        # Update priorities in the replay buffer
        priorities = (current_q_values - target_q_values.detach()).abs() + 1e-6
        self.memory.update_priorities(indices, priorities.detach().cpu().numpy())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss
