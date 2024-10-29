import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
import torch.nn.functional as F




class NoisyDense(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_mu = nn.Parameter(torch.Tensor(out_features))
        self.w_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_sigma = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters (e.g., Xavier for weights)
        nn.init.xavier_uniform_(self.w_mu)
        nn.init.xavier_uniform_(self.w_sigma)
        nn.init.zeros_(self.b_mu)
        nn.init.zeros_(self.b_sigma)

    def forward(self, inputs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # Ensure it's treated as a batch of 1

        # Ensure that inputs to linear layers are 2D
        inputs = inputs.view(inputs.size(0), -1)

        # Sample noise
        noise_w = self.w_sigma.data.new(self.w_mu.size()).normal_() * 0.5
        noise_b = self.b_sigma.data.new(self.b_mu.size()).normal_() * 0.5

        return F.linear(inputs, self.w_mu + noise_w, self.b_mu + noise_b)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, transition):
        # Get the maximum priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            # Replace oldest element in the buffer
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Calculate probabilities based on priorities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for numerical stability

        # Extract the samples
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        # Update priorities for specific transitions
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def anneal_beta(self, step, final_beta=1.0, beta_annealing_steps=100000):
        # Linearly anneal beta to `final_beta` over `beta_annealing_steps`
        self.beta = min(final_beta, self.beta + (final_beta - self.beta) * (step / beta_annealing_steps))

    def __len__(self):
        return len(self.buffer)


class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, n_step=1, memory_capacity=1000000, gamma=0.99):
        super(DQNCNN, self).__init__()

        # Store parameters
        self.noisy = noisy
        self.n_step = n_step
        self.gamma = gamma
        self.memory = PrioritizedReplayBuffer(capacity=memory_capacity)
        self.n_step_buffer = deque(maxlen=n_step)

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
            NoisyDense(512, num_actions) if noisy else nn.Linear(512, num_actions)
        )

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
            NoisyDense(512, num_actions) if noisy else nn.Linear(512, num_actions)
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.00025)

    def forward(self, x):
        return self.main_network(x)

    def select_action(self, state):
        # Convert state to a tensor
        state_tensor = torch.FloatTensor(state)

        # Check if the state tensor needs a batch dimension
        if state_tensor.dim() == 3:
            state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension, shape becomes (1, 4, 84, 84)

        with torch.no_grad():
            # Forward pass to get Q-values from the noisy network
            q_values = self.forward(state_tensor)

        # Return the action with the highest Q-value
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Store transition in the n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If n-step buffer is full, calculate n-step return and store in memory
        if len(self.n_step_buffer) == self.n_step:
            n_step_return, n_step_next_state, n_step_done = self._get_n_step_info()
            state, action = self.n_step_buffer[0][:2]
            self.memory.push((state, action, n_step_return, n_step_next_state, n_step_done))

    def _get_n_step_info(self):
        """Calculate n-step reward and get the n-step next state and done flag."""
        n_step_return = 0
        for idx in range(self.n_step):
            reward, done = self.n_step_buffer[idx][2], self.n_step_buffer[idx][4]
            n_step_return += (reward * (self.gamma ** idx))
            if done:
                break
        n_step_next_state = self.n_step_buffer[-1][3]
        n_step_done = self.n_step_buffer[-1][4]
        return n_step_return, n_step_next_state, n_step_done

    def replay(self, batch_size, step, beta=0.4):
        if len(self.memory) < batch_size:
            return None

        minibatch, indices, weights = self.memory.sample(batch_size)

        # Extract components from minibatch
        state_batch = torch.FloatTensor(np.array([s[0] for s in minibatch]))
        action_batch = torch.LongTensor(np.array([s[1] for s in minibatch]))
        reward_batch = torch.FloatTensor(np.array([s[2] for s in minibatch]))
        next_state_batch = torch.FloatTensor(np.array([s[3] for s in minibatch]))
        done_batch = torch.FloatTensor(np.array([s[4] for s in minibatch]))
        weights_batch = torch.FloatTensor(weights)

        # Determine the number of channels (frames) in the state batch
        num_channels = state_batch.shape[1]

        # Reshape states assuming they have been stacked correctly
        state_batch = state_batch.view(batch_size, num_channels, 84, 84)
        next_state_batch = next_state_batch.view(batch_size, num_channels, 84, 84)

        # Calculate the current Q-values
        q_values = self.forward(state_batch)
        next_q_values = self.target_network(next_state_batch)

        # Select the Q-values corresponding to the actions taken
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values
        max_next_q_values = next_q_values.max(1)[0]
        target_q_value = reward_batch + (self.gamma * (1 - done_batch) * max_next_q_values)

        # Calculate the loss
        loss = (weights_batch * nn.functional.mse_loss(q_value, target_q_value.detach(), reduction='none')).mean()

        # Perform the optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
        self.optimizer.step()

        # Update priorities in the replay buffer
        priorities = (q_value - target_q_value.detach()).abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)

        return loss

    def update_target_network(self, tau=0.005):
        # Soft update of target network parameters using the polyak averaging formula
        for target_param, param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


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


def preprocess_observation(obs):
    # Print the shape of the input to verify it
    #print("Input shape before conversion:", obs.shape)

    # Check if the input is a single frame with 3 channels
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        # Convert to grayscale
        obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Input observation must be a single RGB frame.")

    # Resize the observation
    obs_resized = cv2.resize(obs_gray, (84, 84))

    # Normalize pixel values
    normalized_observation = obs_resized / 255.0

    # Add channel dimension
    return np.expand_dims(normalized_observation, axis=0)
