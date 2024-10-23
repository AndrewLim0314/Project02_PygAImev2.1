import gymnasium as gym
import torch
import numpy as np
from dqn import DQN
from ale_py import ALEInterface

# Load the saved model
def load_model():
    input_dim = 210 * 160 * 3  # Adjust based on your environment
    output_dim = 9  # Number of possible actions in Ms. Pacman

    dqn = DQN(input_dim=input_dim, output_dim=output_dim)
    model_save_path = '/Users/25lim/ADV ML AI/Project02_PygAImev2.1/dqn_model.pth'
    dqn.epsilon = 0.01  # Just the epsilon final so it doesn't explore, only exploit

    # Load the saved weights
    dqn.load_state_dict(torch.load(model_save_path, weights_only=True))
    dqn.eval()  # Set the model to evaluation mode (no training)

    return dqn


def test_dqn():
    # Initialize the Arcade Learning Environment
    ale = ALEInterface()
    gym.register_envs(ale)
    env = gym.make('ALE/MsPacman-v5', render_mode="human")
    dqn = load_model()

    # Reset the environment
    state, info = env.reset()
    state = state.flatten().astype(np.float32)

    done = False
    total_reward = 0

    while not done:
        # Convert state to a tensor and select action using the trained DQN
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        action = torch.argmax(q_values).item()

        # Step in the environment using the selected action
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state.flatten().astype(np.float32)

        # Update state and accumulate reward
        state = next_state
        total_reward += reward
        done = terminated or truncated

    print(f"Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    test_dqn()
