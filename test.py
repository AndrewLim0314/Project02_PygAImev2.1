import gymnasium as gym
import torch
from NNnSPDDDQN import DQNCNN
from ale_py import ALEInterface
from NNnSPDDDQN import preprocess_observation

# Load the saved model
def load_model():
    input_dim = (1,84,84)  # Adjust based on your environment
    output_dim = 9  # Number of possible actions in Ms. Pacman

    dqn = DQNCNN(input_dim, output_dim)
    model_save_path = '/Users/25lim/ADV ML AI/Project02_PygAImev2.1/Models/ddqn_model.pth'
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

    # Reset the environment and preprocess the first state
    observation, info = env.reset()
    state = preprocess_observation(observation)

    done = False
    total_reward = 0

    while not done:
        # Convert state to a 4D tensor: (batch_size, channels, height, width)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Use the trained DQN to select an action
        with torch.no_grad():
            q_values = dqn(state_tensor)
        action = torch.argmax(q_values).item()

        # Step in the environment using the selected action
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess_observation(next_observation)

        # Update the state and accumulate reward
        state = next_state
        total_reward += reward
        done = terminated or truncated

    print(f"Total Reward: {total_reward}")
    env.close()



if __name__ == "__main__":
    test_dqn()
