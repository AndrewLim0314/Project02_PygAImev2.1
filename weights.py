import torch
from dqn import DQN

# Define the input and output dimensions (use the same dimensions as when training)
input_dim = 210 * 160 * 3  # Example input dimension, adjust as needed
output_dim = 9  # Example output dimension, adjust as needed

# Initialize the untrained DQN model
untrained_dqn = DQN(input_dim=input_dim, output_dim=output_dim)

# Load the saved DQN model
model_save_path = '/Users/25lim/ADV ML AI/Project02_PygAImev2.1/dqn_model.pth'
saved_dqn = DQN(input_dim=input_dim, output_dim=output_dim)

# Load the saved weights into the model
saved_dqn.load_state_dict(torch.load(model_save_path))

# Function to compare weights of two models
def compare_weights(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(param1, param2):
            return False  # Weights are different
    return True  # Weights are the same

# Compare the untrained model with the saved model
weights_are_same = compare_weights(untrained_dqn, saved_dqn)

if weights_are_same:
    print("The weights of the untrained model and the saved model are the same.")
else:
    print("The weights of the untrained model and the saved model are different.")
