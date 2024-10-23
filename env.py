import gymnasium as gym

def create_mspacman_env(render_mode="rgb_array"):
    # Create the Ms. Pacman environment directly without registration
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    return env

