import gymnasium as gym
import cv2


def preprocess_and_identify_colors(observation):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)

    # Define HSV color ranges for Pacman and ghosts
    # These ranges need to be determined based on the actual colors in the game
    pacman_color_range = ((20, 100, 100), (30, 255, 255))  # Example range for yellow
    ghost_color_range = ((110, 50, 50), (130, 255, 255))  # Example range for blue

    # Create masks for Pacman and ghosts
    pacman_mask = cv2.inRange(hsv_image, pacman_color_range[0], pacman_color_range[1])
    ghost_mask = cv2.inRange(hsv_image, ghost_color_range[0], ghost_color_range[1])

    # Find contours for Pacman
    pacman_contours, _ = cv2.findContours(pacman_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in pacman_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(observation, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around Pacman

    # Find contours for ghosts
    ghost_contours, _ = cv2.findContours(ghost_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in ghost_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(observation, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around ghosts

    return observation, pacman_contours, ghost_contours


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, collision_penalty=-10):
        super(CustomRewardWrapper, self).__init__(env)
        self.collision_penalty = collision_penalty

    def preprocess_and_identify_colors(self, observation):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)

        # Define HSV color ranges for Pacman and ghosts
        pacman_color_range = ((20, 100, 100), (30, 255, 255))  # Example range for yellow
        ghost_color_range = ((110, 50, 50), (130, 255, 255))  # Example range for blue

        # Create masks for Pacman and ghosts
        pacman_mask = cv2.inRange(hsv_image, pacman_color_range[0], pacman_color_range[1])
        ghost_mask = cv2.inRange(hsv_image, ghost_color_range[0], ghost_color_range[1])

        # Find contours for Pacman and ghosts
        pacman_contours, _ = cv2.findContours(pacman_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ghost_contours, _ = cv2.findContours(ghost_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return pacman_contours, ghost_contours

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        # Preprocess the observation to get contours
        pacman_contours, ghost_contours = self.preprocess_and_identify_colors(observation)

        # Check for collisions between Pacman and ghosts
        for pacman_contour in pacman_contours:
            for ghost_contour in ghost_contours:
                if self.is_collision(pacman_contour, ghost_contour):
                    reward += self.collision_penalty  # Apply negative reward for collision

        return observation, reward, done, info

    def is_collision(self, pacman_contour, ghost_contour):
        # Check if the bounding rectangles of Pacman and ghost overlap
        pacman_rect = cv2.boundingRect(pacman_contour)
        ghost_rect = cv2.boundingRect(ghost_contour)
        overlap = self.rect_overlap(pacman_rect, ghost_rect)
        return overlap

    def rect_overlap(self, rect1, rect2):
        # Check if two rectangles overlap
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2)


def create_mspacman_env(render_mode="rgb_array"):
    # Create the Ms. Pacman environment directly without registration
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    env = CustomRewardWrapper(env)
    return env


