import numpy as np
import gym
from gym import spaces
from os import path

class FourRoomEnv(gym.Env):
    def __init__(self):
        self.grid_size = 13
        self.np_random = np.random.RandomState()
        self.walls = self._create_walls()
        self.agent_pos = (2, 2)
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.step_count = 0
        
        # define goals
        self.goals = [
            (9, 3),      # top-right room
            (4, 9),     # bottom-left room
            (10, 10)      # bottom-right room
        ]
        self.num_objectives = 3
        
    def _create_walls(self):
        """Define which grid cells are walls."""
        walls = set()
        # Add outer walls (vertical)
        for y in range(self.grid_size):
            walls.add((0, y))
            walls.add((self.grid_size - 1, y))

        # Add outer walls (horizontal)
        for x in range(self.grid_size):
            walls.add((x, 0))
            walls.add((x, self.grid_size - 1))

        # Add inner walls to create four rooms
        for y in range(1, self.grid_size - 1):
            if y == 3 or y == 10:
                continue
            walls.add((self.grid_size // 2, y))

        for x in range(1, 6):
            if x == 3:
                continue
            walls.add((x, 6))
        
        for x in range(6, self.grid_size - 1):
            if x == 9:
                continue
            walls.add((x, 7))
        return walls

    def step(self, action):
        # actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        # used only for rewards, if no reward is reached gym wrapper terminates after max steps set in __init__.py
        done = False
        
        # stochastic transitions
        if self.np_random.rand() < 0.1:
            action = self.np_random.randint(0, 4)

        new_pos = self._get_next_pos(self.agent_pos, action)
        
        # if new position is a wall, stay in place
        if new_pos in self.walls:
            new_pos = self.agent_pos
        
        # update agent position
        self.agent_pos = new_pos
        
        # check for rewards
        reward = np.zeros(self.num_objectives, dtype=np.float32)
        for i, goal in enumerate(self.goals):
            if self.agent_pos == goal:
                reward[i] = 1.0
                done = True
                break

        self.step_count += 1  
        obs = np.array(self.agent_pos, dtype=np.int32)
        
        terminated = done
        truncated = False
        return obs, 0.0, terminated, truncated, {'obj': reward}
    

    def _get_next_pos(self, pos, action):
        """Compute next position given action."""
        x, y = pos
        if action == 0:  # UP
            y = max(0, y - 1)
        elif action == 1:  # DOWN
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # LEFT
            x = max(0, x - 1)
        elif action == 3:  # RIGHT
            x = min(self.grid_size - 1, x + 1)
        return (x, y)

    def reset(self):
        self.agent_pos = (2, 2)
        self.step_count = 0
        self.visited = set()
        return np.array(self.agent_pos, dtype=np.int32), {}

    def render(self, mode='human'):
        """Visualize the environment using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in self.walls:
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=0, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
        
        # Draw goals
        colors = ['red', 'green', 'blue']
        for i, (goal_x, goal_y) in enumerate(self.goals):
            circle = patches.Circle((goal_x, goal_y), 0.3, color=colors[i], alpha=0.6)
            ax.add_patch(circle)
        
        # Draw agent
        agent_x, agent_y = self.agent_pos
        agent_circle = patches.Circle((agent_x, agent_y), 0.2, color='yellow', edgecolor='black', linewidth=2)
        ax.add_patch(agent_circle)
        
        # Set axis properties
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'FourRoom Environment (Step: {self.step_count})')
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            # Convert to numpy array for video recording
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image
        
        plt.close(fig)
