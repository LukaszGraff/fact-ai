import numpy as np
import gym
from gym import spaces


class NineRoomEnv(gym.Env):
    """
    3x3-room gridworld (19x19) with 8 goal rooms and a central start room.
    Layout:
    - Outer boundary walls.
    - Two interior vertical walls (x=6, x=12) and two interior horizontal walls (y=6, y=12).
    - Each interior wall has doorways aligned with the centers of rooms:
        door y-positions: 3, 9, 15 on vertical walls
        door x-positions: 3, 9, 15 on horizontal walls
    - Agent starts at the center room center (9, 9).
    - Goals are at the centers of the 8 non-central rooms:
        (3,3), (9,3), (15,3), (3,9), (15,9), (3,15), (9,15), (15,15)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.grid_size = 19
        self.np_random = np.random.RandomState()

        self.walls = self._create_walls()

        # Start in the center room
        self.start_pos = (9, 9)
        self.agent_pos = self.start_pos

        # up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32
        )

        self.step_count = 0

        # 8 goals (all rooms except center)
        self.goals = [
            (3, 3),   # top-left
            (9, 3),   # top-middle
            (15, 3),  # top-right
            (3, 9),   # middle-left
            (15, 9),  # middle-right
            (3, 15),  # bottom-left
            (9, 15),  # bottom-middle
            (15, 15), # bottom-right
        ]
        self.num_objectives = len(self.goals)

    def _create_walls(self):
        """Define which grid cells are walls for a 3x3 room layout."""
        walls = set()
        gs = self.grid_size

        # Outer walls
        for y in range(gs):
            walls.add((0, y))
            walls.add((gs - 1, y))
        for x in range(gs):
            walls.add((x, 0))
            walls.add((x, gs - 1))

        # Interior walls separating rooms: x=6,12 and y=6,12
        vertical_walls_x = [6, 12]
        horizontal_walls_y = [6, 12]

        # Doorway coordinates aligned with room centers
        doorway_ys = [3, 9, 15]  # openings in vertical walls
        doorway_xs = [3, 9, 15]  # openings in horizontal walls

        # Add vertical interior walls with doorways
        for wx in vertical_walls_x:
            for y in range(1, gs - 1):
                if y in doorway_ys:
                    continue
                walls.add((wx, y))

        # Add horizontal interior walls with doorways
        for wy in horizontal_walls_y:
            for x in range(1, gs - 1):
                if x in doorway_xs:
                    continue
                walls.add((x, wy))

        return walls

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

    def step(self, action):
        # actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        done = False

        # stochastic transitions (match FourRoom design choice)
        if self.np_random.rand() < 0.1:
            action = self.np_random.randint(0, 4)

        new_pos = self._get_next_pos(self.agent_pos, action)

        # wall collision => stay
        if new_pos in self.walls:
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # multi-objective reward vector in info['obj']
        reward_vec = np.zeros(self.num_objectives, dtype=np.float32)
        for i, goal in enumerate(self.goals):
            if self.agent_pos == goal:
                reward_vec[i] = 1.0
                done = True
                break

        self.step_count += 1
        obs = np.array(self.agent_pos, dtype=np.int32)

        # Scalar reward is 0.0; vector reward is in info (match FourRoom design choice)
        return obs, 0.0, done, {"obj": reward_vec}

    def reset(self):
        self.agent_pos = self.start_pos
        self.step_count = 0
        self.visited = set()
        return np.array(self.agent_pos, dtype=np.int32)

    def render(self, mode="human"):
        """Visualize the environment using matplotlib (match FourRoom design choice)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Draw walls
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in self.walls:
                    rect = patches.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        linewidth=0,
                        edgecolor="none",
                        facecolor="black",
                    )
                    ax.add_patch(rect)

        # Draw goals (cycling colors)
        colors = ["red", "green", "blue", "purple", "orange", "cyan", "magenta", "brown"]
        for i, (gx, gy) in enumerate(self.goals):
            circle = patches.Circle((gx, gy), 0.3, color=colors[i % len(colors)], alpha=0.6)
            ax.add_patch(circle)

        # Draw agent
        ax_x, ax_y = self.agent_pos
        agent_circle = patches.Circle((ax_x, ax_y), 0.2, color="yellow", edgecolor="black", linewidth=2)
        ax.add_patch(agent_circle)

        # Axis properties
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"NineRoom Environment (Step: {self.step_count})")

        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image

        plt.close(fig)
