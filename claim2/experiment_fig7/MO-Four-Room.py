import numpy as np
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
from gym import spaces
from os import path

class FourRoomEnv(gym.Env):
    def __init__(self):
        self.grid_size = 13
        self.np_random = np.random.RandomState()
        self.agent_start = (2, 2)
        self.walls = self._create_walls()
        self.agent_pos = self.agent_start
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.step_count = 0
        self.max_steps = 200
        
        # define goals
        self.goals = [
            (9, 3),      # Top-left room
            (4, 9),     # Top-right room
            (10, 10)      # Bottom-center room
        ]
        self.num_objectives = 3
        self._check_connectivity()
        
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

        for x in range(1, self.grid_size - 1):
            if x == 3 or x == 9:
                continue
            walls.add((x, 6))
        
        for x in range(1, self.grid_size - 1):
            if x == 3 or x == 9:
                continue
            walls.add((x, 7))
        return walls
    
    def _neighbors(self, cell):
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield (nx, ny)

    def _reachable_goals(self):
        start = self.agent_start
        if start in self.walls:
            return set()
        visited = {start}
        queue = [start]
        while queue:
            cell = queue.pop(0)
            for nxt in self._neighbors(cell):
                if nxt in visited or nxt in self.walls:
                    continue
                visited.add(nxt)
                queue.append(nxt)
        return {g for g in self.goals if g in visited}

    def _check_connectivity(self):
        reachable = self._reachable_goals()
        print(f"[FourRoomEnv] Reachable goals: {sorted(reachable)}")
        if len(reachable) != len(self.goals):
            missing = [g for g in self.goals if g not in reachable]
            print(f"[FourRoomEnv] Unreachable goals: {missing}")

    def step(self, action):
        # actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        
        # stochastic transitions
        if self.np_random.rand() < 0.1:
            action = self.np_random.randint(0, 4)

        new_pos = self._get_next_pos(self.agent_pos, action)
        
        # if new position is a wall, stay in place
        if new_pos in self.walls:
            new_pos = self.agent_pos
        
        self.agent_pos = new_pos
        
        # check for rewards
        reward = np.zeros(self.num_objectives, dtype=np.float32)
        for i, goal in enumerate(self.goals):
            if self.agent_pos == goal:
                reward[i] = 1.0

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps
        obs = np.array(self.agent_pos)
        info = {'obj': reward}
        if truncated or terminated:
            return obs, 0.0, terminated, truncated, info
        return obs, 0.0, terminated, truncated, info
    

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

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self.np_random.seed(seed)
            except AttributeError:
                self.np_random = np.random.RandomState(seed)
        self.agent_pos = (2, 2)
        self.step_count = 0
        self.visited = set()
        return np.array(self.agent_pos), {}

    def render(self, mode='human'):
        # Implement the render function to visualize the environment
        pass
