import gym
from gym import spaces
import numpy as np


class RandomMOMDPEnv(gym.Env):
    def __init__(self, seed=0, n_states=50, n_actions=4, reward_dim=3, gamma=0.99, max_steps=200):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.max_steps = max_steps
        self.obj_dim = reward_dim
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_states,), dtype=np.float32)
        self._build_mdp(seed)
        self._init_state = 0
        self.state = self._init_state
        self.step_rng = np.random.default_rng(seed)
        self._max_episode_steps = max_steps
        self.step_count = 0

    def _build_mdp(self, seed):
        rng = np.random.default_rng(seed)
        self.goal_states = rng.choice(np.arange(1, self.n_states), size=self.reward_dim, replace=False)
        self.transition_next_states = np.zeros((self.n_states, self.n_actions, 4), dtype=np.int32)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                self.transition_next_states[s, a] = rng.choice(self.n_states, size=4, replace=False)
        self.transition_probs = rng.dirichlet(np.ones(4), size=(self.n_states, self.n_actions))

    def seed(self, seed=None):
        if seed is None:
            return
        self.step_rng = np.random.default_rng(seed)

    def _one_hot(self, state):
        obs = np.zeros((self.n_states,), dtype=np.float32)
        obs[state] = 1.0
        return obs

    def reset(self):
        self.state = self._init_state
        self.step_count = 0
        return self._one_hot(self.state)

    def step(self, action):
        action = int(action)
        next_candidates = self.transition_next_states[self.state, action]
        probs = self.transition_probs[self.state, action]
        next_state = self.step_rng.choice(next_candidates, p=probs)
        reward = np.zeros((self.reward_dim,), dtype=np.float32)
        done = False
        goal_match = np.where(self.goal_states == next_state)[0]
        if goal_match.size > 0:
            reward[goal_match[0]] = 1.0
            done = True
        self.state = int(next_state)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        obs = self._one_hot(self.state)
        info = {"obj": reward}
        return obs, reward, done, info
