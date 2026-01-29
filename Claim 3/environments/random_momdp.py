### This file is used for creating the class describing the Multi-Objective Markov Decision Process environment
# for evaluating claim 3 in our experiments


import gym
import numpy as np
from gym import spaces
from typing import Dict, Any, Tuple, Optional


class RandomMOMDP(gym.Env):
    """Exact Random MOMDP from FairDICE - compatible with main.py pipeline."""

    def __init__(self, seed=42):
        super().__init__()

        # set by paper
        self.num_states = 50
        self.num_actions = 4
        self.num_objectives = 3
        self.gamma = 0.95
        self.max_steps = 100

        # State 0 is initial state
        self.initial_state = 0

        # Gym spaces for main.py
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_states,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_actions)

        # Set seed
        self.seed(seed)

        # Generate environment
        self._generate_transitions()
        self._generate_rewards()

        self.current_step = 0
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def _generate_transitions(self):
        """Dirichet(1,1,1,1) distribution over exactly 4 next states per (s,a)."""
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                # Pick 4 distinct next states
                next_states = self.np_random.choice(self.num_states, size=4, replace=False)
                # Dir(1,1,1,1) = uniform
                probs = np.ones(4) / 4.0
                self.P[s, a, next_states] = probs

    def _one_hot(self, s: int) -> np.ndarray:
        x = np.zeros(self.num_states, dtype=np.float32)
        x[s] = 1.0
        return x

    def _generate_rewards(self):
        """3 goal states from 1-49 with one-hot rewards."""
        self.R = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Pick 3 goal states from states 1-49
        goal_states = self.np_random.choice(
            np.arange(1, 50),
            size=3,
            replace=False
        )

        for goal_idx, goal_state in enumerate(goal_states):
            # One-hot reward for this objective
            one_hot = np.zeros(self.num_objectives)
            one_hot[goal_idx] = 1.0
            self.R[goal_state, :, :] = one_hot  # All actions in goal state

    def reset(self, seed=None, options: Optional[Dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset - returns state 0 (fixed initial state)."""
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        self.state = self.initial_state
        return self._one_hot(self.state), {}

    def step(self, action: int) -> Tuple[Any, np.ndarray, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        current_state = self.state
        self.current_step += 1

        # Sample next state
        next_state_dist = self.P[current_state, action]
        self.state = self.np_random.choice(self.num_states, p=next_state_dist)

        # Multi-objective reward vector
        reward_vector = self.R[current_state, action].copy()

        done = self.current_step >= self.max_steps  # combine terminated & truncated

        obs = self._one_hot(self.state)
        info = {}

        return obs, reward_vector, done, info

    @property
    def obj_dim(self) -> int:
        """CRITICAL: main.py expects this attribute for reward_dim."""
        return self.num_objectives


    def evaluate_policy(self, policy_weights: np.ndarray, num_episodes: int = 100) -> np.ndarray:
        """Exact policy evaluation for JAX policy."""
        returns = np.zeros(self.num_objectives)

        for _ in range(num_episodes):
            obs, _ = self.reset()
            traj_reward = np.zeros(self.num_objectives)
            step = 0

            while step < self.max_steps:
                # Convert JAX policy to numpy for evaluation
                state_idx = int(np.argmax(obs))
                action_probs = policy_weights[state_idx]  # Assumes policy[state] shape (4,)
                action = self.np_random.choice(self.num_actions, p=action_probs)

                obs, reward_vector, terminated, truncated, _ = self.step(action)
                traj_reward += reward_vector * (self.gamma ** step)
                step += 1

                if terminated or truncated:
                    break

            returns += traj_reward

        return returns / num_episodes


# Register with gym for gym.make("RandomMOMDP-v0")
gym.register(
    id="RandomMOMDP-v0",
    entry_point="environments.random_momdp:RandomMOMDP",
)
