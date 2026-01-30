import numpy as np
import gym


class MoFourRoomGymEnv(gym.Env):
    """Gym-style adapter for mo-gymnasium's four-room-v0 environment.

    This environment presents the legacy Gym API expected by the current
    codebase:
    - reset() -> obs
    - step() -> (obs, reward_scalar, done, info)

    It exposes the multi-objective reward vector via info["obj"] and returns
    a scalar reward of 0.0 (ignored by callers in this project).
    """

    def __init__(self):
        import mo_gymnasium as mo_gym

        self._env = mo_gym.make("four-room-v0")
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        # Known objective count for the FourRoom benchmark.
        self.num_objectives = 3

        # Gym's TimeLimit wrapper may set this; provide a sensible default.
        self._max_episode_steps = 200

        self._last_reset_info = {}

    def reset(self, seed=None, **kwargs):
        result = self._env.reset(seed=seed, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_reset_info = dict(info)
        return obs

    def step(self, action):
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = result
            done = bool(done)

        if info is None:
            info = {}
        else:
            info = dict(info)

        reward_vec = np.asarray(reward, dtype=np.float32)
        info.setdefault("obj", reward_vec)

        # The existing codebase ignores the scalar reward and reads info["obj"].
        reward_scalar = 0.0
        return obs, reward_scalar, done, info

    def seed(self, seed=None):
        # Legacy Gym API: forward to reset(seed=...).
        self.reset(seed=seed)
        return [] if seed is None else [seed]

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        return self._env.close()

