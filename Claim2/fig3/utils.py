from gym import Wrapper
from gym.spaces.box import Box
import gym
try:
    import d4rl  # optional; only needed for MOOfflineEnv with D4RL datasets
except ImportError:
    d4rl = None
import environments
import numpy as np
import pickle

class MOOfflineEnv(Wrapper):
    def __init__(self, env_name, dataset='d4rl', num_objective=2, reset_noise_scale=1e-3):
        self.num_objective = num_objective
        self.env_name = env_name
        self.dataset = dataset
        if dataset == 'd4rl':
            env = gym.make(f"{self.env_name.lower()}-medium-v2")
        else:
            env = gym.make(f"{env_name}")
        env.reward_space = np.zeros((num_objective,))
        self._max_episode_steps = env._max_episode_steps
        super(MOOfflineEnv, self).__init__(env)

        if self.env.spec.id=='MO-Hopper-v2':
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def cal_reward_in_d4rl(self, action):
        xposbefore = self.env.sim.data.qpos[0]
        obs, reward, done, info = self.env.step(action)
        xposafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward1 = (xposafter - xposbefore)/self.env.dt + alive_bonus
        reward2 = 4.0 - 1.0 * np.square(action).sum() + alive_bonus
        if self.num_objective == 1:
            reward = np.array([reward])
        else:
            reward = np.array([reward1, reward2])
        return obs, reward, done, info

    def step(self, action):
        if self.env.spec.id=='MO-Hopper-v2':
            action = action * np.array([2, 2, 4])

        if self.dataset == 'd4rl':
            obs, reward, done, info = self.cal_reward_in_d4rl(action)
        else:
            obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info  
    
    def get_dataset(self, dataset_type):
        if self.dataset == 'd4rl':
            env = gym.make(f"{self.env_name.lower()}-{dataset_type}-v2")
            dataset = env.get_dataset()
            return self.d4rl2d4morl(dataset, env)
        elif self.dataset == 'd4morl':
            dataset_path = f"./data/{self.env_name}/{self.env_name}_50000_{dataset_type}.pkl"
            with open(dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            return dataset
    
    def d4rl2d4morl(self, dataset, env, max_episode_len=1000):
        def calculate_reward_vector(info_qpos, action, timeouts, terminals ): 
            N = action.shape[0]
            reward1, reward2 = [], []
            episode_step = 0
            for i in range(N):
                alive_bonus = 1.0
                xposafter, xposbefore = info_qpos[min(i+1, N-1), 0], info_qpos[i, 0]
                r1 = (xposafter - xposbefore)/env.dt + alive_bonus
                r2 = 4.0 - 1.0 * np.square(action[i]).sum() + alive_bonus

                if bool(terminals[i]) or timeouts[i]:
                    r1 = reward1[-1]  # Use the previous reward to approximate the current reward in the last step
                    episode_step = 0

                reward1.append(r1)
                reward2.append(r2)
                episode_step += 1

            reward1, reward2 = np.array(reward1), np.array(reward2)
            return np.stack([reward1, reward2], axis=1)

        reward = calculate_reward_vector(dataset['infos/qpos'], dataset['actions'], 
                                                    dataset['timeouts'], dataset['terminals'])
        assert len(reward)==len(dataset['observations'])
        step = 0
        ret = np.zeros((reward.shape[-1]))
        observations, actions, next_observations, raw_rewards, terminals = [], [], [], [], []
        d4morl_dataset = []
        for i, rw in enumerate(reward):
            step += 1
            ret += rw
            observations.append(dataset['observations'][i])
            actions.append(dataset['actions'][i])
            next_observations.append(dataset['next_observations'][i])
            raw_rewards.append(rw)

            if dataset['terminals'][i] or dataset['timeouts'][i] or step==max_episode_len:
                terminals.append(True)
                preference = (ret / np.linalg.norm(ret, ord=1)).reshape(1, -1).repeat(len(raw_rewards), 0)
                d4morl_dataset.append({
                    'observations': np.array(observations),
                    'actions': np.array(actions),
                    'next_observations': np.array(next_observations),
                    'raw_rewards': np.array(raw_rewards),
                    'terminals': np.array(terminals),
                    'preference': preference,
                }) 
                step = 0
                ret = np.zeros((reward.shape[-1]))
                observations, actions, next_observations, raw_rewards, terminals = [], [], [], [], []
            else:
                terminals.append(False)
        return d4morl_dataset

    def get_normalized_score(self, tot_rewards):
        if self.num_objective == 1:
            return np.array([self.env.get_normalized_score(tot_rewards[0])])
        else:
            return tot_rewards
        
        
def normalization(x, mean, std):
    x = (x - mean) / std
    return x

def min_max_normalization(x, min, max, eps=1e-8):
    x = (x - min) / (max - min + eps)
    return x
