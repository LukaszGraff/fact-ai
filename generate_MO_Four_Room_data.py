import numpy as np
try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym
import pickle
import os
from tqdm import tqdm
from fourroom_registration import ensure_fourroom_registered

def generate_offline_data(
    env_name,
    num_trajectories=300,
    quality='amateur',
    preference_dist='uniform',
    max_steps=500,
    seed=None,
    save_path=None,
):
    """Generate offline dataset by running random policy."""
    ensure_fourroom_registered()
    env = gym.make(env_name)
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "max_steps"):
        env.unwrapped.max_steps = max_steps
    if hasattr(env, "_max_episode_steps"):
        env._max_episode_steps = max_steps
    
    trajectories = []
    
    for traj_idx in tqdm(range(num_trajectories), desc="Generating trajectories"):
        if seed is not None:
            episode_seed = int(seed) + traj_idx
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(episode_seed)
            reset_out = env.reset(seed=episode_seed)
        else:
            reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs, _ = reset_out
        else:
            obs = reset_out
        
        trajectory = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'raw_rewards': [],  # Multi-objective rewards
        }
        
        for step in range(max_steps):
            action = env.action_space.sample()  # Random
            
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, _, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, _, done, info = step_out
            reward = info['obj']  # Multi-objective reward vector
            
            trajectory['observations'].append(obs)
            trajectory['next_observations'].append(next_obs)
            trajectory['actions'].append(action)
            trajectory['raw_rewards'].append(reward)
            
            obs = next_obs
            if done:
                break
        
        # Convert lists to arrays
        trajectory['observations'] = np.array(trajectory['observations'])
        trajectory['next_observations'] = np.array(trajectory['next_observations'])
        trajectory['actions'] = np.array(trajectory['actions'])
        trajectory['raw_rewards'] = np.array(trajectory['raw_rewards'])
        
        trajectories.append(trajectory)
    
    # Save
    os.makedirs(f"./data/{env_name}", exist_ok=True)
    if save_path is None:
        save_path = f"./data/{env_name}/{env_name}_50000_{quality}_{preference_dist}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {save_path}")

if __name__ == "__main__":
    generate_offline_data("MO-FourRoom-v2", num_trajectories=300, max_steps=500)
