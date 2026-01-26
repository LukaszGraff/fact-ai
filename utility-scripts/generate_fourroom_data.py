import numpy as np
import gym
import environments  # Register custom environments
import pickle
import os
from tqdm import tqdm

def _parse_goal_mix(goal_mix, num_goals):
    if goal_mix is None:
        return np.ones(num_goals, dtype=np.float32) / num_goals
    parts = [p.strip() for p in goal_mix.split(",") if p.strip()]
    if len(parts) != num_goals:
        raise ValueError(f"goal_mix must have {num_goals} comma-separated values")
    probs = np.array([float(p) for p in parts], dtype=np.float32)
    if np.any(probs < 0):
        raise ValueError("goal_mix values must be non-negative")
    if probs.sum() == 0:
        raise ValueError("goal_mix must sum to a positive value")
    return probs / probs.sum()


def _greedy_action_to_goal(env, goal):
    """Select a greedy action that minimizes Manhattan distance to a goal."""
    x, y = env.agent_pos
    candidates = [
        (0, (x, y - 1)),  # UP
        (1, (x, y + 1)),  # DOWN
        (2, (x - 1, y)),  # LEFT
        (3, (x + 1, y)),  # RIGHT
    ]
    best_action = None
    best_dist = float("inf")
    for action, (nx, ny) in candidates:
        if (nx, ny) in env.walls:
            continue
        dist = abs(nx - goal[0]) + abs(ny - goal[1])
        if dist < best_dist:
            best_dist = dist
            best_action = action
    if best_action is None:
        # If all moves blocked (shouldn't happen), fallback to random
        return env.action_space.sample()
    return best_action


def generate_offline_data(
    env_name,
    num_trajectories=300,
    quality='amateur',
    preference_dist='uniform',
    max_steps=400,
    behavior='random',
    goal_mix=None,
    epsilon_greedy=0.0,
):
    """Generate offline dataset by running a behavior policy."""
    env = gym.make(env_name)
    unwrapped = env.unwrapped
    goal_probs = _parse_goal_mix(goal_mix, len(unwrapped.goals))
    
    trajectories = []
    
    for traj_idx in tqdm(range(num_trajectories), desc="Generating trajectories"):
        obs = env.reset()
        goal_idx = None
        if behavior == 'greedy_mix':
            goal_idx = np.random.choice(len(unwrapped.goals), p=goal_probs)
        
        trajectory = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'raw_rewards': [],  # Multi-objective rewards
            'terminals': [],    # Terminal flags
        }
        
        for step in range(max_steps):
            if behavior == 'random':
                action = env.action_space.sample()
            elif behavior == 'greedy_mix':
                if np.random.rand() < epsilon_greedy:
                    action = env.action_space.sample()
                else:
                    goal = unwrapped.goals[goal_idx]
                    action = _greedy_action_to_goal(unwrapped, goal)
            else:
                raise ValueError("behavior must be 'random' or 'greedy_mix'")
            
            next_obs, _, done, info = env.step(action)
            reward = info['obj']  # Multi-objective reward vector
            
            trajectory['observations'].append(obs)
            trajectory['next_observations'].append(next_obs)
            trajectory['actions'].append(action)
            trajectory['raw_rewards'].append(reward)
            trajectory['terminals'].append(float(done))
            
            obs = next_obs
            if done:
                break
        
        # Convert lists to arrays
        trajectory['observations'] = np.array(trajectory['observations'])
        trajectory['next_observations'] = np.array(trajectory['next_observations'])
        trajectory['actions'] = np.array(trajectory['actions'])
        trajectory['raw_rewards'] = np.array(trajectory['raw_rewards'])
        trajectory['terminals'] = np.array(trajectory['terminals'])
        
        trajectories.append(trajectory)
    
    # Save
    os.makedirs(f"./data/{env_name}", exist_ok=True)
    save_path = f"./data/{env_name}/{env_name}_50000_{quality}_{preference_dist}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} trajectories to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2", help="Environment name")
    parser.add_argument("--num_trajectories", type=int, default=300, help="Number of trajectories to generate")
    parser.add_argument("--quality", type=str, default="amateur", choices=["expert", "amateur"], help="Data quality")
    parser.add_argument("--preference_dist", type=str, default="uniform", choices=["uniform", "wide", "narrow"], help="Preference distribution")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--behavior", type=str, default="random", choices=["random", "greedy_mix"], help="Behavior policy type")
    parser.add_argument("--goal_mix", type=str, default=None, help="Comma-separated mix for goals, e.g. '0.8,0.1,0.1'")
    parser.add_argument("--epsilon_greedy", type=float, default=0.0, help="Random action probability for greedy_mix")
    args = parser.parse_args()
    
    generate_offline_data(
        args.env_name,
        args.num_trajectories,
        args.quality,
        args.preference_dist,
        args.max_steps,
        behavior=args.behavior,
        goal_mix=args.goal_mix,
        epsilon_greedy=args.epsilon_greedy,
    )