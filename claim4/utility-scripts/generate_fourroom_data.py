import numpy as np
import gym
import os
import sys
# for relative imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import environments
import pickle
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


def _optimal_action_to_goal(env, goal):
    """Select the first action along a shortest path to the goal (BFS on grid)."""
    start = env.agent_pos
    if start == goal:
        return env.action_space.sample()

    grid_size = env.grid_size
    walls = env.walls
    # (action, dx, dy) for consistent neighbor ordering
    moves = [
        (0, 0, -1),  # UP
        (1, 0, 1),   # DOWN
        (2, -1, 0),  # LEFT
        (3, 1, 0),   # RIGHT
    ]

    from collections import deque
    queue = deque([start])
    parent = {start: (None, None)}  # pos -> (prev_pos, action)

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for action, dx, dy in moves:
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if nx < 0 or ny < 0 or nx >= grid_size or ny >= grid_size:
                continue
            if nxt in walls:
                continue
            if nxt in parent:
                continue
            parent[nxt] = ((x, y), action)
            queue.append(nxt)

    if goal not in parent:
        return env.action_space.sample()

    # Walk back from goal to find the first action from start
    cur = goal
    action = None
    while True:
        prev, act = parent[cur]
        if prev is None:
            break
        action = act
        if prev == start:
            break
        cur = prev
    return action if action is not None else env.action_space.sample()


def generate_offline_data(
    env_name,
    num_trajectories=300,
    quality='amateur',
    preference_dist='uniform',
    max_steps=200,
    behavior='random',
    goal_mix=None,
    epsilon_optimal=0.0,
    random_frac=0.5,
):
    """Generate offline dataset by running a behavior policy."""
    env = gym.make(env_name)
    if behavior in ('optimal_mix', 'mixed_optimal'):
        unwrapped = env.unwrapped
        goal_probs = _parse_goal_mix(goal_mix, len(unwrapped.goals))

    # If mixing, generate random trajectories first to estimate goal skew.
    random_goal_counts = None
    optimal_goal_probs = None
    if behavior == 'mixed_optimal':
        num_random = int(round(num_trajectories * random_frac))
        num_random = max(0, min(num_random, num_trajectories))
        random_goal_counts = np.zeros(len(unwrapped.goals), dtype=int)
    else:
        num_random = 0
    
    trajectories = []
    
    for traj_idx in tqdm(range(num_trajectories), desc="Generating trajectories"):
        obs = env.reset()
        goal_idx = None
        if behavior == 'optimal_mix':
            goal_idx = np.random.choice(len(unwrapped.goals), p=goal_probs)
        elif behavior == 'mixed_optimal':
            if traj_idx < num_random:
                goal_idx = None
            else:
                if optimal_goal_probs is None:
                    # After random block, compute remaining goal needs.
                    target = goal_probs * num_trajectories
                    remaining = target - random_goal_counts
                    remaining = np.clip(remaining, 0, None)
                    if remaining.sum() == 0:
                        optimal_goal_probs = np.ones(len(unwrapped.goals), dtype=np.float32) / len(unwrapped.goals)
                    else:
                        optimal_goal_probs = remaining / remaining.sum()
                goal_idx = np.random.choice(len(unwrapped.goals), p=optimal_goal_probs)
        
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
            elif behavior == 'optimal_mix':
                if np.random.rand() < epsilon_optimal:
                    action = env.action_space.sample()
                else:
                    goal = unwrapped.goals[goal_idx]
                    action = _optimal_action_to_goal(unwrapped, goal)
            elif behavior == 'mixed_optimal':
                if traj_idx < num_random:
                    action = env.action_space.sample()
                else:
                    if np.random.rand() < epsilon_optimal:
                        action = env.action_space.sample()
                    else:
                        goal = unwrapped.goals[goal_idx]
                        action = _optimal_action_to_goal(unwrapped, goal)
            else:
                raise ValueError("behavior must be 'random', 'optimal_mix', or 'mixed_optimal'")
            
            next_obs, _, done, info = env.step(action)
            reward = info['obj']  # Multi-objective reward vector
            
            trajectory['observations'].append(obs)
            trajectory['next_observations'].append(next_obs)
            trajectory['actions'].append(action)
            trajectory['raw_rewards'].append(reward)
            trajectory['terminals'].append(float(done))
            
            obs = next_obs
            if done:
                if behavior == 'mixed_optimal' and traj_idx < num_random:
                    if reward.max() > 0:
                        random_goal_counts[np.argmax(reward)] += 1
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
    parser.add_argument(
        "--preference_dist",
        type=str,
        default="uniform",
        help="Dataset tag used in the output filename (e.g., uniform, wide, narrow, or custom).",
    )
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--behavior", type=str, default="random", choices=["random", "optimal_mix", "mixed_optimal"], help="Behavior policy type")
    parser.add_argument("--goal_mix", type=str, default=None, help="Comma-separated mix for goals, e.g. '0.8,0.1,0.1'")
    parser.add_argument("--epsilon_optimal", type=float, default=0.0, help="Random action probability for optimal_mix")
    parser.add_argument("--random_frac", type=float, default=0.5, help="Fraction of random trajectories for mixed_optimal")
    args = parser.parse_args()
    
    generate_offline_data(
        args.env_name,
        args.num_trajectories,
        args.quality,
        args.preference_dist,
        args.max_steps,
        behavior=args.behavior,
        goal_mix=args.goal_mix,
        epsilon_optimal=args.epsilon_optimal,
        random_frac=args.random_frac,
    )
