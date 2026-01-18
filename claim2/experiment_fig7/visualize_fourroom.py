"""Visualize FourRoom trajectories with a trained policy."""

import numpy as np
import gym
import environments
import jax
from FairDICE import init_train_state, get_model
from utils import normalization
from types import SimpleNamespace
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualize_episode(env, policy, config, num_steps=400, is_discrete=True, use_greedy=True):
    """Run one episode and collect trajectory data."""
    state = env.reset()
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'done': False
    }
    
    for step in range(num_steps):
        trajectory['states'].append(state.copy())
        
        # Normalize state
        s_t = normalization(state, config.state_mean, config.state_std)
        s_t = s_t.reshape(1, -1)  # Add batch dimension
        
        # Get action from policy
        output = policy(s_t)
        
        if is_discrete:
            logits, probs = output
            if use_greedy:
                action = int(np.argmax(np.array(probs[0])))  # Greedy action
            else:
                action = int(np.array(probs[0]).argmax())  # Same as greedy
        else:
            dist = output
            action = float(dist.mean()[0])
        
        # Step environment
        next_state, _, done, info = env.step(action)
        reward = info['obj']
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        
        state = next_state
        
        if done:
            trajectory['done'] = True
            break
    
    return trajectory

def create_static_visualization(trajectory, env, save_path=None):
    """Create a static image showing the trajectory."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    grid_size = env.grid_size
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) in env.walls:
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=0, 
                                        edgecolor='none', facecolor='black', alpha=0.8)
                ax.add_patch(rect)
    
    # Draw goals
    colors = ['red', 'green', 'blue']
    for i, (goal_x, goal_y) in enumerate(env.goals):
        circle = patches.Circle((goal_x, goal_y), 0.3, color=colors[i], alpha=0.7)
        ax.add_patch(circle)
        ax.text(goal_x, goal_y-0.6, f'Goal {i+1}', ha='center', fontsize=10)
    
    # Draw trajectory path
    states = np.array(trajectory['states'])
    ax.plot(states[:, 0], states[:, 1], 'y-', linewidth=2, alpha=0.6, label='Agent Path')
    
    # Draw start position
    ax.plot(states[0, 0], states[0, 1], 'go', markersize=12, label='Start')
    
    # Draw end position
    ax.plot(states[-1, 0], states[-1, 1], 'r*', markersize=20, label='End')
    
    # Highlight goal hits
    rewards = np.array(trajectory['rewards'])
    for i, reward in enumerate(rewards):
        if np.any(reward > 0):  # Goal was hit
            goal_idx = np.argmax(reward)
            state_pos = states[i]
            ax.plot(state_pos[0], state_pos[1], 'o', color=colors[goal_idx], 
                   markersize=8, alpha=0.8)
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'FourRoom Trajectory (Steps: {len(states)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    return fig

def create_heatmap_visualization(trajectories, env, save_path=None):
    """Create a heatmap showing state visitation frequency."""
    grid_size = env.grid_size
    heatmap = np.zeros((grid_size, grid_size))
    
    # Count visits to each cell
    for trajectory in trajectories:
        states = np.array(trajectory['states'])
        for x, y in states:
            if 0 <= x < grid_size and 0 <= y < grid_size:
                heatmap[int(y), int(x)] += 1
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw heatmap
    im = ax.imshow(heatmap, cmap='gray_r', alpha=0.7, origin='upper')
    
    # Draw walls
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) in env.walls:
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=0, 
                                        edgecolor='none', facecolor='black', alpha=0.8)
                ax.add_patch(rect)
    
    # Draw goals
    colors = ['red', 'green', 'blue']
    for i, (goal_x, goal_y) in enumerate(env.goals):
        circle = patches.Circle((goal_x, goal_y), 0.3, color=colors[i], alpha=0.9, 
                               edgecolor='white', linewidth=2)
        ax.add_patch(circle)
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('State Visitation Heatmap')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Visit Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model (optional)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to search for latest model")
    parser.add_argument("--num_episodes", type=int, default=300, help="Number of episodes to visualize")
    parser.add_argument("--max_steps", type=int, default=400, help="Max steps per episode")
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--random_policy", action="store_true", help="Use random policy instead of trained")
    
    args = parser.parse_args()
    
    # Load environment
    env = gym.make(args.env_name)
    np.random.seed(args.seed)
    
    # Load the same data that was used for training to get normalization parameters
    data_path = f"./data/{args.env_name}/{args.env_name}_50000_amateur_uniform.pkl"
    if os.path.exists(data_path):
        import pickle
        with open(data_path, "rb") as f:
            trajs = pickle.load(f)
        all_states = np.concatenate([traj['observations'] for traj in trajs], axis=0)
        state_mean = all_states.mean(axis=0)
        state_std = all_states.std(axis=0) + 1e-8
        print(f"Loaded normalization params from data: mean={state_mean}, std={state_std}")
    else:
        # Fallback to approximate values
        state_mean = np.array([6.0, 6.0])
        state_std = np.array([4.0, 4.0])
        print(f"Warning: Using fallback normalization params: mean={state_mean}, std={state_std}")
    
    # Create config
    config = SimpleNamespace(
        seed=args.seed,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        is_discrete=True,
        hidden_dims=[64, 64],  # Must match training: 2 layers with dim 64
        layer_norm=True,
        temperature=1.0,
        tanh_squash_distribution=False,
        total_train_steps=10000,
        gamma=0.99,
        beta=0.001,
        divergence="SOFT_CHI",
        gradient_penalty_coeff=1e-4,
        nu_lr=3e-4,
        policy_lr=3e-4,
        mu_lr=3e-4,
        state_mean=state_mean,
        state_std=state_std,
        reward_dim=3,
    )
    
    # Resolve model path if not provided
    if args.model_path is None and not args.random_policy:
        if os.path.isdir(args.results_dir):
            candidate_paths = []
            for entry in os.listdir(args.results_dir):
                run_dir = os.path.join(args.results_dir, entry)
                model_dir = os.path.join(run_dir, "model")
                if os.path.isdir(model_dir):
                    candidate_paths.append(model_dir)

            if candidate_paths:
                args.model_path = os.path.abspath(max(candidate_paths, key=os.path.getmtime))
                print(f"Using latest model from {args.model_path}")
            else:
                print(f"No model directories found in {args.results_dir}. Using random policy.")
                args.random_policy = True
        else:
            print(f"Results directory not found: {args.results_dir}. Using random policy.")
            args.random_policy = True

    # Normalize provided model path to absolute for Orbax
    if args.model_path is not None:
        args.model_path = os.path.abspath(args.model_path)

    # Get policy
    if args.random_policy or args.model_path is None:
        print("Using random policy for visualization...")
        train_state = init_train_state(config)
        policy, _, _ = get_model(train_state.policy_state)
    else:
        print(f"Loading model from {args.model_path}...")
        if not os.path.exists(args.model_path):
            print(f"Model path not found: {args.model_path}")
            return
        else:
            try:
                from FairDICE import load_model
                train_state = load_model(args.model_path, config)
                policy, _, _ = get_model(train_state.policy_state)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {e}")
                return
    
    # Generate trajectories
    print(f"Generating {args.num_episodes} episodes...")
    trajectories = []
    total_rewards = []
    
    for ep in range(args.num_episodes):
        trajectory = visualize_episode(env, policy, config, num_steps=args.max_steps, is_discrete=True, use_greedy=True)
        trajectories.append(trajectory)
        
        # Calculate total rewards per objective
        rewards = np.array(trajectory['rewards'])
        total_reward = np.sum(rewards, axis=0)
        total_rewards.append(total_reward)
        
        print(f"Episode {ep+1}: Steps={len(trajectory['states'])}, "
              f"Rewards={total_reward}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Heatmap of all trajectories
    create_heatmap_visualization(trajectories, env, 
                                 save_path="visitation_heatmap.png")
    
    # Summary statistics
    total_rewards = np.array(total_rewards)
    print(f"\nAverage rewards across episodes:")
    print(f"  Objective 1: {total_rewards[:, 0].mean():.2f} ± {total_rewards[:, 0].std():.2f}")
    print(f"  Objective 2: {total_rewards[:, 1].mean():.2f} ± {total_rewards[:, 1].std():.2f}")
    print(f"  Objective 3: {total_rewards[:, 2].mean():.2f} ± {total_rewards[:, 2].std():.2f}")

if __name__ == "__main__":
    main()
