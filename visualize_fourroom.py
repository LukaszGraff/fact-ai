"""Visualize FourRoom policies.

Supports:
- Single-model visualization (existing behavior): roll out episodes and plot policy heatmap.
- Sweep aggregation: load many saved models under --sweep_dir and produce one aggregated policy heatmap.
"""

import numpy as np
import gym
import jax
import jax.numpy as jnp
from FairDICE import init_train_state, get_model as fairdice_get_model
from FairDICE import load_model as fairdice_load_model
from utils import normalization
from types import SimpleNamespace
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import json
import pickle
from typing import Optional


def _load_normalization_from_data(env_name: str, quality: str = "amateur", preference_dist: str = "uniform"):
    data_path = f"./data/{env_name}/{env_name}_50000_{quality}_{preference_dist}.pkl"
    if not os.path.exists(data_path):
        return None
    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
    all_states = np.concatenate([traj["observations"] for traj in trajs], axis=0)
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0) + 1e-8
    return state_mean, state_std


def _select_model_fns(config):
    learner = getattr(config, "learner", None)
    # Currently only FairDICE is supported
    return fairdice_load_model, fairdice_get_model


def _resolve_run_dirs_with_models(sweep_dir: str):
    run_dirs = []
    for entry in sorted(os.listdir(sweep_dir)):
        run_dir = os.path.join(sweep_dir, entry)
        model_dir = os.path.join(run_dir, "model")
        if os.path.isdir(model_dir):
            run_dirs.append(run_dir)
    return run_dirs


def _load_config_for_run(run_dir: str, env, fallback_state_mean, fallback_state_std):
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        if "state_mean" in cfg and cfg["state_mean"] is not None:
            cfg["state_mean"] = np.array(cfg["state_mean"], dtype=np.float32)
        if "state_std" in cfg and cfg["state_std"] is not None:
            cfg["state_std"] = np.array(cfg["state_std"], dtype=np.float32)
        if "reward_min" in cfg and cfg["reward_min"] is not None:
            cfg["reward_min"] = np.array(cfg["reward_min"], dtype=np.float32)
        if "reward_max" in cfg and cfg["reward_max"] is not None:
            cfg["reward_max"] = np.array(cfg["reward_max"], dtype=np.float32)
        config = SimpleNamespace(**cfg)
    else:
        config = SimpleNamespace(
            seed=0,
            env_name=getattr(env, "spec", None).id if getattr(env, "spec", None) else "MO-FourRoom-v2",
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            reward_dim=getattr(env, "num_objectives", 3),
            is_discrete=True,
            hidden_dims=[256, 256],
            layer_norm=True,
            temperature=1.0,
            tanh_squash_distribution=False,
            total_train_steps=100_000,
            gamma=0.99,
            beta=0.01,
            divergence="SOFT_CHI",
            gradient_penalty_coeff=1e-4,
            nu_lr=3e-4,
            policy_lr=3e-4,
            mu_lr=3e-4,
        )

    if not hasattr(config, "seed") or config.seed is None:
        config.seed = 0
    if not hasattr(config, "state_dim") or config.state_dim is None:
        config.state_dim = env.observation_space.shape[0]
    if not hasattr(config, "action_dim") or config.action_dim is None:
        config.action_dim = env.action_space.n
    if not hasattr(config, "reward_dim") or config.reward_dim is None:
        config.reward_dim = getattr(env, "num_objectives", 3)
    if not hasattr(config, "is_discrete") or config.is_discrete is None:
        config.is_discrete = True
    if not hasattr(config, "hidden_dims") or config.hidden_dims is None:
        config.hidden_dims = [256, 256]
    if not hasattr(config, "layer_norm") or config.layer_norm is None:
        config.layer_norm = True
    if not hasattr(config, "temperature") or config.temperature is None:
        config.temperature = 1.0
    if not hasattr(config, "tanh_squash_distribution") or config.tanh_squash_distribution is None:
        config.tanh_squash_distribution = False
    if not hasattr(config, "policy_lr") or config.policy_lr is None:
        config.policy_lr = 3e-4
    if not hasattr(config, "nu_lr") or config.nu_lr is None:
        config.nu_lr = 3e-4
    if not hasattr(config, "mu_lr") or config.mu_lr is None:
        config.mu_lr = 3e-4
    if not hasattr(config, "total_train_steps") or config.total_train_steps is None:
        config.total_train_steps = 100_000
    if not hasattr(config, "gamma") or config.gamma is None:
        config.gamma = 0.99
    if not hasattr(config, "beta") or config.beta is None:
        config.beta = 0.01
    if not hasattr(config, "divergence") or config.divergence is None:
        config.divergence = "SOFT_CHI"
    if not hasattr(config, "gradient_penalty_coeff") or config.gradient_penalty_coeff is None:
        config.gradient_penalty_coeff = 1e-4

    if not hasattr(config, "state_mean") or config.state_mean is None:
        config.state_mean = fallback_state_mean
    if not hasattr(config, "state_std") or config.state_std is None:
        config.state_std = fallback_state_std

    config.is_discrete = True
    return config

def visualize_episode(
    env,
    policy,
    config,
    num_steps=400,
    is_discrete=True,
    use_greedy=True,
    rng_key=None,
    action_sampler="policy",
):
    """Run one episode and collect trajectory data.
    
    Args:
        use_greedy: If True, use argmax (greedy). If False, sample from distribution (stochastic).
        rng_key: JAX random key for stochastic sampling. Required if use_greedy=False.
        action_sampler: "policy" to use the model policy, "random_uniform" for uniform random actions.
    """
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        state = reset_out[0]
    else:
        state = reset_out
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'done': False
    }
    
    # Store the initial state
    trajectory['states'].append(state.copy())

    for step in range(num_steps):
        
        if action_sampler == "random_uniform":
            action = env.action_space.sample()
        else:
            # Normalize state
            s_t = normalization(state, config.state_mean, config.state_std)
            s_t = s_t.reshape(1, -1)  # Add batch dimension

            # Get action from policy
            output = policy(s_t)

            if is_discrete:
                # DiscretePolicy returns a Categorical distribution
                dist = output
                if use_greedy:
                    # Greedy: take argmax of logits
                    action = int(jnp.argmax(dist.logits[0]))
                else:
                    # Stochastic: sample from the policy's probability distribution
                    if rng_key is not None:
                        rng_key, subkey = jax.random.split(rng_key)
                        action = int(jax.random.categorical(subkey, dist.logits[0]))
                    else:
                        # Fallback to numpy sampling if no JAX key provided
                        probs_np = np.array(jax.nn.softmax(dist.logits[0]))
                        action = int(np.random.choice(len(probs_np), p=probs_np))
            else:
                dist = output
                action = float(dist.mean()[0])
        
        # Step environment
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_state, _, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, _, done, info = step_out
        reward = info['obj']
        
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)

        # Store the post-step state so terminal goal states are included in heatmaps.
        state = next_state
        trajectory['states'].append(state.copy())
        
        if done:
            trajectory['done'] = True
            break
    
    return trajectory


def _add_normalized_episode_visits(heatmap, states, grid_size, per_state_cap=3):
    """Accumulate per-episode visitation counts with per-state cap (no normalization)."""
    episode_map = np.zeros_like(heatmap, dtype=np.float64)
    for x, y in states:
        if 0 <= int(x) < grid_size and 0 <= int(y) < grid_size:
            y_idx = int(y)
            x_idx = int(x)
            if episode_map[y_idx, x_idx] < per_state_cap:
                episode_map[y_idx, x_idx] += 1.0
    heatmap += episode_map
    return heatmap

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
    # rewards[t] corresponds to transition from states[t] -> states[t+1]
    rewards = np.array(trajectory['rewards'])
    for t, reward in enumerate(rewards):
        if np.any(reward > 0):  # Goal was hit
            goal_idx = int(np.argmax(reward))
            state_pos = states[t + 1]
            ax.plot(state_pos[0], state_pos[1], 'o', color=colors[goal_idx],
                    markersize=8, alpha=0.8)
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # Inverted: y=0 at top, y=max at bottom
    ax.set_aspect('equal')
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
        heatmap = _add_normalized_episode_visits(
            heatmap, trajectory['states'], grid_size
        )
    
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
    ax.set_ylim(grid_size - 0.5, -0.5)  # Inverted: y=0 at top, y=max at bottom
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


def create_policy_heatmap_visualization(
    policy,
    config,
    env,
    trajectories=None,
    visitation_heatmap=None,
    save_path=None,
    title="Policy Visualization",
    show=True,
    policy_probs=None,
    per_state_cap=3,
):
    """Create a paper-style heatmap showing policy arrows and state visitation.
    
    This creates a visualization similar to the FairDICE paper with:
    - Blue intensity showing state visitation frequency
    - Arrows showing the policy's preferred action at each state
    - Dark red walls
    - Goals marked with colored squares
    """
    grid_size = env.grid_size
    
    # Calculate state visitation heatmap
    if visitation_heatmap is not None:
        heatmap = np.asarray(visitation_heatmap, dtype=np.float64)
    else:
        heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
        if trajectories is not None:
            for trajectory in trajectories:
                heatmap = _add_normalized_episode_visits(
                    heatmap, trajectory['states'], grid_size, per_state_cap=per_state_cap
                )
    
    # Normalize heatmap to [0, 1]
    if heatmap.max() > 0:
        heatmap_norm = heatmap / heatmap.max()
    else:
        heatmap_norm = heatmap
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create custom blue colormap (white to dark blue)
    colors_blue = ['#FFFFFF', '#E6F0FF', '#B3D1FF', '#80B3FF', '#4D94FF', '#1A75FF', '#0052CC', '#003D99', '#002966']
    blue_cmap = LinearSegmentedColormap.from_list('custom_blue', colors_blue)
    
    # Draw cells with blue intensity based on visitation
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) in env.walls:
                # Walls in dark red
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=0, 
                                        edgecolor='none', facecolor='#8B0000')
                ax.add_patch(rect)
            else:
                # Non-wall cells with blue intensity
                intensity = heatmap_norm[y, x]
                color = blue_cmap(intensity)
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=0.5, 
                                        edgecolor='#DDDDDD', facecolor=color)
                ax.add_patch(rect)
    
    # Draw policy arrows for each non-wall cell
    # Action mapping: UP=0, DOWN=1, LEFT=2, RIGHT=3
    action_dim = int(env.action_space.n)
    arrow_dx = {0: 0, 1: 0, 2: -0.35, 3: 0.35}  # x offset
    arrow_dy = {0: -0.35, 1: 0.35, 2: 0, 3: 0}  # y offset (inverted for display)
    
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) not in env.walls:
                if policy_probs is None:
                    state = np.array([x, y], dtype=np.float32)
                    s_t = normalization(state, config.state_mean, config.state_std)
                    s_t = s_t.reshape(1, -1)
                    output = policy(s_t)
                    probs = jax.nn.softmax(output.logits[0])
                else:
                    probs = policy_probs[y, x]

                # Draw arrows for all actions with opacity based on probability
                for action in range(action_dim):
                    dx = arrow_dx[action]
                    dy = arrow_dy[action]
                    alpha = float(probs[action])
                    if alpha <= 0:
                        continue
                    arrow_color = (0.3, 0.2, 0.2)
                    ax.arrow(
                        x,
                        y,
                        dx,
                        dy,
                        head_width=0.2,
                        head_length=0.1,
                        fc=arrow_color,
                        ec=arrow_color,
                        alpha=alpha,
                        linewidth=1.5,
                    )
    
    # Draw goals with colored squares (like in paper)
    goal_colors = ['#FF4444', '#44FF44', '#4444FF']  # Red, Green, Blue
    for i, (goal_x, goal_y) in enumerate(env.goals):
        # Outer square (border)
        rect_outer = patches.Rectangle((goal_x-0.35, goal_y-0.35), 0.7, 0.7, 
                                       linewidth=3, edgecolor='#00FF00', 
                                       facecolor='none')
        ax.add_patch(rect_outer)
    
    # Draw start position with orange square
    start_x, start_y = 2, 2  # Default start position
    rect_start = patches.Rectangle((start_x-0.35, start_y-0.35), 0.7, 0.7, 
                                   linewidth=3, edgecolor='#FFA500', 
                                   facecolor='none')
    ax.add_patch(rect_start)
    
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # y=0 at top
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved policy heatmap to {save_path}")

    if show:
        plt.show()
    return fig


def _aggregate_sweep_policy_heatmap(
    env,
    sweep_dir: str,
    env_name: str,
    seed: int,
    episodes_per_model: int,
    max_steps: int,
    stochastic: bool,
    limit_models: Optional[int],
    output_path: str,
    per_state_cap: int,
):
    unwrapped = env.unwrapped
    grid_size = env.grid_size
    fallback = _load_normalization_from_data(env_name)
    if fallback is None:
        raise FileNotFoundError(f"Normalization dataset not found under ./data/{env_name}.")
    fallback_mean, fallback_std = fallback

    run_dirs = _resolve_run_dirs_with_models(sweep_dir)
    if limit_models is not None:
        run_dirs = run_dirs[:limit_models]
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories with model/ found under {sweep_dir}")

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float64)
    action_dim = int(env.action_space.n)
    policy_prob_sum = np.zeros((grid_size, grid_size, action_dim), dtype=np.float64)
    policy_prob_count = np.zeros((grid_size, grid_size), dtype=np.float64)

    rng_key = jax.random.PRNGKey(seed)
    walls = getattr(unwrapped, 'walls', set())

    for idx, run_dir in enumerate(run_dirs):
        model_dir = os.path.join(run_dir, 'model')
        config = _load_config_for_run(run_dir, env, fallback_mean, fallback_std)
        load_model_fn, get_model_fn = _select_model_fns(config)
        train_state = load_model_fn(os.path.abspath(model_dir), config)
        policy, _, _ = get_model_fn(train_state.policy_state)
        policy.eval()

        grid_states = []
        grid_coords = []
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in walls:
                    continue
                grid_coords.append((x, y))
                grid_states.append(np.array([x, y], dtype=np.float32))

        if grid_states:
            grid_states_arr = np.stack(grid_states, axis=0)
            norm_states = normalization(grid_states_arr, config.state_mean, config.state_std)
            dist = policy(norm_states)
            probs = np.array(jax.nn.softmax(dist.logits, axis=-1))
            for (x, y), p_row in zip(grid_coords, probs):
                policy_prob_sum[y, x, :] += p_row
                policy_prob_count[y, x] += 1.0

        for _ in range(episodes_per_model):
            rng_key, ep_key = jax.random.split(rng_key)
            traj = visualize_episode(
                env,
                policy,
                config,
                num_steps=max_steps,
                is_discrete=True,
                use_greedy=not stochastic,
                rng_key=ep_key,
            )
            heatmap = _add_normalized_episode_visits(
                heatmap, traj['states'], grid_size, per_state_cap=per_state_cap
            )

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(run_dirs)} models")

    # Compute average probabilities
    with np.errstate(invalid='ignore', divide='ignore'):
        policy_probs = np.where(
            policy_prob_count[:, :, None] > 0,
            policy_prob_sum / np.maximum(policy_prob_count[:, :, None], 1e-12),
            0.0,
        )

    # Use any config for normalization; arrows use policy_probs and do not require policy.
    dummy_config = SimpleNamespace(state_mean=fallback_mean, state_std=fallback_std)
    create_policy_heatmap_visualization(
        policy=None,
        config=dummy_config,
        env=env,
        trajectories=None,
        visitation_heatmap=heatmap,
        save_path=output_path,
        title="Aggregate State Visitation and Policy Visualization",
        show=False,
        policy_probs=policy_probs,
        per_state_cap=per_state_cap,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model (optional)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to search for latest model")
    parser.add_argument("--sweep_dir", type=str, default=None, help="If set, aggregate across all run subdirs with model/ under this directory")
    parser.add_argument("--episodes_per_model", type=int, default=5, help="(Sweep mode) episodes per model")
    parser.add_argument("--limit_models", type=int, default=None, help="(Sweep mode) optional cap for quick tests")
    parser.add_argument("--output", type=str, default=None, help="Output image path (defaults: policy_heatmap.png or aggregate_policy_heatmap.png)")
    parser.add_argument("--per_state_cap", type=int, default=1, help="Per-episode cap for state visits in heatmaps.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to visualize")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--random_policy", action="store_true", help="Use random policy instead of trained")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action selection (sample from policy) instead of greedy")
    parser.add_argument(
        "--uniform_random",
        action="store_true",
        help="Visualize the uniform random behavior policy used for data collection",
    )
    
    args = parser.parse_args()
    
    # Load environment
    env = gym.make(args.env_name)
    np.random.seed(args.seed)

    if args.sweep_dir is not None:
        output_path = args.output or os.path.join(args.sweep_dir, "aggregate_policy_heatmap.png")
        _aggregate_sweep_policy_heatmap(
            env=env,
            sweep_dir=args.sweep_dir,
            env_name=args.env_name,
            seed=args.seed,
            episodes_per_model=args.episodes_per_model,
            max_steps=args.max_steps,
            stochastic=args.stochastic,
            limit_models=args.limit_models,
            output_path=output_path,
            per_state_cap=args.per_state_cap,
        )
        return
    
    # Load the same data that was used for training to get normalization parameters
    data_path = f"./data/{args.env_name}/{args.env_name}_50000_amateur_uniform.pkl"
    if os.path.exists(data_path):
        import pickle
        try:
            with open(data_path, "rb") as f:
                trajs = pickle.load(f)
            all_states = np.concatenate([traj['observations'] for traj in trajs], axis=0)
            state_mean = all_states.mean(axis=0)
            state_std = all_states.std(axis=0) + 1e-8
            print(f"Loaded normalization params from data: mean={state_mean}, std={state_std}")
        except (ModuleNotFoundError, AttributeError, ImportError) as exc:
            # Handle pickle compatibility issues across numpy versions.
            state_mean = np.array([6.0, 6.0])
            state_std = np.array([4.0, 4.0])
            print(f"Warning: Failed to load normalization params from {data_path}: {exc}")
            print(f"Warning: Using fallback normalization params: mean={state_mean}, std={state_std}")
    else:
        # Fallback to approximate values
        state_mean = np.array([6.0, 6.0])
        state_std = np.array([4.0, 4.0])
        print(f"Warning: Using fallback normalization params: mean={state_mean}, std={state_std}")
    
    # Try to load config from saved model directory
    config = None
    if args.model_path is not None and not args.random_policy and not args.uniform_random:
        run_dir = os.path.abspath(os.path.join(args.model_path, os.pardir))
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            if "state_mean" in config_dict:
                config_dict["state_mean"] = np.array(config_dict["state_mean"])
            if "state_std" in config_dict:
                config_dict["state_std"] = np.array(config_dict["state_std"])
            if "reward_min" in config_dict and config_dict["reward_min"] is not None:
                config_dict["reward_min"] = np.array(config_dict["reward_min"])
            if "reward_max" in config_dict and config_dict["reward_max"] is not None:
                config_dict["reward_max"] = np.array(config_dict["reward_max"])
            config = SimpleNamespace(**config_dict)
            print(f"Loaded config from {config_path}")
            print(f"  hidden_dims: {config.hidden_dims}, num_layers: {config.num_layers if hasattr(config, 'num_layers') else 'N/A'}")
    elif args.model_path is None and not args.random_policy and not args.uniform_random:
        # Find the model directory first to load config
        if os.path.isdir(args.results_dir):
            candidate_paths = []
            for entry in os.listdir(args.results_dir):
                run_dir = os.path.join(args.results_dir, entry)
                model_dir = os.path.join(run_dir, "model")
                if os.path.isdir(model_dir):
                    candidate_paths.append(run_dir)
            if candidate_paths:
                latest_run_dir = max(candidate_paths, key=os.path.getmtime)
                config_path = os.path.join(latest_run_dir, "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    # Convert lists back to numpy arrays where needed
                    if 'state_mean' in config_dict:
                        config_dict['state_mean'] = np.array(config_dict['state_mean'])
                    if 'state_std' in config_dict:
                        config_dict['state_std'] = np.array(config_dict['state_std'])
                    if 'reward_min' in config_dict and config_dict['reward_min'] is not None:
                        config_dict['reward_min'] = np.array(config_dict['reward_min'])
                    if 'reward_max' in config_dict and config_dict['reward_max'] is not None:
                        config_dict['reward_max'] = np.array(config_dict['reward_max'])
                    config = SimpleNamespace(**config_dict)
                    print(f"Loaded config from {config_path}")
                    print(f"  hidden_dims: {config.hidden_dims}, num_layers: {config.num_layers if hasattr(config, 'num_layers') else 'N/A'}")
    
    # Fallback to default config if not loaded
    if config is None:
        print("Warning: Could not load config from model directory, using defaults")
        config = SimpleNamespace(
            seed=args.seed,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            is_discrete=True,
            hidden_dims=[256, 256],
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
    else:
        # Override normalization params from data if loaded config doesn't have them or they differ
        config.state_mean = state_mean
        config.state_std = state_std
    
    # Resolve model path if not provided
    if args.model_path is None and not args.random_policy and not args.uniform_random:
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
    if args.uniform_random:
        print("Using uniform random behavior policy for visualization...")
        train_state = None
        policy, _, _ = None, None, None
    elif args.random_policy or args.model_path is None:
        print("Using random policy for visualization...")
        train_state = init_train_state(config)
        policy, _, _ = fairdice_get_model(train_state.policy_state)
    else:
        print(f"Loading model from {args.model_path}...")
        if not os.path.exists(args.model_path):
            print(f"Model path not found: {args.model_path}")
            return
        else:
            try:
                load_model_fn, get_model_fn = _select_model_fns(config)
                print("Calling load_model...")
                train_state = load_model_fn(args.model_path, config)
                print("load_model returned. Calling get_model...")
                policy, _, _ = get_model_fn(train_state.policy_state)
                print("Model loaded successfully!")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading model: {e}")
                return
    
    # Generate trajectories
    use_greedy = not args.stochastic
    print(f"Generating {args.num_episodes} episodes (action selection: {'greedy' if use_greedy else 'stochastic'})...")
    trajectories = []
    total_rewards = []
    rng_key = jax.random.PRNGKey(args.seed)
    
    for ep in range(args.num_episodes):
        rng_key, ep_key = jax.random.split(rng_key)
        action_sampler = "random_uniform" if args.uniform_random else "policy"
        trajectory = visualize_episode(
            env,
            policy,
            config,
            num_steps=args.max_steps,
            is_discrete=True,
            use_greedy=use_greedy,
            rng_key=ep_key,
            action_sampler=action_sampler,
        )
        trajectories.append(trajectory)
        
        # Calculate total rewards per objective
        rewards = np.array(trajectory['rewards'])
        total_reward = np.sum(rewards, axis=0)
        total_rewards.append(total_reward)
        
        print(f"Episode {ep+1}: Steps={len(trajectory['states'])}, "
              f"Rewards={total_reward}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Paper-style policy heatmap with arrows
    if args.output is not None:
        output_path = args.output
    else:
        if args.uniform_random:
            filename = "uniform_random_policy_heatmap.png"
        else:
            filename = "policy_heatmap.png"
        if args.model_path is not None:
            output_path = os.path.join(os.path.dirname(args.model_path), filename)
        else:
            output_path = filename

    policy_probs = None
    if args.uniform_random:
        grid_size = env.grid_size
        action_dim = int(env.action_space.n)
        policy_probs = np.zeros((grid_size, grid_size, action_dim), dtype=np.float32)
        walls = getattr(env.unwrapped, "walls", set())
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in walls:
                    continue
                policy_probs[y, x, :] = 1.0 / action_dim
    create_policy_heatmap_visualization(
        policy,
        config,
        env,
        trajectories,
        save_path=output_path,
        title="Uniform Random Behavior Policy" if args.uniform_random else "State Visitation and Policy Visualization",
        show=(args.output is None),
        policy_probs=policy_probs,
        per_state_cap=args.per_state_cap,
    )
    
    # Also generate the simple visitation heatmap
    
    
    # Summary statistics
    total_rewards = np.array(total_rewards)
    print(f"\nAverage rewards across episodes:")
    print(f"  Objective 1: {total_rewards[:, 0].mean():.2f} ± {total_rewards[:, 0].std():.2f}")
    print(f"  Objective 2: {total_rewards[:, 1].mean():.2f} ± {total_rewards[:, 1].std():.2f}")
    print(f"  Objective 3: {total_rewards[:, 2].mean():.2f} ± {total_rewards[:, 2].std():.2f}")

if __name__ == "__main__":
    main()
