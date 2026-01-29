import numpy as np
import jax
from utils import normalization, min_max_normalization
import os
import jax.numpy as jnp
import csv



def compute_metrics(returns):
    """
    returns: (num_episodes, num_objectives)
    Returns averaged metrics across episodes.
    """
    # Sum rewards per episode → Utilitarian welfare
    utilitarian = np.sum(returns, axis=1)  # shape: (num_episodes,)

    # Jain’s fairness index per episode
    sum_R = np.sum(returns, axis=1)                  # Σ Ri
    sum_R2 = np.sum(returns**2, axis=1)             # Σ Ri^2
    n = returns.shape[1]                             # number of objectives
    jain = (sum_R**2) / (n * sum_R2 + 1e-8)         # avoid div by zero

    # Nash social welfare per episode
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    nsw = np.sum(np.log(returns + eps), axis=1)

    # Average over episodes
    return {
        "utilitarian": np.mean(utilitarian),
        "jain": np.mean(jain),
        "nsw": np.mean(nsw)
    }

def evaluate_policy(config, policy, env, save_dir, num_episodes=50, max_steps=500, t_env=None, key=jax.random.PRNGKey(0)):
    """
    Evaluation function adapted for RandomMOMDP (multi-objective discrete environment).
    """
    policy.eval()
    raw_returns = []
    normalized_returns = []

    is_discrete = hasattr(config, 'is_discrete') and config.is_discrete

    @jax.jit
    def select_action(observation):
        output = policy(observation)
        if is_discrete:
            logits, probs = output
            action = jnp.argmax(probs, axis=-1)  # shape (1,)
            action = jnp.squeeze(action)  # scalar
        else:
            dist = output
            action = dist.mean()
            action = jnp.squeeze(action)  # flatten to 1D
        return action

    for ep in range(num_episodes):
        state, _ = env.reset()  # unpack tuple

        done = False
        steps = 0
        raw_rewards_list = []
        normalized_rewards_list = []

        while not done and steps < max_steps:
            s_t = normalization(state, config.state_mean, config.state_std)
            s_t = s_t[None, :]  # Add batch dimension
            action = select_action(s_t)
            if is_discrete:
                action = int(action)
            else:
                action = (action * config.ACTION_SCALE + config.ACTION_BIAS).astype(np.float32)

            next_state, reward_vector, done, info = env.step(action)

            raw_rewards = reward_vector  # use reward returned directly
            #done = terminated or truncated
            state = next_state

            raw_rewards_list.append(raw_rewards)
            if config.normalize_reward:
                normalized_rewards = min_max_normalization(raw_rewards, config.reward_min, config.reward_max)
            else:
                normalized_rewards = raw_rewards
            normalized_rewards_list.append(normalized_rewards)

            steps += 1

        traj_return = np.zeros_like(raw_rewards_list[0])
        for t, r in enumerate(raw_rewards_list):
            traj_return += (config.gamma ** t) * r
        raw_returns.append(traj_return)

        # Discounted sum for normalized rewards (optional)
        traj_norm_return = np.zeros_like(normalized_rewards_list[0])
        for t, r in enumerate(normalized_rewards_list):
            traj_norm_return += (config.gamma ** t) * r
        normalized_returns.append(traj_norm_return)

    final_metrics = compute_metrics(np.array(raw_returns))
    print(f"Final metrics: {final_metrics}")

    avg_raw_returns = np.mean(raw_returns, axis=0)
    avg_normalized_returns = np.mean(normalized_returns, axis=0)

    # Optional: save results
    #if save_dir is not None:
        #os.makedirs(save_dir, exist_ok=True)
        #if t_env is not None:
            #np.save(os.path.join(save_dir, f"raw_returns_step_{t_env}.npy"), raw_returns)
            #np.save(os.path.join(save_dir, f"normalized_returns_step_{t_env}.npy"), normalized_returns)
            #log_final_metrics(save_dir, config, final_metrics)

    print(f"Avg raw returns: {avg_raw_returns}")
    print(f"Avg normalized returns: {avg_normalized_returns}")

    return final_metrics

def compute_metrics(returns):
    """
    returns: (num_episodes, num_objectives)
    Returns averaged metrics across episodes.
    """
    # Sum rewards per episode → Utilitarian welfare
    utilitarian = np.sum(returns, axis=1)  # shape: (num_episodes,)

    # Jain’s fairness index per episode
    sum_R = np.sum(returns, axis=1)                  # Σ Ri
    sum_R2 = np.sum(returns**2, axis=1)             # Σ Ri^2
    n = returns.shape[1]                             # number of objectives
    jain = (sum_R**2) / (n * sum_R2 + 1e-8)         # avoid div by zero

    # Nash social welfare per episode
    # Add small epsilon to avoid log(0)
    returns_shifted = returns - returns.min(axis=1, keepdims=True) + 1e-8
    nsw = np.sum(np.log(returns_shifted), axis=1)

    # Average over episodes
    return {
        "utilitarian": np.mean(utilitarian),
        "jain": np.mean(jain),
        "nsw": np.mean(nsw)
    }


def log_final_metrics(save_dir, config, final_metrics):
    """Append final metrics for the current experiment to a CSV."""
    csv_path = os.path.join(save_dir, "experiment_results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "beta", "seed", "utilitarian", "jain", "nsw"])
        if not file_exists:
            writer.writeheader()  # write header if file is new
        writer.writerow({
            "alpha": getattr(config, "alpha", 0.0),
            "beta": getattr(config, "beta", 0.0),
            "seed": getattr(config, "seed", 0),
            **final_metrics
        })
