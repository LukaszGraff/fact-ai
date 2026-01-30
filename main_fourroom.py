import os
import gym
import environments
from environments.norm import state_norm_params
import pickle
import numpy as np
from collections import defaultdict
from utils import normalization, min_max_normalization
from buffer import Buffer
import argparse
from types import SimpleNamespace
from evaluation import evaluate_policy
import random
from tqdm import tqdm
import wandb
import pandas as pd
import jax
import sys
import shutil
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner",
        type=str,
        default="FairDICE",
        choices=["FairDICE"],
        help="Learner type",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, default=0.001, help="beta hyperparameter")
    parser.add_argument("--divergence", type=str, default="SOFT_CHI", help="Divergence type (SOFT_CHI/CHI/KL)")
    parser.add_argument("--gradient_penalty_coeff", type=float, default=1e-4, help="Gradient penalty coefficient")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the network")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the policy")
    parser.add_argument("--layer_norm", type=bool, default=True, help="Use layer normalization if set")
    parser.add_argument("--nu_lr", type=float, default=3e-4, help="Nu learning rate")
    parser.add_argument("--policy_lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--quality", type=str, choices=["expert", "amateur"], default="expert", help="Dataset quality")
    parser.add_argument(
        "--preference_dist",
        type=str,
        default="uniform",
        help="Dataset tag used in the offline data filename (e.g., uniform, wide, narrow, or custom).",
    )
    parser.add_argument("--max_seq_len", type=int, default=200, help="Max sequence length in trajectories")
    parser.add_argument("--normalize_reward", type=bool, default=False, help="Whether to normalize reward")
    parser.add_argument("--env_name", type=str, default="MO-FourRoom-v2", help="Environment name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Running mode: 'train' or 'eval'")
    parser.add_argument("--load_path", type=str, default=None, help="Path to a saved model checkpoint (for eval mode).")
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval") 
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save the model checkpoint")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the run directory name under save_path (useful for sweeps). If not set, a timestamped name is used.",
    )
    parser.add_argument(
        "--save_model_mode",
        type=str,
        choices=["best_nsw", "last"],
        default="last",
        help="Model saving mode: 'best_nsw' keeps best checkpoint by FourRoom NSW-of-mean during training; 'last' saves only final model.",
    )
    # CQL baseline args removed.
    parser.add_argument("--seed", type=int, default=0, help="Random seed")    
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter to control the optimiazation objective. alpha=0: Utilitarian Welfare; alpha=1: Nash Social Welfare")
    
    args, unknown = parser.parse_known_args()
    config = SimpleNamespace(**vars(args))
    
    # Load data
    data_root = os.environ.get("DATA_ROOT", ".")
    data_path = os.path.join(
        data_root,
        "data",
        config.env_name,
        f"{config.env_name}_50000_{config.quality}_{config.preference_dist}.pkl",
    )
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Generating data...")
        from generate_fourroom_data import generate_offline_data
        generate_offline_data(config.env_name, num_trajectories=300, quality=config.quality, preference_dist=config.preference_dist)
    
    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
        print(f"Loaded {len(trajs)} trajectories from {data_path}")
    
    # Create environment
    env = gym.make(config.env_name)
    config.state_dim = env.observation_space.shape[0]
    config.action_dim = env.action_space.n  # Discrete action space
    config.reward_dim = env.num_objectives
    
    # Flag for discrete action handling
    config.is_discrete = True
    
    # For continuous policies (not used here but needed for compatibility)
    config.tanh_squash_distribution = False
    
    # For discrete environments, we don't use state normalization from norm.py
    # Use simple z-score normalization based on data
    all_states = np.concatenate([traj['observations'] for traj in trajs], axis=0)
    config.state_mean = all_states.mean(axis=0)
    config.state_std = all_states.std(axis=0) + 1e-8
    
    # For discrete actions, no action scaling needed
    config.ACTION_SCALE = 1.0
    config.ACTION_BIAS = 0.0
    
    # Compute reward statistics
    reward_min, reward_max = None, None        
    for traj in trajs:
        r = traj["raw_rewards"]           
        r_min = r.min(axis=0)                  
        r_max = r.max(axis=0)

        if reward_min is None:                 
            reward_min, reward_max = r_min, r_max
        else:                                  
            reward_min = np.minimum(reward_min, r_min)
            reward_max = np.maximum(reward_max, r_max)
    config.reward_min = reward_min
    config.reward_max = reward_max

    # Process trajectories
    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(traj["raw_rewards"], reward_min, reward_max)
        else:
            traj["rewards"] = traj["raw_rewards"]
        traj["states"] = normalization(traj['observations'], config.state_mean, config.state_std)
        traj['next_states'] = normalization(traj['next_observations'], config.state_mean, config.state_std)
        # For discrete actions, keep them as-is (no scaling)
        traj['actions'] = traj['actions'].astype(np.float32)
        traj["init_observations"] = np.tile(traj['observations'][0], (traj['observations'].shape[0], 1))
        traj["init_states"] = np.tile(traj['states'][0], (traj['states'].shape[0], 1))
        # Use terminals from data if available, otherwise mark only last step as terminal
        if "terminals" not in traj:
            traj["terminals"] = np.zeros(len(traj['observations']))
            traj["terminals"][-1] = 1.0

    # Concatenate all trajectories into batch
    tmp = defaultdict(list)
    for traj in trajs:
        for key, value in traj.items():
            tmp[key].append(value)        

    batch = defaultdict(list)
    for key, values in tmp.items():
        batch[key] = np.concatenate(values, axis=0) 
        
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
    
    config.hidden_dims = [config.hidden_dim] * config.num_layers

    if config.run_name is not None:
        run_name = str(config.run_name)
    else:
        time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")
        run_name = f"{time_stamp}_{config.learner}_{config.env_name}_{config.quality}_{config.preference_dist}_{config.divergence}_beta{config.beta}_seed{config.seed}"
    
    if config.learner == "FairDICE":
        from FairDICE import init_train_state, train_step, get_model, save_model, load_model
    else:
        raise ValueError("Invalid learner type.")
    
    save_dir = f"{config.save_path}/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, "eval"))

    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    train_state = init_train_state(config)
    train_carry = (train_state, key)
    buffer = Buffer(batch)
    
    # Save best model during training based on NSW (computed as NSW-of-mean returns for FourRoom)
    best_nsw_score = float('-inf') if config.save_model_mode == "best_nsw" else None
    best_step = None
            
    def train_body(carry, _):
        train_state, key = carry
        key, subkey = jax.random.split(key)
        data = buffer.sample(subkey, config.batch_size)
        train_state, update_info = train_step(config, train_state, data, subkey)
        return (train_state, key), (update_info)
    
    if config.wandb:
        wandb.init(
            project=f"exp_{config.tag}",
            name=run_name,
            config=vars(config)
        )
    
    train_iterations = config.total_train_steps // config.log_interval
    for iter in tqdm(range(train_iterations), desc="Training ..."):
        step = (iter + 1) * config.log_interval  
        train_carry, update_info = jax.lax.scan(train_body, train_carry, length=config.log_interval)
        policy = get_model(train_carry[0].policy_state)[0]
        
        # Evaluation
        raw_returns, normalized_returns = evaluate_policy(
            config, 
            policy, 
            env,
            os.path.join(save_dir, "eval"),
            num_episodes=config.eval_episodes, 
            max_steps=config.max_seq_len,
            t_env=step
        )

        # Show policy loss during training (survives tqdm + Slurm stdout buffering).
        policy_loss_value = float(np.asarray(update_info["policy_loss"][-1]))
        log_parts = [f"policy_loss={policy_loss_value:.6f}"]
        if "nu_loss" in update_info:
            nu_loss_value = float(np.asarray(update_info["nu_loss"][-1]))
            log_parts.append(f"nu_loss={nu_loss_value:.6f}")
        if "mu" in update_info:
            mu_value = np.asarray(update_info["mu"][-1]).reshape(-1)
            mu_str = np.array2string(mu_value, precision=4, separator=", ")
            log_parts.append(f"mu={mu_str}")
        tqdm.write(f"Step {step}: " + " | ".join(log_parts))
        sys.stdout.flush()

        if config.save_model_mode == "best_nsw":
            # NSW for FourRoom: NSW over the average return vector across episodes (NSW-of-mean)
            eps = 0.001
            returns_for_nsw = normalized_returns if config.normalize_reward else raw_returns
            avg_returns = np.mean(returns_for_nsw, axis=0)
            nsw_score = float(np.sum(np.log(np.asarray(avg_returns) + eps)))

            # Save best model checkpoint during training
            if best_nsw_score is None or nsw_score > best_nsw_score:
                best_nsw_score = nsw_score
                best_step = step

                model_dir = os.path.abspath(os.path.join(save_dir, "model"))
                tmp_dir = os.path.abspath(os.path.join(save_dir, "model_tmp"))
                prev_dir = os.path.abspath(os.path.join(save_dir, "model_prev"))

                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
                save_model(train_carry[0], tmp_dir)

                if os.path.exists(prev_dir):
                    shutil.rmtree(prev_dir)
                if os.path.exists(model_dir):
                    os.rename(model_dir, prev_dir)
                os.rename(tmp_dir, model_dir)
                if os.path.exists(prev_dir):
                    shutil.rmtree(prev_dir)

                print(
                    f"New best model at step {step} with NSW(score_of_mean_returns)={best_nsw_score:.6f} (avg_returns={avg_returns})"
                )
        
        if config.wandb:
            for key, value in update_info.items():
                if "loss" in key or "grad" in key or "debug" in key:
                    wandb.log({f"{key}": value[-1]}, step=step)
                else:
                    for i in range(config.reward_dim):
                        wandb.log({f"{key}_{i}": value[-1][i]}, step=step)

    if config.wandb:
        wandb.finish()
    
    if config.save_path:
        # Save config as JSON for later use
        import json

        def _json_safe(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
            return value

        config_dict = {k: _json_safe(v) for k, v in vars(config).items()}
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        model_dir = os.path.abspath(os.path.join(save_dir, "model"))

        if config.save_model_mode == "last":
            # Save only the final model (write to temp dir then swap into place).
            tmp_dir = os.path.abspath(os.path.join(save_dir, "model_tmp"))
            prev_dir = os.path.abspath(os.path.join(save_dir, "model_prev"))
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            save_model(train_carry[0], tmp_dir)

            if os.path.exists(prev_dir):
                shutil.rmtree(prev_dir)
            if os.path.exists(model_dir):
                os.rename(model_dir, prev_dir)
            os.rename(tmp_dir, model_dir)
            if os.path.exists(prev_dir):
                shutil.rmtree(prev_dir)
        else:
            # best_nsw mode: model_dir is updated during training; if no save happened, save last.
            if not os.path.exists(model_dir):
                save_model(train_carry[0], model_dir)
            elif best_step is not None and best_nsw_score is not None:
                print(f"Final saved model is best from step {best_step} (NSW={best_nsw_score:.6f}).")
    
    print(f"Training complete. Results saved to {save_dir}")

if __name__ == "__main__":
    main()
