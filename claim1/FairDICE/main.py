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
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", type=str, default="limodice", help="Learner type")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--beta", type=float, default=0.001, help="beta hyperparameter")
    parser.add_argument("--divergence", type=str, default="SOFT_CHI", help="Divergence type (SOFT_CHI/CHI/KL)")
    parser.add_argument("--gradient_penalty_coeff", type=float, default=1e-4, help="Gradient penalty coefficient")
    parser.add_argument("--tanh_squash_distribution", type=bool, default=False, help="Use tanh-squash distribution for actions if set")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the network")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the policy")
    parser.add_argument("--layer_norm", type=bool, default=True, help="Use layer normalization if set")
    parser.add_argument("--nu_lr", type=float, default=3e-4, help="Nu learning rate")
    parser.add_argument("--policy_lr", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--mu_lr", type=float, default=3e-4, help="Mu learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--quality", type=str, choices=["expert", "amateur"], default="expert", help="Dataset quality")
    parser.add_argument("--preference_dist", type=str, choices=["uniform", "wide", "narrow"], default="uniform", help="Preference distribution")
    parser.add_argument("--max_seq_len", type=int, default=500, help="Max sequence length in trajectories")
    parser.add_argument("--normalize_reward", type=bool, default=False, help="Whether to normalize reward")
    parser.add_argument("--env_name", type=str, default="MO-Hopper-v2", help="Environment name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Running mode: 'train' or 'eval'")
    parser.add_argument("--load_path", type=str, default=None, help="Path to a saved model checkpoint (for eval mode).")
    parser.add_argument("--total_train_steps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--log_interval", type=int, default=1000, help="Log interval") 
    parser.add_argument("--eval_episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save the model checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")    
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
    
    args, unknown = parser.parse_known_args()
    config = SimpleNamespace(**vars(args))
    data_path = f"../data/{config.env_name}/{config.env_name}_50000_{config.quality}_{config.preference_dist}.pkl"

    with open(data_path, "rb") as f:
        trajs = pickle.load(f)
        print("Loaded trajectories from", data_path)
        
    env = gym.make(config.env_name)
    config.state_dim = env.observation_space.shape[0]
    # Handle continuous and discrete action spaces
    if hasattr(env.action_space, 'shape'):
        config.action_dim = env.action_space.shape[0]
    else:
        # Discrete action space
        config.action_dim = env.action_space.n
    config.reward_dim = env.obj_dim
    config.state_mean = state_norm_params[config.env_name]["mean"]
    config.state_std = np.sqrt(state_norm_params[config.env_name]["var"])
    config.ACTION_HIGH = env.action_space.high
    config.ACTION_LOW  = env.action_space.low
    config.ACTION_SCALE = (config.ACTION_HIGH - config.ACTION_LOW) / 2.0  
    config.ACTION_BIAS  = (config.ACTION_HIGH + config.ACTION_LOW) / 2.0 
    
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

    for traj in trajs:
        if config.normalize_reward:
            traj["rewards"] = min_max_normalization(traj["raw_rewards"], reward_min, reward_max)
        else:
            traj["rewards"] = traj["raw_rewards"]
        traj["states"] = normalization(traj['observations'], config.state_mean, config.state_std)
        traj['next_states'] = normalization(traj['next_observations'], config.state_mean, config.state_std)
        traj['actions'] = (traj['actions'] - config.ACTION_BIAS) / config.ACTION_SCALE
        traj["init_observations"] = np.tile(traj['observations'][0], (traj['observations'].shape[0], 1))
        traj["init_states"] = np.tile(traj['states'][0], (traj['states'].shape[0], 1))

    tmp = defaultdict(list)

    for traj in trajs:
        for key, value in traj.items():
            tmp[key].append(value)        

    batch = defaultdict(list)

    for key, values in tmp.items():
        batch[key] = np.concatenate(values, axis=0) 
        
    for key, value in batch.items():
        print(key, value.shape)
    
    config.hidden_dims = [config.hidden_dim] * config.num_layers


    time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    run_name = f"{time_stamp}_{config.learner}_{config.env_name}_{config.quality}_{config.preference_dist}_{config.divergence}_beta{config.beta}_seed{config.seed}"
    
    if config.learner == "FairDICE":
        from FairDICE import init_train_state, train_step, get_model, save_model, load_model
    else:
        raise ValueError("Invalid learner type.")
    
    
    save_dir = f"{config.save_path}/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir + "/eval")


    random.seed(config.seed); np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    train_state = init_train_state(config)
    train_carry = (train_state, key)
    buffer = Buffer(batch)
            
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
            config=config
            )   
    train_iterations = config.total_train_steps // config.log_interval
    for iter in tqdm(range(train_iterations), desc="Training ..."):
        step = (iter + 1) * config.log_interval  
        train_carry, update_info = jax.lax.scan(train_body, train_carry, length=config.log_interval)
        policy = get_model(train_carry[0].policy_state)[0]
        evaluate_policy(config, 
                    policy, 
                    env,
                    save_dir + "/eval",
                    num_episodes=config.eval_episodes, 
                    max_steps=config.max_seq_len,
                    t_env=step)        
        if config.wandb:
            for key, value in update_info.items():
                if "loss" in key or "grad" in key or "debug"   in key:
                    wandb.log({f"{key}": value[-1]}, step=step)
                else:
                    for i in range(config.reward_dim):
                        wandb.log({f"{key}_{i}": value[-1][i]}, step=step)

    if config.wandb:
        wandb.finish()
    if config.save_path:
        save_model(train_carry[0], os.path.abspath(save_dir + "/model"))

if __name__ == "__main__":
    main()
    
    
    



    

    


