from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from utils import normalization, min_max_normalization
import os
import wandb

def evaluate_policy(config, policy, env, save_dir, num_episodes=3, max_steps=500, t_env=None, key=jax.random.PRNGKey(0)):
    policy.eval()
    raw_returns = []
    normalized_returns = []
    discounted_raw_returns = []          
    discounted_normalized_returns = []
    
    # Check if action space is discrete
    is_discrete = hasattr(config, 'is_discrete') and config.is_discrete
    
    @jax.jit
    def select_action(observation):
        output = policy(observation)
        if is_discrete:
            # For discrete policies: output is (logits, probs)
            logits, probs = output
            action = jnp.argmax(probs, axis=-1)  # deterministic action: greedy
        else:
            # For continuous policies: output is a distribution
            dist = output
            action = dist.mean()  # deterministic action
            action = action.flatten()
        return action

    for iter in range(num_episodes):
        env.seed(iter)
        state = env.reset()
        done = False
        steps = 0
        raw_rewards_list = []
        normalized_rewards_list = []
        discounted_raw_rewards_list = []
        discounted_normalized_rewards_list = []
        steps_list = []
        
        while not done and steps < max_steps:
            s_t = normalization(state, config.state_mean, config.state_std)
            action = select_action(s_t)
            
            if is_discrete:
                # For discrete actions, convert to int
                action = int(action)
            else:
                # For continuous actions, apply scaling and bias
                action = (action * config.ACTION_SCALE + config.ACTION_BIAS).astype(np.float32)
            
            state, _, done, info= env.step(action)
            


            raw_rewards = info['obj']
            raw_rewards_list.append(raw_rewards)
            discounted_raw_rewards = raw_rewards * (config.gamma ** steps)
            discounted_raw_rewards_list.append(discounted_raw_rewards)
            if config.normalize_reward:
                normalized_rewards = min_max_normalization(raw_rewards, config.reward_min, config.reward_max)
            else:
                normalized_rewards = raw_rewards
            normalized_rewards_list.append(normalized_rewards)
            discounted_normalized_rewards = normalized_rewards * (config.gamma ** steps)
            discounted_normalized_rewards_list.append(discounted_normalized_rewards)
            
            steps += 1
    
        steps_list.append(steps)
        raw_returns.append(np.sum(raw_rewards_list, axis=0))
        normalized_returns.append(np.sum(normalized_rewards_list, axis=0))
        discounted_raw_returns.append(np.sum(discounted_raw_rewards_list, axis=0))
        discounted_normalized_returns.append(np.sum(discounted_normalized_rewards_list, axis=0))

    avg_raw_returns = np.mean(raw_returns, axis=0)
    avg_normalized_returns = np.mean(normalized_returns, axis=0)
    avg_discounted_raw_returns = np.mean(discounted_raw_returns, axis=0)
    avg_discounted_normalized_returns = np.mean(discounted_normalized_returns, axis=0)
    avg_steps = np.mean(steps_list)
    avg_raw_nsw_score = np.mean(np.sum(np.log(raw_returns), axis=1))
    avg_normalized_nsw_score = np.mean(np.sum(np.log(normalized_returns), axis=1))
    avg_discounted_raw_nsw_score = np.mean(np.sum(np.log(discounted_raw_returns), axis=1))
    avg_discounted_normalized_nsw_score = np.mean(np.sum(np.log(discounted_normalized_returns), axis=1))
    avg_raw_usw_score = np.mean(np.sum(raw_returns, axis=1))
    avg_normalized_usw_score = np.mean(np.sum(normalized_returns, axis=1))
    avg_raw_discounted_usw_score = np.mean(np.sum(discounted_raw_returns, axis=1))
    avg_normalized_discounted_usw_score = np.mean(np.sum(discounted_normalized_returns, axis=1))
    
    if t_env is not None:
        if t_env == config.total_train_steps:
            np.save(os.path.join(save_dir, f"raw_returns_step_{t_env}.npy"), raw_returns)
            np.save(os.path.join(save_dir, f"normalized_returns_step_{t_env}.npy"), normalized_returns)
            np.save(os.path.join(save_dir, f"steps_step_{t_env}.npy"), steps_list)

        if config.wandb:
            for i in range(config.reward_dim):
                wandb.log({
                    f"eval/avg_raw_return_{i}": avg_raw_returns[i],
                    f"eval/avg_normalized_return_{i}": avg_normalized_returns[i],
                    f"eval/avg_discounted_raw_return_{i}": avg_discounted_raw_returns[i],
                    f"eval/avg_discounted_normalized_return_{i}": avg_discounted_normalized_returns[i],
                }, step=t_env)
            wandb.log({
                "eval/avg_steps": avg_steps,
                "eval/avg_normalized_nsw_score": avg_normalized_nsw_score,
                "eval/avg_normalized_usw_score": avg_normalized_usw_score,
                "eval/avg_raw_discounted_nsw_score": avg_discounted_raw_nsw_score,
                "eval/avg_raw_discounted_usw_score": avg_raw_discounted_usw_score,
                "eval/avg_normalized_discounted_nsw_score": avg_discounted_normalized_nsw_score,
                "eval/avg_normalized_discounted_usw_score": avg_normalized_discounted_usw_score,
                "eval/avg_raw_nsw_score": avg_raw_nsw_score,
                "eval/avg_raw_usw_score": avg_raw_usw_score,
            }, step=t_env)
        else:
            print(f"Avg raw returns: {avg_raw_returns}")
            print(f"Avg normalized returns: {avg_normalized_returns}")
            print(f"Avg discounted raw returns: {avg_discounted_raw_returns}")
            print(f"Avg discounted normalized returns: {avg_discounted_normalized_returns}")
            print(f"Avg steps: {avg_steps}")
            print(f"Avg raw NSW score: {avg_raw_nsw_score}")
            print(f"Avg normalized NSW score: {avg_normalized_nsw_score}")
            print(f"Avg discounted raw NSW score: {avg_discounted_raw_nsw_score}")
            print(f"Avg discounted normalized NSW score: {avg_discounted_normalized_nsw_score}")

    return raw_returns, normalized_returns