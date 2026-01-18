from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from utils import normalization, min_max_normalization
import os
import wandb

def _reset_env(env, seed):
    reset_out = env.reset(seed=seed) if "seed" in getattr(env.reset, "__code__", (None,)).co_varnames else env.reset()
    if isinstance(reset_out, tuple):
        obs, _ = reset_out
    else:
        obs = reset_out
    return obs


def _step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
        return obs, reward, done, info
    return step_out


def evaluate_policy(
    config,
    policy,
    env,
    save_dir,
    num_episodes=3,
    max_steps=500,
    t_env=None,
    key=jax.random.PRNGKey(0),
    eval_mode="greedy",
    eval_seed=None,
):
    policy.eval()
    raw_returns = []
    normalized_returns = []
    discounted_raw_returns = []          
    discounted_normalized_returns = []
    hit_any_goal_count = 0
    per_goal_hits = np.zeros(3, dtype=np.int32)
    termination_steps = []
    
    # Check if action space is discrete
    is_discrete = hasattr(config, 'is_discrete') and config.is_discrete
    
    @jax.jit
    def select_action(observation, rng):
        output = policy(observation)
        if is_discrete:
            # For discrete policies: output is (logits, probs)
            logits, _ = output
            if eval_mode == "greedy":
                action = jnp.argmax(logits, axis=-1)
                action = jnp.reshape(action, ())
            else:
                action = jax.random.categorical(rng, logits, axis=-1)
        else:
            # For continuous policies: output is a distribution
            dist = output
            action = dist.mean()  # deterministic action
            action = action.flatten()
        return action

    steps_list = []
    trace_enabled = t_env is not None and not getattr(config, "_eval_trace_done", False)
    action_mode = "GREEDY" if eval_mode == "greedy" else "SAMPLE"
    trace_steps = []
    trace_done = None
    for iter in range(num_episodes):
        try:
            env.seed(iter)
            state = env.reset()
            if isinstance(state, tuple):
                state, _ = state
        except Exception:
            state = _reset_env(env, iter)
        done = False
        steps = 0
        raw_rewards_list = []
        normalized_rewards_list = []
        discounted_raw_rewards_list = []
        discounted_normalized_rewards_list = []
        
        terminated = False
        last_step_reward = None
        while not done and steps < max_steps:
            s_t = normalization(state, config.state_mean, config.state_std)
            key, subkey = jax.random.split(key)
            action = select_action(s_t, subkey)
            
            if is_discrete:
                # For discrete actions, convert to int
                action = int(action)
                if getattr(config, "debug_eval", False):
                    action_dim = getattr(config, "action_dim", 4)
                    print(
                        f"[evaluate_policy] action={action} valid={0 <= action < action_dim}"
                    )
            else:
                # For continuous actions, apply scaling and bias
                action = (action * config.ACTION_SCALE + config.ACTION_BIAS).astype(np.float32)
            
            pre_state = state
            step_out = env.step(action)
            if len(step_out) == 5:
                state, _, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                state, _, done, info = step_out
            


            raw_rewards = info['obj']
            last_step_reward = raw_rewards
            raw_rewards_list.append(raw_rewards)
            if trace_enabled and iter == 0 and steps < 30:
                trace_steps.append(
                    {
                        "t": steps,
                        "state": pre_state,
                        "action": action,
                        "mode": action_mode,
                        "next_state": state,
                        "reward": raw_rewards,
                        "terminated": terminated,
                        "done": done,
                    }
                )
                trace_done = done
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
    
        if terminated:
            hit_any_goal_count += 1
            idx = int(np.argmax(last_step_reward)) if last_step_reward is not None else 0
            per_goal_hits[idx] += 1
            termination_steps.append(steps)

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
    goal_hit_rate = hit_any_goal_count / max(num_episodes, 1)
    per_goal_hit_rate = per_goal_hits / max(num_episodes, 1)
    avg_termination_step = float(np.mean(termination_steps)) if termination_steps else 0.0
    eps = 1e-8
    avg_raw_nsw_score = float(np.sum(np.log(np.clip(avg_raw_returns, eps, None))))
    avg_normalized_nsw_score = float(np.sum(np.log(np.clip(avg_normalized_returns, eps, None))))
    avg_discounted_raw_nsw_score = float(np.sum(np.log(np.clip(avg_discounted_raw_returns, eps, None))))
    avg_discounted_normalized_nsw_score = float(
        np.sum(np.log(np.clip(avg_discounted_normalized_returns, eps, None)))
    )
    avg_raw_usw_score = np.mean(np.sum(raw_returns, axis=1))
    avg_normalized_usw_score = np.mean(np.sum(normalized_returns, axis=1))
    avg_raw_discounted_usw_score = np.mean(np.sum(discounted_raw_returns, axis=1))
    avg_normalized_discounted_usw_score = np.mean(np.sum(discounted_normalized_returns, axis=1))
    
    metrics = {
        "avg_raw_returns": avg_raw_returns,
        "avg_normalized_returns": avg_normalized_returns,
        "avg_discounted_raw_returns": avg_discounted_raw_returns,
        "avg_discounted_normalized_returns": avg_discounted_normalized_returns,
        "avg_steps": avg_steps,
        "avg_raw_nsw_score": avg_raw_nsw_score,
        "avg_normalized_nsw_score": avg_normalized_nsw_score,
        "avg_discounted_raw_nsw_score": avg_discounted_raw_nsw_score,
        "avg_discounted_normalized_nsw_score": avg_discounted_normalized_nsw_score,
        "avg_raw_usw_score": avg_raw_usw_score,
        "avg_normalized_usw_score": avg_normalized_usw_score,
        "avg_raw_discounted_usw_score": avg_raw_discounted_usw_score,
        "avg_normalized_discounted_usw_score": avg_normalized_discounted_usw_score,
        "goal_hit_rate": goal_hit_rate,
        "per_goal_hit_rate": per_goal_hit_rate,
        "avg_termination_step": avg_termination_step,
    }

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
            print(f"Eval episodes: {num_episodes}")
            print(f"Avg raw returns: {avg_raw_returns}")
            print(f"Avg normalized returns: {avg_normalized_returns}")
            print(f"Avg discounted raw returns: {avg_discounted_raw_returns}")
            print(f"Avg discounted normalized returns: {avg_discounted_normalized_returns}")
            print(f"Avg steps: {avg_steps}")
            print(
                "Goal hit rate: "
                f"{goal_hit_rate:.2f} | Per-goal: "
                f"{per_goal_hit_rate.tolist()} | Terminated avg step: "
                f"{avg_termination_step:.1f}"
            )
            print(f"Avg raw NSW score: {avg_raw_nsw_score}")
            print(f"Avg normalized NSW score: {avg_normalized_nsw_score}")
            print(f"Avg discounted raw NSW score: {avg_discounted_raw_nsw_score}")
            print(f"Avg discounted normalized NSW score: {avg_discounted_normalized_nsw_score}")
            if trace_enabled and trace_steps:
                print("[eval trace] first episode (first 30 steps)")
                for entry in trace_steps:
                    print(
                        f"t={entry['t']} state={entry['state']} "
                        f"action={entry['action']} next_state={entry['next_state']} "
                        f"mode={entry['mode']} reward={entry['reward']} "
                        f"terminated={entry['terminated']} done={entry['done']}"
                    )
                setattr(config, "_eval_trace_done", True)

    return raw_returns, normalized_returns, metrics
    if eval_seed is not None:
        key = jax.random.PRNGKey(int(eval_seed))
