import argparse
import json
import os


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute objective reach rates for a saved model.")
    parser.add_argument(
        "--run_dir",
        default="/home/scur0132/claim4/fact-ai/results/sweep_20260123_160808/seed_9_beta_0.001_h256_bs64",
        help="Path to run directory containing config.json and model/",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for action sampling.")
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device for JAX (cpu or gpu).",
    )
    return parser.parse_args()


args = _parse_args()
if args.device == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ.pop("JAX_PLATFORMS", None)

import numpy as np
import gym
import jax

import environments  # register env
from FairDICE import load_model, get_model
from utils import normalization


def main() -> None:
    run_dir = args.run_dir
    num_episodes = args.episodes
    seed_base = args.seed

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    for key in ("state_mean", "state_std", "reward_min", "reward_max"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = np.array(cfg[key], dtype=np.float32)

    class Cfg:
        pass

    config = Cfg()
    for k, v in cfg.items():
        setattr(config, k, v)
    config.is_discrete = True

    model_dir = os.path.join(run_dir, "model")
    train_state = load_model(os.path.abspath(model_dir), config)
    policy, _, _ = get_model(train_state.policy_state)
    policy.eval()

    env = gym.make(config.env_name)

    max_steps = config.max_seq_len
    counts = np.zeros(config.reward_dim, dtype=int)

    for ep in range(num_episodes):
        env.seed(ep)
        state = env.reset()
        done = False
        steps = 0
        last_nonzero = None
        while not done and steps < max_steps:
            s_t = normalization(state, config.state_mean, config.state_std).reshape(1, -1)
            dist = policy(s_t)
            key = jax.random.PRNGKey(seed_base + ep * 100000 + steps)
            action = int(jax.random.categorical(key, dist.logits[0]))
            next_state, _, done, info = env.step(action)
            raw_rewards = info["obj"]
            if np.any(raw_rewards != 0):
                last_nonzero = raw_rewards
            state = next_state
            steps += 1

        if last_nonzero is None:
            continue
        max_val = last_nonzero.max()
        if max_val <= 0:
            continue
        winners = np.where(last_nonzero == max_val)[0]
        if winners.size:
            counts[winners] += 1

    print("run_dir:", run_dir)
    print("num_episodes:", num_episodes)
    for i, c in enumerate(counts):
        print(f"objective_{i}: {c} ({c/num_episodes*100:.2f}%)")


if __name__ == "__main__":
    main()
