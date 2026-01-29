import environments
import gym
import numpy as np
import pickle
from pathlib import Path
from collections import deque
import numpy as np

def bfs_optimal_policy(env):
    num_states, num_actions = env.num_states, env.num_actions
    optimal_policy = np.zeros((num_states, num_actions), dtype=np.float32)

    # Precompute transitions: for each state & action, get reachable next states
    transitions = {}
    for s in range(num_states):
        for a in range(num_actions):
            next_states = np.where(env.P[s, a] > 0)[0]
            transitions[(s, a)] = next_states

    for s in range(num_states):
        print(f"State {s}: reachable states per action:", transitions)

        visited = [False] * num_states
        queue = deque()
        queue.append((s, []))  # state, path (list of actions)
        visited[s] = True
        found_goal = False

        while queue and not found_goal:
            current_state, path = queue.popleft()

            if current_state in env.goal_states:
                # Take first action on path as optimal
                if path:
                    first_action = path[0]
                    optimal_policy[s, first_action] = 1.0
                found_goal = True
                break

            for a in range(num_actions):
                for ns in transitions[(current_state, a)]:
                    if not visited[ns]:
                        visited[ns] = True
                        queue.append((ns, path + [a]))

        # If no path found (should not happen), pick random action
        if not found_goal:
            optimal_policy[s, np.random.randint(num_actions)] = 1.0

    return optimal_policy

def make_behavior_policy(optimal_policy, epsilon=0.5):
    num_states, num_actions = optimal_policy.shape
    behavior_policy = np.zeros_like(optimal_policy)
    for s in range(num_states):
        # mix optimal + uniform random
        behavior_policy[s] = epsilon * optimal_policy[s] + (1 - epsilon) / num_actions
    return behavior_policy




def generate_random_momdp_trajs(
        env_name: str = "RandomMOMDP-v1",
        num_trajectories: int = 100,
        horizon: int = 100,
        seed: int = 1,
        quality: str = "expert",
        preference_dist: str = "uniform",
        total_steps_tag: int = 50000,
):
    print("Creating environment...")
    env = gym.make(env_name, seed=seed)

    # 🔥 FIX: Remove ALL gym wrappers (TimeLimit, etc.)
    while hasattr(env, 'env'):
        print("  Removing wrapper:", type(env).__name__)
        env = env.env

    print("Goal states:", env.goal_states)

    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]  # 50
    action_dim = env.action_space.n  # 4

    reward_dim = env.obj_dim

    print(f"Dims: state={state_dim}, action={action_dim}, reward={reward_dim}")

    trajectories = []
    print("Generating trajectories...")

    # Step 1: Compute optimal policy
    optimal_policy = bfs_optimal_policy(env)

    # Step 2: Mix to 0.5-optimal
    behavior_policy = make_behavior_policy(optimal_policy, epsilon=0.5)

    for ep in range(num_trajectories):
        obs, _ = env.reset(seed=seed + ep)
        obs_list = []
        next_obs_list = []
        act_list = []
        rew_list = []
        done_list = []
        mask_list = []

        for t in range(horizon):
            state_idx = int(np.argmax(obs))  # get current state from one-hot
            action = np.random.choice(env.action_space.n, p=behavior_policy[state_idx])
            # Handle ANY step return format
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward_vec, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                next_obs, reward_vec, done, info = step_result
                terminated = done
                truncated = False
            elif len(step_result) == 3:
                next_obs, reward_vec, done = step_result
                terminated = done
                truncated = False
            else:
                print(f"WARNING: Unexpected step format: {len(step_result)}")
                next_obs, reward_vec, terminated = step_result[0], step_result[1], step_result[2]
                truncated = False

            obs_list.append(obs.astype(np.float32))
            next_obs_list.append(next_obs.astype(np.float32))
            act_list.append(np.array([float(action)], dtype=np.float32))
            rew_list.append(np.array(reward_vec, dtype=np.float32))
            done_list.append(terminated or truncated)
            mask_list.append(0.0 if (terminated or truncated) else 1.0)

            obs = next_obs
            if terminated or truncated:
                break

        traj = {
            "observations": np.stack(obs_list),
            "next_observations": np.stack(next_obs_list),
            "actions": np.stack(act_list),
            "raw_rewards": np.stack(rew_list),
            "dones": np.array(done_list, dtype=bool),
            "masks": np.array(mask_list, dtype=np.float32),
        }

        trajectories.append(traj)

    data_dir = Path(f"./data_terminate/{env_name}")
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{env_name}_{total_steps_tag}_{quality}_{preference_dist}_{seed}.pkl"
    save_path = data_dir / filename

    with open(save_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"✅ Saved {len(trajectories)} trajectories to {save_path}")
    print(f"Total steps: {sum(len(t['observations']) for t in trajectories)}")


if __name__ == "__main__":
    generate_random_momdp_trajs()
