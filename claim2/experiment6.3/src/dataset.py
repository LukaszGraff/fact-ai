import numpy as np


def value_iteration_opt_policy(env, r_scalar, tol=1e-10, max_iter=200000):
    S, A, g = env.n_states, env.n_actions, env.gamma
    V = np.zeros(S, dtype=np.float64)

    # precompute P(s'|s,a) in dense-ish form
    P = np.zeros((S, A, S), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            ns = env.next_state_support[s, a]
            p = env.next_state_prob[s, a]
            P[s, a, ns] = p

    print("[dataset.value_iteration_opt_policy] Starting value iteration")
    converged = False
    for it in range(max_iter):
        Q = r_scalar + g * (P @ V)  # (S,A)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            converged = True
            print(
                f"[dataset.value_iteration_opt_policy] Converged after {it+1} iterations"
            )
            break
        V = V_new
    if not converged:
        print("[dataset.value_iteration_opt_policy] Reached max iterations without full convergence")

    Q = r_scalar + g * (P @ V)
    a_star = np.argmax(Q, axis=1)
    pi_star = np.zeros((S, A), dtype=np.float64)
    pi_star[np.arange(S), a_star] = 1.0
    print("[dataset.value_iteration_opt_policy] Derived greedy policy")
    return pi_star

def eval_policy_return_exact(env, pi):
    S, A, g = env.n_states, env.n_actions, env.gamma

    # Dense transition P[s,a,s']
    P = np.zeros((S, A, S), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            ns = env.next_state_support[s, a]
            p = env.next_state_prob[s, a]
            P[s, a, ns] = p

    # reward is on next state in your env: r_sum(s') = sum_k reward_on_state[s',k]
    r_sum_state = np.sum(env.reward_on_state, axis=1)  # (S,)

    # Expected scalar reward for each (s,a): E[r_sum(next_state) | s,a]
    r_sa = np.einsum("iaj,j->ia", P, r_sum_state)  # (S,A)

    # Policy-induced transition and reward
    P_pi = np.einsum("ia,iaj->ij", pi, P)          # (S,S)
    r_pi = np.einsum("ia,ia->i", pi, r_sa)         # (S,)

    # Solve (I - gamma P_pi) V = r_pi
    V = np.linalg.solve(np.eye(S) - g * P_pi, r_pi)

    return float(V[env.init_state])

def make_behavior_policy_half_optimal(env):
    S, A = env.n_states, env.n_actions
    print("[dataset.make_behavior_policy_half_optimal] Constructing behavior policy")

    # scalar reward = sum of objectives, per-state reward
    r_scalar = np.zeros((S, A), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            # expected reward after taking (s,a) and transitioning
            ns = env.next_state_support[s, a]
            p = env.next_state_prob[s, a]
            exp_r = 0.0
            for k in range(4):
                exp_r += p[k] * np.sum(env.reward_on_state[ns[k]])
            r_scalar[s, a] = exp_r

    pi_unif = np.ones((S, A), dtype=np.float64) / A
    pi_star = value_iteration_opt_policy(env, r_scalar)

    J_unif = eval_policy_return_exact(env, pi_unif)
    J_opt = eval_policy_return_exact(env, pi_star)
    J_target = 0.5 * (J_unif + J_opt)

    def pi_mix(p):
        return (1 - p) * pi_unif + p * pi_star

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        J_mid = eval_policy_return_exact(env, pi_mix(mid))
        if J_mid < J_target:
            lo = mid
        else:
            hi = mid
    p_behavior = 0.5 * (lo + hi)
    print(
        "[dataset.make_behavior_policy_half_optimal] Finished with stats: "
        f"J_unif={J_unif:.4f}, J_opt={J_opt:.4f}, target={J_target:.4f}, p={p_behavior:.3f}"
    )
    return pi_mix(p_behavior), dict(J_unif=J_unif, J_opt=J_opt, J_target=J_target, p_behavior=p_behavior)

def collect_offline_dataset(env, pi_b, n_traj=100, seed=10000):
    rng = np.random.default_rng(seed)
    old_rng = env.rng
    env.rng = rng

    states, actions, rewards, next_states, dones, ep_id, t_id = [], [], [], [], [], [], []
    print(
        f"[dataset.collect_offline_dataset] Collecting {n_traj} trajectories (seed={seed})"
    )
    progress_step = max(1, n_traj // 5)
    for ep in range(n_traj):
        s = env.reset()
        for t in range(env.horizon):
            a = rng.choice(env.n_actions, p=pi_b[s])
            ns, r_vec, done = env.step(s, a)
            states.append(s)
            actions.append(a)
            rewards.append(r_vec)
            next_states.append(ns)
            dones.append(done)
            ep_id.append(ep)
            t_id.append(t)
            s = ns
        if (ep + 1) % progress_step == 0 or ep == n_traj - 1:
            print(
                "[dataset.collect_offline_dataset] Completed "
                f"{ep + 1}/{n_traj} trajectories"
            )

    env.rng = old_rng
    print("[dataset.collect_offline_dataset] Dataset assembly finished")
    return {
        "states": np.array(states, dtype=np.int64),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float64),
        "next_states": np.array(next_states, dtype=np.int64),
        "dones": np.array(dones, dtype=np.bool_),
        "episode_id": np.array(ep_id, dtype=np.int64),
        "timestep": np.array(t_id, dtype=np.int64),
    }
