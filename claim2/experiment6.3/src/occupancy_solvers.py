import numpy as np
import cvxpy as cp


def empirical_d_dataset(dataset, n_states, n_actions, gamma):
    # discounted occupancy estimate from trajectories
    # weight each transition by (1-gamma)*gamma^t so it sums ~1
    print("[occupancy.empirical_d_dataset] Estimating discounted occupancy")
    dD = np.zeros((n_states, n_actions), dtype=np.float64)
    for s, a, t in zip(dataset["states"], dataset["actions"], dataset["timestep"]):
        dD[s, a] += (1.0 - gamma) * (gamma ** int(t))
    dD /= max(dD.sum(), 1e-12)
    # smooth to avoid divide-by-zero
    dD = np.maximum(dD, 1e-12)
    dD /= dD.sum()
    print("[occupancy.empirical_d_dataset] Completed normalization")
    return dD

def build_transition_tensor(env):
    S, A = env.n_states, env.n_actions
    print("[occupancy.build_transition_tensor] Building dense transition tensor")
    P = np.zeros((S, A, S), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            ns = env.next_state_support[s, a]
            p = env.next_state_prob[s, a]
            P[s, a, ns] = p
    print("[occupancy.build_transition_tensor] Tensor ready")
    return P

def expected_obj_rewards(env, P):
    # expected immediate reward vector for each (s,a): E[r(s') | s,a]
    S, A, K = env.n_states, env.n_actions, env.n_obj
    print("[occupancy.expected_obj_rewards] Computing expected rewards")
    Rsa = np.zeros((S, A, K), dtype=np.float64)
    for s in range(S):
        for a in range(A):
            for sp in range(S):
                if P[s, a, sp] > 0:
                    Rsa[s, a] += P[s, a, sp] * env.reward_on_state[sp]
    print("[occupancy.expected_obj_rewards] Finished expected reward tensor")
    return Rsa

def solve_fairdice_fixed(env, dD, beta, mu, P=None, Rsa=None):
    """
    Solve: max_d  mu^T R(d) - beta * chi2(d || dD)
    s.t. discounted occupancy flow constraints
    """
    S, A, g = env.n_states, env.n_actions, env.gamma
    if P is None:
        P = build_transition_tensor(env)
    if Rsa is None:
        Rsa = expected_obj_rewards(env, P)  # (S,A,K)

    print(
        "[occupancy.solve_fairdice_fixed] Solving FairDICE-fixed with "
        f"beta={beta}, mu={np.round(mu, 4)}"
    )

    d = cp.Variable((S, A), nonneg=True)

    # flow constraints
    rho = cp.sum(d, axis=1)  # (S,)
    init = np.zeros(S); init[env.init_state] = 1.0
    inflow = cp.sum(cp.sum(cp.multiply(d[:, :, None], P), axis=0), axis=0)  # (S,)
    constraints = [
        rho == (1 - g) * init + g * inflow,
        cp.sum(d) == 1.0
    ]

    # returns per objective
    R = cp.sum(cp.sum(cp.multiply(d[:, :, None], Rsa), axis=0), axis=0)  # (K,)

    # chi2 divergence term: sum dD * 0.5 * ((d/dD)-1)^2
    ratio = cp.multiply(d, 1.0 / dD)
    chi2 = 0.5 * cp.sum(cp.multiply(dD, cp.square(ratio - 1.0)))

    obj = cp.Maximize(mu @ R - beta * chi2)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    status = prob.status
    print(
        "[occupancy.solve_fairdice_fixed] Solver status="
        f"{status}, obj={prob.value:.6f}"
    )

    if status not in {"optimal", "optimal_inaccurate"}:
        print(
            "[occupancy.solve_fairdice_fixed] Warning: solver did not converge;"
            " skipping datapoint"
        )
        return None, None

    d_star = np.array(d.value, dtype=np.float64)
    R_star = np.array([np.sum(d_star * Rsa[:, :, k]) for k in range(env.n_obj)], dtype=np.float64)
    print("[occupancy.solve_fairdice_fixed] Extraction complete")
    return d_star, R_star

def solve_fairdice_nsw(env, dD, beta, eps=1e-8, P=None, Rsa=None):
    """
    Solve: max_d  (1/K) sum_k log(R_k(d)+eps) - beta * chi2(d||dD)
    """
    S, A, g = env.n_states, env.n_actions, env.gamma
    if P is None:
        P = build_transition_tensor(env)
    if Rsa is None:
        Rsa = expected_obj_rewards(env, P)

    print(f"[occupancy.solve_fairdice_nsw] Solving NSW objective with beta={beta}")

    d = cp.Variable((S, A), nonneg=True)
    rho = cp.sum(d, axis=1)
    init = np.zeros(S); init[env.init_state] = 1.0
    inflow = cp.sum(cp.sum(cp.multiply(d[:, :, None], P), axis=0), axis=0)
    constraints = [
        rho == (1 - g) * init + g * inflow,
        cp.sum(d) == 1.0
    ]

    R = cp.sum(cp.sum(cp.multiply(d[:, :, None], Rsa), axis=0), axis=0)

    ratio = cp.multiply(d, 1.0 / dD)
    chi2 = 0.5 * cp.sum(cp.multiply(dD, cp.square(ratio - 1.0)))

    welfare = (1.0 / env.n_obj) * cp.sum(cp.log(R + eps))
    obj = cp.Maximize(welfare - beta * chi2)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    status = prob.status
    print(
        "[occupancy.solve_fairdice_nsw] Solver status="
        f"{status}, obj={prob.value:.6f}"
    )

    if status not in {"optimal", "optimal_inaccurate"}:
        print(
            "[occupancy.solve_fairdice_nsw] Warning: solver did not converge;"
            " skipping datapoint"
        )
        return None, None

    d_star = np.array(d.value, dtype=np.float64)
    R_star = np.array([np.sum(d_star * Rsa[:, :, k]) for k in range(env.n_obj)], dtype=np.float64)
    print("[occupancy.solve_fairdice_nsw] Extraction complete")
    return d_star, R_star
