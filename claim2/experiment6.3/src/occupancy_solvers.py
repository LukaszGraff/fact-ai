import numpy as np
import cvxpy as cp


def _discounted_returns_expr(d_var, Rsa, scale):
    return scale * cp.sum(cp.sum(cp.multiply(d_var[:, :, None], Rsa), axis=0), axis=0)


def _discounted_returns_numpy(d_arr, Rsa, scale):
    return scale * np.array([
        np.sum(d_arr * Rsa[:, :, k]) for k in range(Rsa.shape[2])
    ], dtype=np.float64)


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
    assert 0.0 < g < 1.0
    scale = 1.0 / (1.0 - g)
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
    R = _discounted_returns_expr(d, Rsa, scale)

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
    R_star = _discounted_returns_numpy(d_star, Rsa, scale)
    assert np.isfinite(R_star).all()
    print("[occupancy.solve_fairdice_fixed] Extraction complete")
    return d_star, R_star

def solve_fairdice_nsw(env, dD, beta, eps=1e-8, P=None, Rsa=None, verbose=False):
    """
    Solve: max_d  (1/K) sum_k log(R_k(d)) - beta * chi2(d||dD)
    Returns discounted occupancy d*, returns R*, and dual-based mu*.
    """
    S, A, g = env.n_states, env.n_actions, env.gamma
    assert 0.0 < g < 1.0
    scale = 1.0 / (1.0 - g)
    K = env.n_obj
    if P is None:
        P = build_transition_tensor(env)
    if Rsa is None:
        Rsa = expected_obj_rewards(env, P)

    print(f"[occupancy.solve_fairdice_nsw] Solving NSW objective with beta={beta}")

    d = cp.Variable((S, A), nonneg=True)
    k = cp.Variable(K)
    rho = cp.sum(d, axis=1)
    init = np.zeros(S); init[env.init_state] = 1.0
    inflow = cp.sum(cp.sum(cp.multiply(d[:, :, None], P), axis=0), axis=0)
    constraints = [
        rho == (1 - g) * init + g * inflow,
        cp.sum(d) == 1.0,
        k >= eps,
    ]

    R = _discounted_returns_expr(d, Rsa, scale)

    return_constraints = []
    for idx in range(K):
        c = R[idx] == k[idx]
        constraints.append(c)
        return_constraints.append(c)

    ratio = cp.multiply(d, 1.0 / dD)
    chi2 = 0.5 * cp.sum(cp.multiply(dD, cp.square(ratio - 1.0)))

    welfare = (1.0 / K) * cp.sum(cp.log(k))
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
        return None, None, None

    d_star = np.array(d.value, dtype=np.float64)
    R_star = _discounted_returns_numpy(d_star, Rsa, scale)
    assert np.isfinite(R_star).all()
    mu = np.zeros(K, dtype=np.float64)
    for idx, c in enumerate(return_constraints):
        dual_val = c.dual_value
        if dual_val is None:
            mu[idx] = 0.0
        else:
            mu[idx] = float(np.array(dual_val).squeeze())
    if np.mean(mu) < 0:
        mu = -mu
    mu = np.maximum(mu, 1e-12)

    if verbose:
        mu_norm = mu / mu.sum() if mu.sum() > 0 else mu
        inv = 1.0 / np.maximum(R_star, eps)
        inv /= inv.sum()
        print("[occupancy.solve_fairdice_nsw] R*=", np.round(R_star, 6))
        print(
            "[occupancy.solve_fairdice_nsw] mu_norm=",
            np.round(mu_norm, 6),
            " | (1/R)_norm=",
            np.round(inv, 6),
        )

    print("[occupancy.solve_fairdice_nsw] Extraction complete")
    return d_star, R_star, mu
