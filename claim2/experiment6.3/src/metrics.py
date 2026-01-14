import numpy as np

def nsw(R, eps=1e-12):
    R = np.maximum(R, eps)
    return float(np.exp(np.mean(np.log(R))))

def mu_from_nsw_opt(R, eps=1e-12):
    R = np.maximum(R, eps)
    mu = 1.0 / R
    mu = mu / np.sum(mu)
    return mu

def perturb_mu(mu, sigma, rng):
    eps = rng.normal(loc=0.0, scale=sigma, size=mu.shape)
    mu_t = mu * (1.0 + eps)
    mu_t = np.maximum(mu_t, 1e-12)
    mu_t = mu_t / np.sum(mu_t)
    return mu_t
