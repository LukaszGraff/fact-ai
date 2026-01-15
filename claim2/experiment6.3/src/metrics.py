import numpy as np


def nsw(R, eps=1e-12):
    R = np.maximum(R, eps)
    return float(np.exp(np.mean(np.log(R))))


def nsw_log(R, eps=1e-12):
    R = np.maximum(R, eps)
    return float(np.mean(np.log(R)))


def nsw_log_sum(R, eps=1e-12):
    R = np.maximum(R, eps)
    return float(np.sum(np.log(R)))

def mu_from_nsw_opt(R, eps=1e-12):
    R = np.maximum(R, eps)
    mu = 1.0 / R
    mu = mu / np.sum(mu)
    return mu

def perturb_mu(mu, sigma, rng):
    log_mu = np.log(np.clip(mu, 1e-12, 1.0))
    eps = rng.normal(loc=0.0, scale=sigma, size=mu.shape)
    log_mu_t = log_mu + eps
    log_mu_t -= np.max(log_mu_t)
    mu_t = np.exp(log_mu_t)
    mu_t /= np.sum(mu_t)
    return mu_t
