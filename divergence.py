from typing_extensions import assert_never
from enum import Enum
import jax
import jax.numpy as jnp


SOFT_CHI_NEG_CLIP = -3.0
SOFT_CHI_POS_CLIP = 10.0


class FDivergence(str, Enum):
    KL = "KL"
    CHI = "Chi"
    SOFT_CHI = "SoftChi"
    DUAL_DICE = "DualDICE"

def f(x, f_divergence: FDivergence, eps: float = 1e-10):
    x = jnp.array(x)

    if f_divergence == FDivergence.KL:
        return x * jnp.log(x + eps)
    elif f_divergence == FDivergence.CHI:
        return (x - 1) ** 2 / 2
    elif f_divergence == FDivergence.SOFT_CHI:
        return jnp.where(x < 1.0, x * jnp.log(x + eps) - x + 1, (x - 1) ** 2 / 2)
    elif f_divergence == FDivergence.DUAL_DICE:
        return 2 / 3 * jnp.abs(x) ** (3 / 2)
    else:
        assert_never(f_divergence)        

def f_derivative_inverse(y, f_divergence: FDivergence, t: float = 1.0):
    y = jnp.array(y)
    if f_divergence == FDivergence.KL:
        return jnp.exp(y - 1.0)
    elif f_divergence == FDivergence.CHI:
        return y + 1.0
    elif f_divergence == FDivergence.SOFT_CHI:
        y_neg = jnp.clip(y, SOFT_CHI_NEG_CLIP, 0.0)
        y_pos = jnp.clip(y, 0.0, SOFT_CHI_POS_CLIP)
        return jnp.where(y < 0.0, jnp.exp(y_neg), y_pos + 1.0)
    elif f_divergence == FDivergence.DUAL_DICE:
        raise ValueError(f"This function doesn't exist for {f_divergence}.")
    else:
        assert_never(f_divergence)
            
def state_action_ratio(
    nu,
    next_nu,
    rewards,
    beta: float,
    discount: float,
    f_divergence: FDivergence,
):

    nu = jnp.array(nu)
    next_nu = jnp.array(next_nu)
    rewards = jnp.array(rewards)

    e = rewards + discount * next_nu - nu
    return jax.nn.softplus(f_derivative_inverse(e / beta, f_divergence))
