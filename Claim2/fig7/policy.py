from flax import nnx
import jax.numpy as jnp

from flax import nnx
import jax.numpy as jnp

class MuNetwork(nnx.Module):
    def __init__(self,
                 config):
        fixed_mu = getattr(config, "fixed_mu", None)
        if fixed_mu is not None:
            init_value = jnp.asarray(fixed_mu, dtype=jnp.float32)
        else:
            init_value = jnp.full((config.reward_dim,), 1.0)
        self.mu = nnx.Param(init_value)
        
    def __call__(self):
        return self.mu * 1.0

class MLP(nnx.Module):
    def __init__(self, din, dout = 1, hidden_dims = [256, 256], activation = nnx.relu, rngs: nnx.Rngs = nnx.Rngs(0), activate_final: bool = False, dropout_rate: float = 0.0, layer_norm: bool = False):
        dims = [din] + hidden_dims + [dout]
        
        layer = []
        for i in range(len(dims) - 1):
            layer.append(nnx.Linear(dims[i], dims[i+1], rngs=rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))))
            if i < len(dims) - 2:
                if dropout_rate > 0:
                    layer.append(nnx.Dropout(dropout_rate, rngs=rngs))
                layer.append(activation)
                if layer_norm:
                    layer.append(nnx.LayerNorm(dims[i+1], rngs=rngs))
                
        if activate_final:
            layer.append(activation)
        
        self.layer = nnx.Sequential(
            *layer
        )
    
    def __call__(self, x):
        return self.layer(x)
    
class _Categorical:
    def __init__(self, logits):
        self.logits = logits
        self._log_probs = nnx.log_softmax(logits, axis=-1)

    def log_prob(self, actions):
        actions = actions.astype(jnp.int32)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        return jnp.take_along_axis(self._log_probs, actions, axis=-1).squeeze(-1)

    @property
    def probs(self):
        return jnp.exp(self._log_probs)


class DiscretePolicy(nnx.Module):
    """Policy for discrete action spaces that outputs a Categorical distribution."""
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 action_dim,
                 activation=nnx.relu,
                 temperature=1.0,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 layer_norm: bool = False):
        
        self.temperature = temperature
        self.action_dim = action_dim
        
        self.mlp_layer = MLP(
            input_dim, 
            hidden_dims[-1], 
            hidden_dims[:-1], 
            activation=activation, 
            rngs=rngs, 
            #activate_final=True, 
            layer_norm=layer_norm
        )
        self.logits_layer = nnx.Linear(
            hidden_dims[-1], 
            action_dim, 
            rngs=rngs, 
            kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )

    def __call__(self, inputs):
        x = self.mlp_layer(inputs)
        logits = self.logits_layer(x) / self.temperature
        return _Categorical(logits=logits)
    
    def get_logits(self, inputs):
        """Return raw logits for debugging or other purposes."""
        x = self.mlp_layer(inputs)
        return self.logits_layer(x) / self.temperature

def _get_tfp():
    try:
        import tensorflow_probability.substrates.jax as tfp
    except Exception as exc:
        raise ImportError("tensorflow_probability is required for GaussianPolicy") from exc
    return tfp.distributions, tfp.bijectors

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class GaussianPolicy(nnx.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 action_dim,
                 activation = nnx.relu,
                 temperature=1.0,
                 log_std_scale: float = 1e-3,
                 tanh_squash_distribution=True,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 layer_norm: bool = False
                 ):
        
        self.temperature = temperature
        self.tanh_squash_distribution = tanh_squash_distribution
        
        self.mlp_layer = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation=activation, rngs = rngs, activate_final=True, layer_norm=layer_norm)

        self.mean_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.std_layer = nnx.Linear(
            hidden_dims[-1], action_dim, rngs = rngs, kernel_init=nnx.initializers.orthogonal(log_std_scale)
        )
        self.action_dim = action_dim

    def __call__(self, inputs):
        tfd, tfb = _get_tfp()
        x = self.mlp_layer(inputs)
        
        means = self.mean_layer(x)
        if not self.tanh_squash_distribution:
            means = jnp.tanh(means)
        
        log_stds = self.std_layer(x)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
                
        dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * self.temperature)
        
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Tanh())
        else:
            return dist        
        
class MNDPolicy(nnx.Module):
    def __init__(self,
                 input_dim,
                 action_dim,
                 hidden_dims = [256, 256],
                 activation = nnx.relu,
                 n_mixture: int = 5,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 tanh_squash_distribution: bool = False,
                 temperature = 1.0,
                 log_std_scale: float = 1e-3,
                 layer_norm: bool = False
                 ):        
        self.temparature = temperature
        self.mlp_layer = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation=activation, rngs = rngs, activate_final=True, layer_norm=layer_norm)
        self.mean_layer = nnx.Linear(
            hidden_dims[-1], action_dim * n_mixture, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.std_layer = nnx.Linear(
            hidden_dims[-1], action_dim * n_mixture, rngs = rngs, kernel_init=nnx.initializers.orthogonal(log_std_scale)
        )
        self.logit_layer = nnx.Linear(
            hidden_dims[-1], n_mixture, rngs = rngs, kernel_init=nnx.initializers.orthogonal(jnp.sqrt(2))
        )
        self.tanh_squash = tanh_squash_distribution
        self.n_mixture = n_mixture
        self.action_dim = action_dim
    
    def __call__(self, observations):
        tfd, tfb = _get_tfp()
        x = self.mlp_layer(observations)
        
        means = self.mean_layer(x).reshape(-1, self.n_mixture, self.action_dim)
        if not self.tanh_squash:
            means = nnx.tanh(means)
        log_stds = self.std_layer(x).reshape(-1, self.n_mixture, self.action_dim)
        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        logits = self.logit_layer(x)
        
        dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(
            logits=logits),
        components_distribution=tfd.MultivariateNormalDiag(
        loc=means,       
        scale_diag=jnp.exp(log_stds) * self.temparature))
        
        if self.tanh_squash:
            return tfd.TransformedDistribution(distribution=dist,
                                               bijector=tfb.Tanh())
        else:
            return dist
