from flax import nnx
import jax.numpy as jnp
import jax

class MuNetwork(nnx.Module):
    def __init__(self, config):
        init_value = getattr(config, "mu_init", None)
        if init_value is None:
            init_value = jnp.full((config.reward_dim,), 1.0)
        self.mu = nnx.Param(jnp.array(init_value, dtype=jnp.float32))
        
    def __call__(self):
        mu_raw = nnx.softplus(self.mu)
        return mu_raw / (jnp.sum(mu_raw) + 1e-8)

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
    
class DiscretePolicy(nnx.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 action_dim,
                 activation=nnx.relu,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 layer_norm: bool = False):
        
        self.mlp_layer = MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation=activation, rngs=rngs, activate_final=True, layer_norm=layer_norm)
        self.layer = nnx.Linear(hidden_dims[-1], action_dim, rngs=rngs)
        

    def __call__(self, inputs):
        x = self.mlp_layer(inputs)
        
        logits = self.layer(x)
        probs = nnx.softmax(logits, axis=-1)
        return logits, probs

try:
    import tensorflow_probability.substrates.jax as tfp
except ModuleNotFoundError:
    tfp = None
    tfd = None
    tfb = None
else:
    tfd = tfp.distributions
    tfb = tfp.bijectors

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
        if tfd is None or tfb is None:
            raise ModuleNotFoundError("tensorflow_probability is required for GaussianPolicy")
        
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
        if tfd is None or tfb is None:
            raise ModuleNotFoundError("tensorflow_probability is required for MNDPolicy")
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
