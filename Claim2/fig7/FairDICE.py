from collections import namedtuple
import os
import jax
import jax.numpy as jnp
from flax import nnx
from policy import GaussianPolicy, MuNetwork, DiscretePolicy
from critic import Critic
from divergence import f, FDivergence, f_derivative_inverse
import orbax.checkpoint as orbax

NetworkState = namedtuple('NetworkState', ['graphdef', 'state', 'target_params'])
TrainState = namedtuple('TrainState', ['policy_state', 'nu_state', 'mu_state', 'step'])
Model = namedtuple('Model', ['network', 'optimizer', 'target_network'])

def alpha_scalarization(rewards, alpha):
    # rewards: (batch, n_objectives), all > 0
    eps = 1e-8  # avoid zero
    x = jnp.clip(rewards, a_min=eps)
    if alpha == 1:
        return jnp.sum(jnp.log(x), axis=1, keepdims=True)  # shape (batch,1)
    else:
        return jnp.sum(x ** (1 - alpha) / (1 - alpha), axis=1, keepdims=True)  # shape (batch,1)

def get_model(state: NetworkState) -> Model:
    network, optimizer = nnx.merge(state.graphdef, state.state)
    _, other_variables = state.state.split(nnx.Param, ...)
    target_network, _ = nnx.merge(state.graphdef, state.target_params, other_variables)
    
    return Model(network, optimizer, target_network)

    
import optax

def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)
    
    # Check if we're using discrete actions
    is_discrete = getattr(config, 'is_discrete', False)

    if is_discrete:
        # Use DiscretePolicy for discrete action spaces
        policy = DiscretePolicy(
            input_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            action_dim=config.action_dim,
            activation=nnx.relu,
            temperature=config.temperature,
            rngs=rngs,
            layer_norm=config.layer_norm
        )
    else:
        # Use GaussianPolicy for continuous action spaces
        policy = GaussianPolicy(
            input_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            action_dim=config.action_dim,
            activation=nnx.relu,
            temperature=config.temperature,
            tanh_squash_distribution=config.tanh_squash_distribution,
            rngs=rngs,
            layer_norm=config.layer_norm
        )
    
    policy_tx = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(
                optax.cosine_decay_schedule(-config.policy_lr, config.total_train_steps)
            )
        )
    policy_optim = nnx.Optimizer(policy, policy_tx, wrt=nnx.Param)
    (policy_gd, policy_state) = nnx.split((policy, policy_optim))

    nu = Critic(
        observation_dim=config.state_dim, 
        hidden_dims = config.hidden_dims, 
        layer_norm=config.layer_norm,
        rngs=rngs
    )
    nu_optim = nnx.Optimizer(nu, optax.adam(learning_rate=config.nu_lr), wrt=nnx.Param)
    (nu_gd, nu_state) = nnx.split((nu, nu_optim))
    
    mu = MuNetwork(config)
    mu_optim = nnx.Optimizer(mu, optax.adam(learning_rate=config.mu_lr), wrt=nnx.Param)
    (mu_gd, mu_state) = nnx.split((mu, mu_optim))
    
    nu_target = nu_state.filter(nnx.Param)
    policy_target = policy_state.filter(nnx.Param)
    mu_target = mu_state.filter(nnx.Param)
    
    return TrainState(
        policy_state=NetworkState(policy_gd, policy_state, policy_target),
        nu_state=NetworkState(nu_gd, nu_state, nu_target),
        mu_state=NetworkState(mu_gd, mu_state, mu_target),
        step = jnp.array(0)
        )

def train_step(config, train_state: TrainState, batch, key: jax.random.PRNGKey):   
    key, subkey = jax.random.split(key)
    step = train_state.step
    gamma = config.gamma
    beta = config.beta
    fixed_mu = getattr(config, "fixed_mu", None)
    rewards = batch.rewards 
    states = batch.states
    next_states = batch.next_states
    init_states = batch.init_states
    mask = batch.masks.astype(jnp.float32)
    
    # Check if using discrete actions
    is_discrete = getattr(config, 'is_discrete', False)
    
    policy, policy_optim, _ = get_model(train_state.policy_state)
    nu_network, nu_optim, _ = get_model(train_state.nu_state)
    mu_network, mu_optim, _ = get_model(train_state.mu_state) 

    f_divergence = FDivergence[config.divergence]
    eps = jax.random.uniform(subkey)

    def nu_loss_fn(nu_network, mu_network):
        nu = nu_network(states)
        next_nu = nu_network(next_states)
        init_nu = nu_network(init_states)
        mu = mu_network() 
        assert rewards.shape[-1] == config.reward_dim
        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        # For terminal states, don't bootstrap from next state (mask = 0 for terminals)
        e = (weighted_rewards + gamma * mask * next_nu - nu)
        w = jax.nn.softplus(f_derivative_inverse(e / beta, f_divergence))
        if len(w.shape) == 1:
            w = w.reshape(-1, 1)
        loss_1 = (1 - gamma) * jnp.mean(init_nu)
        # Include all transitions, not just non-terminal ones
        loss_2 = jnp.mean(w * e - beta * f(w, f_divergence))
        w_detached = jax.lax.stop_gradient(w)
        w_sum = jnp.sum(w_detached) + 1e-8
        k_hat = (1.0 - gamma) * (jnp.sum(w_detached * rewards, axis=0) / w_sum)
        k_hat = jnp.maximum(k_hat, 1e-8)
        loss_3 = jnp.sum(jnp.log(k_hat) - mu * k_hat)

        def nu_scalar(x):
            return jnp.squeeze(nu_network(x), -1)  
        interpolated_observations = init_states * eps + next_states * (1 - eps)
        grad_fn = jax.vmap(jax.grad(nu_scalar), in_axes=0)
        nu_grad = grad_fn(interpolated_observations)             
        grad_norm = jnp.linalg.norm(nu_grad, axis=1)            
        grad_penalty = (
            config.gradient_penalty_coeff *
            jnp.mean(jax.nn.relu(grad_norm - 5.) ** 2)
        )
        
        nu_loss = loss_1 + loss_2 + loss_3 + grad_penalty
        return nu_loss, (w, e, mu, grad_penalty)

    if fixed_mu is None:
        (nu_loss, (w, e, mu, grad_penalty)), (nu_grads, mu_grads) = nnx.value_and_grad(
            nu_loss_fn, argnums=(0, 1), has_aux=True
        )(nu_network, mu_network)
        nu_optim.update(nu_network, nu_grads)
        mu_optim.update(mu_network, mu_grads)
        nu_state_ = nnx.state((nu_network, nu_optim))
        mu_state_ = nnx.state((mu_network, mu_optim))
    else:
        fixed_mu = jnp.asarray(fixed_mu, dtype=jnp.float32)

        def nu_loss_fn_fixed(nu_network):
            nu = nu_network(states)
            next_nu = nu_network(next_states)
            init_nu = nu_network(init_states)
            mu = fixed_mu
            assert rewards.shape[-1] == config.reward_dim
            weighted_rewards = (rewards @ mu).reshape(-1, 1)
            e = (weighted_rewards + gamma * mask * next_nu - nu)
            w = jax.nn.softplus(f_derivative_inverse(e / beta, f_divergence))
            if len(w.shape) == 1:
                w = w.reshape(-1, 1)
            loss_1 = (1 - gamma) * jnp.mean(init_nu)
            loss_2 = jnp.mean(w * e - beta * f(w, f_divergence))
            w_detached = jax.lax.stop_gradient(w)
            w_sum = jnp.sum(w_detached) + 1e-8
            k_hat = (1.0 - gamma) * (jnp.sum(w_detached * rewards, axis=0) / w_sum)
            k_hat = jnp.maximum(k_hat, 1e-8)
            loss_3 = jnp.sum(jnp.log(k_hat) - mu * k_hat)

            def nu_scalar(x):
                return jnp.squeeze(nu_network(x), -1)
            interpolated_observations = init_states * eps + next_states * (1 - eps)
            grad_fn = jax.vmap(jax.grad(nu_scalar), in_axes=0)
            nu_grad = grad_fn(interpolated_observations)
            grad_norm = jnp.linalg.norm(nu_grad, axis=1)
            grad_penalty = (
                config.gradient_penalty_coeff *
                jnp.mean(jax.nn.relu(grad_norm - 5.) ** 2)
            )

            nu_loss = loss_1 + loss_2 + loss_3 + grad_penalty
            return nu_loss, (w, e, mu, grad_penalty)

        (nu_loss, (w, e, mu, grad_penalty)), nu_grads = nnx.value_and_grad(
            nu_loss_fn_fixed, has_aux=True
        )(nu_network)
        nu_optim.update(nu_network, nu_grads)
        nu_state_ = nnx.state((nu_network, nu_optim))
        mu_state_ = train_state.mu_state.state
    
    def policy_loss_fn(policy):
        dist = policy(states)
        
        if is_discrete:
            # For discrete actions, convert actions to integer indices
            # Actions in batch are stored as floats, need to convert to int for Categorical log_prob
            actions_int = batch.actions.astype(jnp.int32).flatten()
            log_probs = dist.log_prob(actions_int).reshape(-1, 1)
        else:
            # For continuous actions, use actions directly
            log_probs = dist.log_prob(batch.actions)
            # Ensure log_probs has the right shape
            if log_probs.ndim == 1:
                log_probs = log_probs.reshape(-1, 1)
        
        #weighted_rewards = (rewards @ mu).reshape(-1, 1)
        weighted_rewards = alpha_scalarization(rewards, config.alpha)
        nu_val = nu_network(states)
        next_nu = nu_network(next_states)
        e_val = (weighted_rewards + gamma * next_nu - nu_val)
        stable_w = jax.lax.stop_gradient(
            jax.nn.relu(f_derivative_inverse((e_val - jnp.max(e_val))/ beta, f_divergence))
        )
        stable_w = stable_w / (jnp.mean(stable_w) + 1e-8)
        policy_loss = -(mask * stable_w * log_probs).sum() / (jnp.sum(mask) + 1e-8)
        return policy_loss, log_probs
        
    (policy_loss, log_probs), policy_grads= nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy)
    policy_optim.update(policy, policy_grads)
    policy_state_ = nnx.state((policy, policy_optim))

    train_state = train_state._replace(
        policy_state = train_state.policy_state._replace(
            state = policy_state_,
            ),
        nu_state = train_state.nu_state._replace(
            state = nu_state_,
            ),
        mu_state = train_state.mu_state._replace(state = mu_state_),
        step = step + 1,
    )
    
    return train_state, {
        "policy_loss": policy_loss,
        "nu_loss": nu_loss,
        "mu": mu if fixed_mu is None else jnp.asarray(fixed_mu, dtype=jnp.float32),
        "grad_penalty": grad_penalty,
    }
    
def save_model(train_state: TrainState, path: str):
    # Orbax requires absolute checkpoint paths.
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(path, train_state)

def load_model(path: str, config) -> TrainState:
    # Be consistent with save_model and Orbax's absolute-path requirement.
    path = os.path.abspath(path)
    checkpointer = orbax.PyTreeCheckpointer()
    train_state = init_train_state(config)
    train_state = checkpointer.restore(path, item=train_state)
    return train_state
