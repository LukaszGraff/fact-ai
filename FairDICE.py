from collections import namedtuple
import jax
import jax.numpy as jnp
from flax import nnx
from policy import GaussianPolicy, DiscretePolicy, MuNetwork
from critic import Critic
from divergence import f, FDivergence, f_derivative_inverse
import orbax.checkpoint as orbax

NetworkState = namedtuple('NetworkState', ['graphdef', 'state', 'target_params'])
TrainState = namedtuple('TrainState', ['policy_state', 'nu_state', 'mu_state', 'step', 'mu_prev'])
Model = namedtuple('Model', ['network', 'optimizer', 'target_network'])

def get_model(state: NetworkState) -> Model:
    network, optimizer = nnx.merge(state.graphdef, state.state)
    _, other_variables = state.state.split(nnx.Param, ...)
    target_network, _ = nnx.merge(state.graphdef, state.target_params, other_variables)
    
    return Model(network, optimizer, target_network)

    
import optax

def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)
    debug_mu = hasattr(config, 'debug_mu') and config.debug_mu
    mu_init_noise_std = getattr(config, "mu_init_noise_std", 0.0)
    if mu_init_noise_std > 0.0:
        key = jax.random.PRNGKey(config.seed)
        mu_init = 1.0 + mu_init_noise_std * jax.random.normal(key, (config.reward_dim,))
        config.mu_init = mu_init

    # Check if action space is discrete
    is_discrete = hasattr(config, 'is_discrete') and config.is_discrete
    
    if is_discrete:
        policy = DiscretePolicy(
            input_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            action_dim=config.action_dim,
            activation=nnx.relu,
            rngs=rngs,
            layer_norm=config.layer_norm
        )
    else:
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
    
    mu_prev = mu() if debug_mu else None
    return TrainState(
        policy_state=NetworkState(policy_gd, policy_state, policy_target),
        nu_state=NetworkState(nu_gd, nu_state, nu_target),
        mu_state=NetworkState(mu_gd, mu_state, mu_target),
        step = jnp.array(0),
        mu_prev = mu_prev
        )

def train_step(config, train_state: TrainState, batch, key: jax.random.PRNGKey):   
    key, subkey = jax.random.split(key)
    step = train_state.step
    gamma = config.gamma
    beta = config.beta
    rewards = batch.rewards
    states = batch.states
    next_states = batch.next_states
    init_states = batch.init_states
    mask = batch.masks.astype(jnp.float32)
    if len(mask.shape) == 1:
        mask = mask.reshape(-1, 1)
    
    policy, policy_optim, _ = get_model(train_state.policy_state)
    nu_network, nu_optim, _ = get_model(train_state.nu_state)
    mu_network, mu_optim, _ = get_model(train_state.mu_state) 

    f_divergence = FDivergence[config.divergence]
    eps = jax.random.uniform(subkey)

    debug_mu = hasattr(config, 'debug_mu') and config.debug_mu

    def nu_loss_fn(nu_network, mu_network):
        nu = nu_network(states)
        next_nu = nu_network(next_states)
        init_nu = nu_network(init_states)
        mu = mu_network() 
        assert rewards.shape[-1] == config.reward_dim
        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        e = (weighted_rewards + gamma * mask * next_nu - nu)
        w = jax.nn.relu(f_derivative_inverse(e / beta, f_divergence))
        if len(w.shape) == 1:
            w = w.reshape(-1, 1)
        loss_1 = (1 - gamma) * jnp.mean(init_nu)
        term = (w * e - beta * f(w, f_divergence))
        loss_2 = jnp.mean(term)
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
        if debug_mu:
            e_over_beta = e / beta
            mean_e_over_beta = jnp.mean(e_over_beta)
            max_e_over_beta = jnp.max(e_over_beta)
            min_e_over_beta = jnp.min(e_over_beta)
            frac_e_over_beta_lt_neg10 = jnp.mean((e_over_beta < -10.0).astype(jnp.float32))
            frac_e_over_beta_gt_neg1 = jnp.mean((e_over_beta > -1.0).astype(jnp.float32))
            frac_w_pos = jnp.mean((w > 0).astype(jnp.float32))
            frac_w_gt_5 = jnp.mean((w > 5.0).astype(jnp.float32))
            frac_w_gt_20 = jnp.mean((w > 20.0).astype(jnp.float32))
            mean_w = jnp.mean(w)
            max_w = jnp.max(w)
            terminal_mask = (1.0 - mask)
            terminal_reward_mean = jnp.mean(jnp.linalg.norm(rewards, axis=-1) * terminal_mask.squeeze(-1))
            return nu_loss, (w, e, mu, grad_penalty, loss_1, loss_2, loss_3, mean_e_over_beta, max_e_over_beta, min_e_over_beta, frac_w_pos, mean_w, max_w, frac_e_over_beta_lt_neg10, frac_e_over_beta_gt_neg1, frac_w_gt_5, frac_w_gt_20, terminal_reward_mean, k_hat)
        return nu_loss, (w, e, mu, grad_penalty)

    (nu_loss, aux_out),  (nu_grads, mu_grads) = nnx.value_and_grad(nu_loss_fn, argnums = (0, 1), has_aux=True)(nu_network, mu_network)
    if debug_mu:
        (w, e, mu, grad_penalty, loss_1, loss_2, loss_3, mean_e_over_beta, max_e_over_beta, min_e_over_beta, frac_w_pos, mean_w, max_w, frac_e_over_beta_lt_neg10, frac_e_over_beta_gt_neg1, frac_w_gt_5, frac_w_gt_20, terminal_reward_mean, k_hat) = aux_out
    else:
        (w, e, mu, grad_penalty) = aux_out
    nu_optim.update(nu_network, nu_grads)
    mu_optim.update(mu_network, mu_grads)
    
    nu_state_ = nnx.state((nu_network, nu_optim))
    mu_state_ = nnx.state((mu_network, mu_optim))
    
    # Check if action space is discrete
    is_discrete = hasattr(config, 'is_discrete') and config.is_discrete
    
    def policy_loss_fn(policy):
        output = policy(states)
        
        if is_discrete:
            # For discrete policies: output is (logits, probs)
            logits, probs = output
            log_probs = jnp.log(probs + 1e-8)  # Add small epsilon for numerical stability
            # Select log probs for the actions taken
            actions_int = batch.actions.astype(jnp.int32)
            log_probs = log_probs[jnp.arange(log_probs.shape[0]), actions_int]
            # Reshape to match expected shape for loss computation
            log_probs = log_probs.reshape(-1, 1)
        else:
            # For continuous policies: output is a distribution
            dist = output
            log_probs = dist.log_prob(batch.actions)
            if len(log_probs.shape) == 1:
                log_probs = log_probs.reshape(-1, 1)
            
        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        nu_val = nu_network(states)
        next_nu = nu_network(next_states)
        e_val = (weighted_rewards + gamma * mask * next_nu - nu_val)
        w_raw = jax.lax.stop_gradient(
            jax.nn.relu(f_derivative_inverse((e_val - jnp.max(e_val))/ beta, f_divergence))
        )
        stable_w = w_raw / (jnp.mean(w_raw) + 1e-8)
        policy_mask = jnp.ones_like(mask)
        policy_loss = -(policy_mask * stable_w * log_probs).sum() / (jnp.sum(policy_mask) + 1e-8)
        avg_log_prob = jnp.mean(log_probs)
        avg_w = jnp.mean(stable_w)
        avg_e = jnp.mean(e_val)
        avg_w_raw = jnp.mean(w_raw)
        avg_mask = jnp.mean(mask)
        avg_neg_log_prob = jnp.mean(-log_probs)
        frac_w_pos = jnp.mean((w_raw > 0).astype(jnp.float32))
        avg_weighted_neg_log_prob = jnp.sum(mask * stable_w * (-log_probs)) / (jnp.sum(mask) + 1e-8)
        avg_masked_w = jnp.sum(policy_mask * stable_w) / (jnp.sum(policy_mask) + 1e-8)
        frac_w_pos_masked = jnp.sum(policy_mask * (w_raw > 0)) / (jnp.sum(policy_mask) + 1e-8)
        return policy_loss, (log_probs, avg_log_prob, avg_w, avg_e, avg_w_raw, avg_mask, avg_neg_log_prob, frac_w_pos, avg_weighted_neg_log_prob, avg_masked_w, frac_w_pos_masked)
        
    (policy_loss, (log_probs, avg_log_prob, avg_w, avg_e, avg_w_raw, avg_mask, avg_neg_log_prob, frac_w_pos, avg_weighted_neg_log_prob, avg_masked_w, frac_w_pos_masked)), policy_grads = nnx.value_and_grad(policy_loss_fn, has_aux=True)(policy)
    policy_optim.update(policy, policy_grads)
    policy_state_ = nnx.state((policy, policy_optim))

    mu_prev = train_state.mu_prev
    mu_after = mu_network() if debug_mu else None
    mu_delta = None
    if debug_mu:
        mu_delta = mu_after - mu_prev if mu_prev is not None else jnp.zeros_like(mu_after)

    train_state = train_state._replace(
        policy_state = train_state.policy_state._replace(
            state = policy_state_,
            ),
        nu_state = train_state.nu_state._replace(
            state = nu_state_,
            ),
        mu_state = train_state.mu_state._replace(state = mu_state_),
        step = step + 1,
        mu_prev = mu_after if debug_mu else train_state.mu_prev,
    )

    metrics = {
        "policy_loss": policy_loss,
        "nu_loss": nu_loss,
        "mu": mu_after if debug_mu else mu,
        "grad_penalty": grad_penalty,
    }
    if debug_mu:
        mu_grad_vec = jax.tree_util.tree_leaves(mu_grads)[0]
        abs_mu_grad = jnp.abs(mu_grad_vec)
        mu_grad_norm = jnp.linalg.norm(mu_grad_vec)
        metrics.update({
            "mu_delta": mu_delta,
            "mu_grad": mu_grad_vec,
            "abs_mu_grad_mean": jnp.mean(abs_mu_grad),
            "abs_mu_grad_max": jnp.max(abs_mu_grad),
            "mu_grad_norm": mu_grad_norm,
            "loss_1": loss_1,
            "loss_2": loss_2,
            "loss_3": loss_3,
            "frac_w_pos": frac_w_pos,
            "mean_w": mean_w,
            "max_w": max_w,
            "mean_e_over_beta": mean_e_over_beta,
            "max_e_over_beta": max_e_over_beta,
            "min_e_over_beta": min_e_over_beta,
            "frac_e_over_beta_lt_neg10": frac_e_over_beta_lt_neg10,
            "frac_e_over_beta_gt_neg1": frac_e_over_beta_gt_neg1,
            "frac_w_gt_5": frac_w_gt_5,
            "frac_w_gt_20": frac_w_gt_20,
            "terminal_reward_mean": terminal_reward_mean,
            "k_hat": k_hat,
        })

        def loss2_mu(mu_vec):
            weighted_rewards = (rewards @ mu_vec).reshape(-1, 1)
            e_val = (weighted_rewards + gamma * mask * next_nu - nu_val)
            w_val = jax.nn.relu(f_derivative_inverse(e_val / beta, f_divergence))
            term = (w_val * e_val - beta * f(w_val, f_divergence))
            return jnp.mean(term)

        def loss3_mu(mu_vec):
            k_val = 1.0 / (mu_vec + 1e-8)
            return jnp.sum(jnp.log(k_val) - mu_vec * k_val)

        nu_val = nu_network(states)
        next_nu = nu_network(next_states)
        grad_loss2_mu = jax.grad(loss2_mu)(mu)
        grad_loss3_mu = jax.grad(loss3_mu)(mu)
        metrics.update({
            "grad_loss2_mu": grad_loss2_mu,
            "grad_loss3_mu": grad_loss3_mu,
            "grad_loss2_mu_norm": jnp.linalg.norm(grad_loss2_mu),
            "grad_loss3_mu_norm": jnp.linalg.norm(grad_loss3_mu),
        })
    return train_state, metrics
    
def save_model(train_state: TrainState, path: str):
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(path, train_state)

def load_model(path: str, config) -> TrainState:
    checkpointer = orbax.PyTreeCheckpointer()
    train_state = init_train_state(config)
    train_state =  checkpointer.restore(path, item= train_state)
    return train_state
