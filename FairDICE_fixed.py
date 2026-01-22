from collections import namedtuple
import jax
import jax.numpy as jnp
from flax import nnx
from policy import GaussianPolicy, DiscretePolicy
from critic import Critic
from divergence import f, FDivergence, f_derivative_inverse
import orbax.checkpoint as orbax

NetworkState = namedtuple('NetworkState', ['graphdef', 'state', 'target_params'])
TrainState = namedtuple('TrainState', ['policy_state', 'nu_state', 'step'])
Model = namedtuple('Model', ['network', 'optimizer', 'target_network'])


def get_model(state: NetworkState) -> Model:
    network, optimizer = nnx.merge(state.graphdef, state.state)
    _, other_variables = state.state.split(nnx.Param, ...)
    target_network, _ = nnx.merge(state.graphdef, state.target_params, other_variables)
    return Model(network, optimizer, target_network)


import optax


def init_train_state(config) -> TrainState:
    rngs = nnx.Rngs(config.seed)
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
        hidden_dims=config.hidden_dims,
        layer_norm=config.layer_norm,
        rngs=rngs
    )
    nu_optim = nnx.Optimizer(nu, optax.adam(learning_rate=config.nu_lr), wrt=nnx.Param)
    (nu_gd, nu_state) = nnx.split((nu, nu_optim))

    nu_target = nu_state.filter(nnx.Param)
    policy_target = policy_state.filter(nnx.Param)

    return TrainState(
        policy_state=NetworkState(policy_gd, policy_state, policy_target),
        nu_state=NetworkState(nu_gd, nu_state, nu_target),
        step=jnp.array(0)
    )


def train_step_fixed(config, train_state: TrainState, batch, key: jax.random.PRNGKey):
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

    f_divergence = FDivergence[config.divergence]
    eps = jax.random.uniform(subkey)
    mu = jax.lax.stop_gradient(jnp.array(config.fixed_mu, dtype=jnp.float32))

    def nu_loss_fn(nu_network):
        nu = nu_network(states)
        next_nu = nu_network(next_states)
        init_nu = nu_network(init_states)
        assert rewards.shape[-1] == config.reward_dim
        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        e = (weighted_rewards + gamma * mask * next_nu - nu)
        w = jax.nn.relu(f_derivative_inverse(e / beta, f_divergence))
        loss_1 = (1 - gamma) * jnp.mean(init_nu)
        term = (w * e - beta * f(w, f_divergence))
        loss_2 = jnp.mean(term)

        def nu_scalar(x):
            return jnp.squeeze(nu_network(x), -1)
        interpolated_observations = init_states * eps + next_states * (1 - eps)
        grad_fn = jax.vmap(jax.grad(nu_scalar), in_axes=0)
        nu_grad = grad_fn(interpolated_observations)
        grad_norm = jnp.linalg.norm(nu_grad, axis=1)
        grad_penalty = (
            config.gradient_penalty_coeff *
            jnp.mean(jax.nn.relu(grad_norm - 5.0) ** 2)
        )
        nu_loss = loss_1 + loss_2 + grad_penalty
        return nu_loss, (w, e, grad_penalty)

    (nu_loss, (w, e, grad_penalty)), nu_grads = nnx.value_and_grad(nu_loss_fn, has_aux=True)(nu_network)
    nu_optim.update(nu_network, nu_grads)
    nu_state_ = nnx.state((nu_network, nu_optim))

    is_discrete = hasattr(config, 'is_discrete') and config.is_discrete

    def policy_loss_fn(policy):
        output = policy(states)
        if is_discrete:
            logits, probs = output
            log_probs = jnp.log(probs + 1e-8)
            actions_int = batch.actions.astype(jnp.int32)
            log_probs = log_probs[jnp.arange(log_probs.shape[0]), actions_int]
            log_probs = log_probs.reshape(-1, 1)
        else:
            dist = output
            log_probs = dist.log_prob(batch.actions)
            if len(log_probs.shape) == 1:
                log_probs = log_probs.reshape(-1, 1)

        weighted_rewards = (rewards @ mu).reshape(-1, 1)
        nu_val = nu_network(states)
        next_nu = nu_network(next_states)
        e_val = (weighted_rewards + gamma * mask * next_nu - nu_val)
        w_raw = jax.lax.stop_gradient(
            jax.nn.relu(f_derivative_inverse((e_val - jnp.max(e_val)) / beta, f_divergence))
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

    train_state = train_state._replace(
        policy_state=train_state.policy_state._replace(
            state=policy_state_,
        ),
        nu_state=train_state.nu_state._replace(
            state=nu_state_,
        ),
        step=step + 1,
    )

    return train_state, {
        "policy_loss": policy_loss,
        "nu_loss": nu_loss,
        "grad_penalty": grad_penalty,
        "avg_log_prob": avg_log_prob,
        "avg_w": avg_w,
        "avg_e": avg_e,
        "avg_w_raw": avg_w_raw,
        "avg_mask": avg_mask,
        "avg_neg_log_prob": avg_neg_log_prob,
        "frac_w_pos": frac_w_pos,
        "avg_weighted_neg_log_prob": avg_weighted_neg_log_prob,
        "avg_masked_w": avg_masked_w,
        "frac_w_pos_masked": frac_w_pos_masked,
    }


def save_model(train_state: TrainState, path: str):
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(path, train_state)


def load_model(path: str, config) -> TrainState:
    checkpointer = orbax.PyTreeCheckpointer()
    train_state = init_train_state(config)
    train_state = checkpointer.restore(path, item=train_state)
    return train_state
