from flax import nnx
from policy import MLP

class Critic(nnx.Module):
    def __init__(self, 
                 observation_dim,
                 action_dim: int = 0, 
                 hidden_dims = [256, 256], 
                 activation = nnx.relu, 
                 layer_norm: bool = False,
                 state_action_input: bool = False,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        if state_action_input:
            observation_dim += action_dim
        self.critic = MLP(
            observation_dim, hidden_dims=hidden_dims, activation=activation, rngs = rngs, layer_norm=layer_norm
        )

    def __call__(self, obesrvation):
        return self.critic(obesrvation)