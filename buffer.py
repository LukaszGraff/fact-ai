
import jax
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
from functools import partial

Batch = namedtuple('Batch', ['states', 'next_states', 'actions', 'rewards', 'masks', 'init_states'])

class Buffer():
    def __init__(self,
                 offline_data,
                 is_discrete: bool = False
                 ):

        states = jnp.array(offline_data['states'].astype(np.float32))
        next_states = jnp.array(offline_data['next_states'].astype(np.float32))
        init_states = jnp.array(offline_data['init_states'].astype(np.float32))
        actions_np = np.asarray(offline_data['actions'])
        if actions_np.ndim > 1 and actions_np.shape[-1] == 1:
            actions_np = actions_np.reshape(-1)
        actions_is_int = np.issubdtype(actions_np.dtype, np.integer)
        if is_discrete or actions_is_int:
            actions = jnp.array(actions_np.astype(np.int32))
        else:
            actions = jnp.array(actions_np.astype(np.float32))
        rewards = jnp.array(offline_data['rewards'].astype(np.float32))
        masks = jnp.array(1.0 - offline_data['terminals'].reshape(-1, 1).astype(np.float32))

        
        self.size=len(offline_data['actions'])
        
        self.data = Batch(states, next_states, actions, rewards, masks, init_states)
                
        
    def sample(self, key, batch_size):
        return _sample(self.data, self.size, batch_size, key)

def get_pytree_batch_item(tree_batch, index):
    return jax.tree_util.tree_map(lambda tb: tb[index], tree_batch)

sample_fn = jax.vmap(get_pytree_batch_item, in_axes=(None, 0))

@partial(jax.jit, static_argnums=(1, 2))
def _sample(data: Batch, size, batch_size, key):
    
    sample_pos = jax.random.randint(key, minval=0, maxval=size, shape=(batch_size,))
    return sample_fn(data, sample_pos)
