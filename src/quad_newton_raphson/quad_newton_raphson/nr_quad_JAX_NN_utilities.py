import numpy as np
import flax
from flax import linen as nn
from typing import Any, Callable, Sequence
from jax import random
import jax
from flax import serialization
import jax.numpy as jnp
from functools import partial
from jax import jacfwd



INPUT = np.array([ 2.49944982,  1.60042476, -1.90054772, -0.09470656,  0.61739224,
        0.64762334,  0.17935197,  0.13538294, -0.05906635, 11.50631109,
        0.67154372,  0.71453788,  0.81230723])

OUTPUT = np.array([ 1.87724153,  2.57925116, -0.31868312,  0.59077944])

class FeedForwardLoad(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
    #   print(f"{i = }, {feat = }")
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

modelnew = FeedForwardLoad(features=[13, 128, 256, 256, 128, 4])

# Ensure proper key format
key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
x = INPUT
init_params = modelnew.init(key2, x)

# Restore parameters from bytes
with open("/home/factslabegmc/final_wardi_files/src/quad_newton_raphson/quad_newton_raphson/sim_params.bin", "rb") as f:
    loaded_bytes = f.read()
loaded_params = serialization.from_bytes(init_params, loaded_bytes)


def model_apply(params, state, ctrl):
    x = jnp.concatenate((state, ctrl))
    return modelnew.apply(params, x)

apply_model = jax.jit(partial(model_apply, loaded_params))


@jax.jit
def compute_jacobian(state, ctrl):
    return jacfwd(lambda x: apply_model(state, x))(ctrl)


@jax.jit
def compute_inv_jac(state,ctrl):
    jac = compute_jacobian(state,ctrl)
    cond = jnp.linalg.cond(jac)
    # jac += 1e-3 * jnp.eye(jac.shape[0])
    cond2 = jnp.linalg.cond(jac)

    inv_jacobian = jnp.linalg.pinv(jac)
    inv_jacobian_modified = inv_jacobian.at[:, 2].set(-inv_jacobian[:, 2])
    return inv_jacobian_modified, cond, cond2