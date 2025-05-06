import jax
import jax.numpy as jnp
from jax import jit, jacfwd, lax

############################### 1st Order Hold Functions ###############################
@jit
def dynamics_1order(state, input, input_derivs, g, m):
    x, y, z, vx, vy, vz, roll, pitch, yaw, thrust, rolldot, pitchdot, yawdot = state
    thrust_dot, roll_dd, pitch_dd, yaw_dd = input_derivs

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    # Update derivatives
    vxdot = -(thrust / m) * (sr * sy + cr * cy * sp) #rpy  = phi, theta, psi
    vydot = -(thrust / m) * (cr * sy * sp - cy * sr)
    vzdot = g - (thrust / m) * (cr * cp)

    return jnp.array([vx, vy, vz, vxdot, vydot, vzdot, rolldot, pitchdot, yawdot, thrust_dot, roll_dd, pitch_dd, yaw_dd])

# Function to integrate dynamics over time
@jit
def integrate_dynamics_1order(state, inputs, input_derivs, integration_step, integrations_int, g, m):
    def for_function(i, current_state):
        return current_state + dynamics_1order(current_state, inputs, input_derivs, g, m) * integration_step

    state = jnp.hstack([state, inputs])
    pred_state = lax.fori_loop(0, integrations_int, for_function, state)
    # print(f"done: {pred_state= }")
    return pred_state

# Prediction function 1st order
@jit
def predict_states_1order(state, last_input, input_derivs, T_lookahead, g, m, integration_step=0.1):
    inputs = last_input.flatten()
    integrations_int = 8  # Or another appropriate integer
    pred_state = integrate_dynamics_1order(state, inputs, input_derivs, integration_step, integrations_int, g, m)
    return pred_state[0:9]


# Prediction function
@jit
def predict_outputs_1order(state, last_input, input_derivs, T_lookahead, g, m, C, integration_step=0.1):
    inputs = last_input.flatten()
    integrations_int = 8  # Or another appropriate integer
    pred_state = integrate_dynamics_1order(state, inputs, input_derivs, integration_step, integrations_int, g, m)
    return C@pred_state[0:9]


# Compute Jacobian
@jit
def compute_jacobian_1order(state, last_input, input_derivs, T_lookahead, g, m, C, integration_step):
    jac_fn = jacfwd(lambda x: predict_outputs_1order(state, x, input_derivs, T_lookahead, g, m, C, integration_step))
    return jac_fn(last_input)

# Compute adjusted inverse Jacobian
@jit
def compute_adjusted_invjac_1order(state, last_input, input_derivs, T_lookahead, g, m, C, integration_step):
    jac = compute_jacobian_1order(state, last_input, input_derivs, T_lookahead, g, m, C, integration_step)
    inv_jacobian = jnp.linalg.pinv(jac)
    inv_jacobian_modified = inv_jacobian.at[:, 2].set(-inv_jacobian[:, 2])
    return inv_jacobian_modified




############################### 0 Order Hold Functions ###############################
@jit
def dynamics(state, inputs, g, m):
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    curr_thrust, curr_rolldot, curr_pitchdot, curr_yawdot = inputs

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / m) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / m) * (cr * sy * sp - cy * sr)
    vzdot = g - (curr_thrust / m) * (cr * cp)

    return jnp.array([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])

# Function to integrate dynamics over time
@jit
def integrate_dynamics(state, inputs, integration_step, integrations_int, g, m):
    def for_function(i, current_state):
        return current_state + dynamics(current_state, inputs, g, m) * integration_step

    pred_state = lax.fori_loop(0, integrations_int, for_function, state)
    return pred_state

# Prediction function
@jit
def predict_states(state, last_input, T_lookahead, g, m, integration_step=0.1):
    inputs = last_input.flatten()
    integrations_int = 8 #int(T_lookahead / integration_step)
    pred_state = integrate_dynamics(state, inputs, integration_step, integrations_int, g, m)
    return pred_state

# Prediction function
@jit
def predict_outputs(state, last_input, T_lookahead, g, m, C, integration_step=0.1):
    inputs = last_input.flatten()
    integrations_int = 8 #int(T_lookahead / integration_step)
    pred_state = integrate_dynamics(state, inputs, integration_step, integrations_int, g, m)
    return C@pred_state

# Compute Jacobian
@jit
def compute_jacobian(state, last_input, T_lookahead, g, m, C, integration_step):
    jac_fn = jacfwd(lambda x: predict_outputs(state, x, T_lookahead, g, m, C, integration_step))
    return jac_fn(last_input)

# Compute adjusted inverse Jacobian
@jit
def compute_adjusted_invjac(state, last_input, T_lookahead, g, m, C, integration_step):
    jac = compute_jacobian(state, last_input, T_lookahead, g, m, C, integration_step)
    inv_jacobian = jnp.linalg.pinv(jac)
    inv_jacobian_modified = inv_jacobian.at[:, 2].set(-inv_jacobian[:, 2])
    return inv_jacobian_modified
