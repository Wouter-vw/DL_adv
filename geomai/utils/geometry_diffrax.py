import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.experimental.ode import odeint
import matplotlib.pyplot as plt
from functools import partial
import diffrax

# Note: JAX doesn't have direct equivalents for some SciPy functions like `solve_ivp` or `integrate.quad`.
# We'll use JAX's `odeint` for ODE solving and approximate integrals numerically.

# This function evaluates the differential equation c'' = f(c, c')
def geodesic_system(manifold, c, dc):
    # Input: c, dc (D x N)
    D, N = c.shape
    if dc.shape != c.shape:
        raise ValueError('c and dc must have the same shape.')

    # Evaluate the metric and its derivative
    M, dM = manifold.metric_tensor(c, nargout=2)

    # Prepare the output (D x N)
    ddc = jnp.zeros((D, N))

    # Diagonal Metric Case
    if manifold.is_diagonal():
        for n in range(N):
            dMn = jnp.squeeze(dM[n, :, :])
            numerator = -0.5 * (2 * (dMn * dc[:, n].reshape(-1, 1)) @ dc[:, n] - dMn.T @ (dc[:, n] ** 2))
            ddc = ddc.at[:, n].set(numerator.flatten() / M[n, :])
    else:
        M_inv = jnp.linalg.inv(M)  # N x D x D
        Term1 = dM.reshape(N, D, D * D, order='F')  # N x D x D^2
        Term2 = dM.reshape(N, D * D, D, order='F')  # N x D^2 x D

        for n in range(N):
            numerator = -0.5 * M_inv[n, :, :] @ (
                (2 * Term1[n, :, :] - Term2[n, :, :].T) @ jnp.kron(dc[:, n], dc[:, n])
            )
            ddc = ddc.at[:, n].set(numerator)
    return ddc

# This function changes the 2nd order ODE to two 1st order ODEs
def second2first_order(manifold, state, subset_of_weights):
    D = int(state.shape[0] / 2)

    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    if subset_of_weights == 'last_layer':
        # For the last layer, use the existing implementation
        cmm = geodesic_system(manifold, c, cm)  # D x N
    else:
        # For the full network, use the hvp implementation
        cmm = manifold.geodesic_system(c, cm)
    y = jnp.concatenate((cm.squeeze(), cmm.squeeze()), axis=0)
    return y

# If the ODE solver succeeded, provide the solution
def evaluate_solution(solution, t, t_scale):
    # Input: t (Tx0), t_scale is used to scale the curve to have correct length
    c_dc = solution.ys[1]
    D = int(c_dc.shape[0] / 2)

    if jnp.size(t) == 1:
        c = c_dc[:D].reshape(D, 1)
        dc = c_dc[D:].reshape(D, 1) * t_scale
    else:
        c = c_dc[:D, :]  # D x T
        dc = c_dc[D:, :] * t_scale  # D x T
    return c, dc

# This function implements the exponential map
def expmap(manifold, x, v, subset_of_weights='all'):
    assert subset_of_weights in ['all', 'last_layer'], 'subset_of_weights must be all or last_layer'

    # Input: v, x (D x 1)
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    # The solver needs the function in a specific format
    ode_fun = lambda t, c_dc, args: second2first_order(manifold, c_dc, subset_of_weights).flatten()

    if jnp.linalg.norm(v) > 1e-5:
        curve, failed = new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights)
    else:
        curve = lambda t: (
            x.reshape(D, 1).repeat(t.size, axis=1),
            v.reshape(D, 1).repeat(t.size, axis=1)
        )  # Return tuple (2D x T)
        failed = True

    return curve, failed

# This function solves the initial value problem for the implementation of the expmap
# @partial(jax.jit, static_argnums=(0,3))
def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
    D = x.shape[0]

    # Ensure inputs are JAX arrays
    x = jnp.array(x)
    v = jnp.array(v)

    init = jnp.concatenate((x, v), axis=0).flatten()  # Initial state

    failed = False

    # Time points where to solve the ODE
    t = jnp.linspace(0, 1, 100)

    # Define the ODE term
    term = diffrax.ODETerm(ode_fun)

    # Choose the solver (adaptive step size)
    solver = diffrax.Dopri5()

    # Initial condition
    y0 = init

    # Time span
    t0 = t[0]
    t1 = t[-1]

    # Compute initial step size (dt0)
    dt0 = float(t[1] - t[0])  # Assuming t has more than one element

    # Save at specified time points
    saveat = diffrax.SaveAt(ts=t)

    # Set up the step size controller for adaptive stepping
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-3)

    # Solve the ODE
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller
        )

    # Define the curve function
    curve = lambda tt: evaluate_solution(sol, tt, 1)

    return curve, failed
