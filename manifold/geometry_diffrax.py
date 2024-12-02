import jax
import jax.numpy as jnp
import diffrax
from functools import partial

jax.config.update("jax_enable_x64", True)

# Function to convert second-order ODE to two first-order ODEs
@partial(jax.jit, static_argnums = (0))
def second2first_order(manifold, state):
    D = int(state.shape[0] / 2)

    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    cmm = manifold.geodesic_system(c, cm)
    y = jnp.concatenate((cm.squeeze(), cmm.squeeze()), axis=0)
    return y.reshape(-1)

# Function to evaluate the solution
def evaluate_solution(solution, t, t_scale):
    c_dc = solution.ys[t]
    D = int(c_dc.shape[0] / 2)

    c = c_dc[:D].reshape(D, 1)
    dc = c_dc[D:].reshape(D, 1) * t_scale
    return c, dc

# Function to compute the exponential map
@partial(jax.jit, static_argnames=["manifold"])
def expmap(manifold, x, v):
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    @jax.jit
    def ode_fun(t, c_dc, args):
        return second2first_order(manifold, c_dc).flatten()
    
    init = jnp.concatenate((x, v), axis=0).flatten()
    term = diffrax.ODETerm(ode_fun)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(pcoeff=0.0, icoeff=1.0, dcoeff=0.0, rtol=1e-6, atol=1e-3)

    y0 = init
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.2,
        y0=y0,
        saveat=diffrax.SaveAt(t1 = True),
        stepsize_controller=stepsize_controller,
    )

    final_c, final_dc = evaluate_solution(sol, 1, 1.0)
    return final_c, final_dc, False
