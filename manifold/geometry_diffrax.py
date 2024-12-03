import jax
import jax.numpy as jnp
import diffrax
from functools import partial

jax.config.update("jax_enable_x64", True)

# Converting second-order ODE to two first-order ODEs
@partial(jax.jit, static_argnums = (0))
def first_order_odes(manifold, state):
    D = int(state.shape[0] / 2)

    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    curve = state[:D, :]  # D x N
    d_curve = state[D:, :]  # D x N
    d2_curve = manifold.geodesic_system(curve, d_curve)
    y = jnp.concatenate((d_curve.squeeze(), d2_curve.squeeze()), axis=0)
    return y.reshape(-1)

# Exponential map
@partial(jax.jit, static_argnames=["manifold"])
def expmap(manifold, x, v):
    x = x.reshape(-1, 1) # position
    v = v.reshape(-1, 1) # velocity

    # ODE functions
    @jax.jit
    def function_ode(t, c_dc, args):       
        return first_order_odes(manifold, c_dc).flatten()
    
    # Initial conditions
    init = jnp.concatenate((x, v), axis=0).flatten()

    # Setting parametrs for ODE solver
    term = diffrax.ODETerm(function_ode)
    solver = diffrax.Tsit5() # Integration method

    stepsize_controller = diffrax.PIDController(pcoeff=0.0, 
                                                icoeff=1.0, 
                                                dcoeff=0.0, 
                                                rtol=1e-6, 
                                                atol=1e-3)
    # Solving ODE
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.2,
        y0=init,
        saveat=diffrax.SaveAt(t1 = True),
        stepsize_controller=stepsize_controller,
    )

    # Solution
    curve_velocity = sol.ys[1]
    D = int(curve_velocity.shape[0] / 2)

    curve = curve_velocity[:D].reshape(D, 1)
    velocity = curve_velocity[D:].reshape(D, 1)

    return curve, velocity, False
