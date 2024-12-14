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

# Initiate the ODE Solver
@partial(jax.jit, static_argnums=(0,1))
def solve_with_solver(manifold, solver, x, v):
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    init = jnp.concatenate((x, v), axis=0).flatten()

    @jax.jit
    def function_ode(t, curve_velocity, args):       
        return first_order_odes(manifold, curve_velocity).flatten()

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(function_ode),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=0.2,
        y0=init,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(pcoeff=0.0, icoeff=1.0, dcoeff=0.0, rtol=1e-6, atol=1e-3)
    )
    return sol

# Jitted function to split curve and velocity
@jax.jit
def return_sol(sol):
    curve_velocity = sol.ys[1]
    D = int(curve_velocity.shape[0] / 2)
    curve = curve_velocity[:D].reshape(D, 1)
    velocity = curve_velocity[D:].reshape(D, 1)
    return curve, velocity

# Solving the exponential map. Change solver when fail, if fail again return init
def expmap(manifold, x, v):
    try:
        sol = solve_with_solver(manifold, diffrax.Tsit5(), x, v)
        failed = False
    except Exception as e:
        print("Solver failed with Tsit5, trying Dopri8")
        try:
            sol = solve_with_solver(manifold, diffrax.Dopri8(), x, v)
            failed = False
        except Exception as e:
            print("Solver failed with Dopri8 as well, returning init")
            failed = True
            return x, v, failed
    curve, velocity = return_sol(sol)
    return curve, velocity, failed
