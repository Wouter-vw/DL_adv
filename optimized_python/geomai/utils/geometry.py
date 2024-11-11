import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Enable compilation for speed-up in PyTorch 2.0
torch.set_default_dtype(torch.float64)

# Draws an ellipsoid that corresponds to the metric
def plot_metric(x, cov, color='r', inverse_metric=False, linewidth=1):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    N = 100
    theta = torch.linspace(0, 2 * torch.pi, N).reshape(N, 1)
    points = torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
    points = points * torch.sqrt(eigvals)
    points = torch.matmul(eigvecs, points.T).T
    points = points + x.flatten()
    points = points.cpu().numpy()
    plt.plot(points[:, 0], points[:, 1], c=color, linewidth=linewidth, label='Metric')

# This function evaluates the differential equation c'' = f(c, c')
def geodesic_system(manifold, c, dc):
    # Input: c, dc (D x N)
    D, N = c.shape
    if dc.shape != (D, N):
        print('geodesic_system: second and third input arguments must have same dimensionality\n')
        sys.exit(1)

    # Evaluate the metric and the derivative
    M, dM = manifold.metric_tensor(c, nargout=2)

    # Prepare the output (D x N)
    ddc = torch.zeros(D, N, device=c.device)

    if manifold.is_diagonal():
        # Vectorized computation for diagonal metric
        dMn = dM.squeeze()
        dc_expanded = dc.unsqueeze(2)  # D x N x 1
        term1 = 2 * torch.matmul(dMn * dc_expanded, dc.unsqueeze(1))
        term2 = torch.matmul(dMn.transpose(1, 2), dc ** 2)
        ddc = -0.5 * (term1.squeeze() - term2) / M.T
    else:
        # Non-diagonal metric case
        M_inv = torch.linalg.inv(M)  # N x D x D
        D_squared = D * D
        Term1 = dM.reshape(N, D, D_squared)
        Term2 = dM.reshape(N, D_squared, D)

        dc_kron = torch.einsum('ik,jk->ijk', dc, dc).reshape(N, D_squared)
        temp = torch.bmm(2 * Term1 - Term2.transpose(1, 2), dc_kron.unsqueeze(2)).squeeze(2)
        ddc = -0.5 * torch.bmm(M_inv, temp.unsqueeze(2)).squeeze(2).T

    return ddc

# Converts the 2nd order ODE to two 1st order ODEs
def second2first_order(manifold, state, subset_of_weights):
    D = state.shape[0] // 2

    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    if subset_of_weights == 'last_layer':
        cmm = geodesic_system(manifold, c, cm)  # D x N
    else:
        cmm = manifold.geodesic_system(c, cm)
    y = torch.cat((cm, cmm), dim=0)
    return y

# Provides a linear solution if the solver fails
def evaluate_failed_solution(p0, p1, t):
    # Input: p0, p1 (D x 1), t (T x 1)
    c = (1 - t) * p0 + t * p1  # D x T
    dc = (p1 - p0).unsqueeze(1).repeat(1, t.numel())  # D x T
    return c, dc

# Evaluates the solution from the ODE solver
def evaluate_solution(solution, t, t_scale):
    c_dc = solution(t * t_scale)
    D = c_dc.shape[0] // 2
    if np.size(t) == 1:
        c = c_dc[:D].reshape(D, 1)
        dc = c_dc[D:].reshape(D, 1) * t_scale
    else:
        c = c_dc[:D, :]  # D x T
        dc = c_dc[D:, :] * t_scale  # D x T
    return c, dc

def evaluate_spline_solution(curve, dcurve, t):
    c = curve(t)
    dc = dcurve(t)
    D = int(c.shape[0])

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c.reshape(D, 1)
        dc = dc.reshape(D, 1)
    else:
        c = c.T  # Because the c([0,..,1]) -> N x D
        dc = dc.T
    return c, dc

# Computes the infinitesimal length on a curve
def local_length(manifold, curve, t):
    c, dc = curve(t)
    D = c.shape[0]
    M = manifold.metric_tensor(c, nargout=1)
    if manifold.is_diagonal():
        dist = torch.sqrt(torch.sum(M.T * (dc ** 2), dim=0))
    else:
        dc_T = dc.T
        dc_rep = dc_T.unsqueeze(2).repeat(1, 1, D)
        Mdc = torch.sum(M * dc_rep, dim=1)
        dist = torch.sqrt(torch.sum(Mdc * dc_T, dim=1))
    return dist

# Numerically computes the length of the geodesic curve
def curve_length(manifold, curve, a=0, b=1, tol=1e-5, limit=50):
    N = 1000
    t = torch.linspace(a, b, N)
    dist = local_length(manifold, curve, t)
    length = torch.trapezoid(dist, t)
    return length.item()

# Plots a curve defined as a parametric function
def plot_curve(curve, **kwargs):
    N = 1000
    T = torch.linspace(0, 1, N)
    curve_eval = curve(T)[0]

    D = curve_eval.shape[0]  # Dimensionality of the curve

    if D == 2:
        plt.plot(curve_eval[0, :].cpu().numpy(), curve_eval[1, :].cpu().numpy(), **kwargs)
    elif D == 3:
        plt.plot(curve_eval[0, :].cpu().numpy(), curve_eval[1, :].cpu().numpy(), curve_eval[2, :].cpu().numpy(), **kwargs)

# Vectorizes a matrix by stacking the columns
def vec(x):
    return x.flatten().reshape(-1, 1)

# Implements the exponential map using PyTorch's ODE solver
def expmap(manifold, x, v, subset_of_weights='all'):
    assert subset_of_weights in ['all', 'last_layer'], 'subset_of_weights must be all or last_layer'

    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    def ode_fun(t, state):
        state = state.unsqueeze(1)
        y = second2first_order(manifold, state, subset_of_weights).squeeze(1)
        return y

    if torch.linalg.norm(v) > 1e-5:
        curve, failed = new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights)
    else:
        curve = lambda t: (x.repeat(1, t.numel()), v.repeat(1, t.numel()))
        failed = True

    return curve, failed

# Solves the initial value problem for the exponential map
def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
    D = x.shape[0]
    init = torch.cat((x, v), dim=0).squeeze()

    t = torch.linspace(0, 1, steps=100, device=x.device)
    failed = False

    solution = odeint(ode_fun, init, t, atol=1e-3, rtol=1e-6)
    solution = solution.T  # [state_size x num_timesteps]

    def curve(tt):
        tt = tt.to(x.device)
        c_dc = F.interpolate(solution.unsqueeze(0), size=tt.numel(), mode='linear', align_corners=False).squeeze(0)
        c = c_dc[:D, :]
        dc = c_dc[D:, :]
        return c, dc

    return curve, failed
