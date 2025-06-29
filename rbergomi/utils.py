import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return bs(F, K, s**2 * t, o) - P
    s = brentq(error, 1e-9, 1e+9)
    return s

def generate_rDonsker_Y_cholesky(N, H, T, n): # taken from Horvath et al. 2017b 
    """
    Generates N paths of the Volterra process Y_t using rDonsker method.
    H is the Hurst parameter (H = a + 0.5), T is total time, n is steps/year.
    """
    
    dt = T / n
    t_grid = np.linspace(0, T, n + 1)
    s = len(t_grid)

    # Construct covariance matrix for fractional Brownian motion
    def fBM_cov(H, t_grid):
        return 0.5 * (np.abs(t_grid[:, None])**(2 * H) +
                      np.abs(t_grid[None, :])**(2 * H) -
                      np.abs(t_grid[:, None] - t_grid[None, :])**(2 * H))

    cov_matrix = fBM_cov(H, t_grid)
    L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(s))  # Stability jitter

    # Generate Brownian paths and transform
    Z = np.random.randn(N, s)
    Y = Z @ L.T  # Shape: (N, s)

    return Y


def generate_rDonsker_Y_optimal(N, H, T, n, kernel="optimal"):
    """
    Generates N paths of the Volterra process Y_t using the rDonsker method. CFR Horvath et al. 2017b.
    
    Parameters:
        N : int
            Number of paths
        H : float
            Hurst parameter (H = a + 0.5)
        T : float
            Time horizon
        n : int
            Number of time steps
        kernel : str
            'optimal' (moment-matching) or 'naive' (left-point approximation)
    
    Returns:
        Y : ndarray
            Simulated Volterra process paths of shape (N, n+1)
    """
    dt = T / n
    i = np.arange(1, n + 1)  # start from 1 to avoid 0^x

    # Step 1: Compute kernel weights
    if kernel == "optimal":
        opt_k = np.power((np.power(i,2*H)-np.power(i-1.,2*H))/2.0/H,0.5)
    elif kernel == "naive":
        opt_k = np.power(i,H-0.5)
    else:
        raise ValueError("Invalid kernel choice. Use 'optimal' or 'naive'.")

    # generate brownian motions
    dW = np.sqrt(dt) * np.random.randn(N, n)

    # Step 3: Convolve and construct the fractional process
    Y = np.zeros((N, n + 1))  # +1 to include t=0
    for j in range(N):
        conv = np.convolve(opt_k, dW[j])[:n]  # truncate to n points
        Y[j, 1:] = conv  # leave Y[:, 0] = 0 for t = 0

    # Step 4: Apply fractional Brownian motion scaling
    return Y * T**H


def generate_piecewise_forward_variance(T=1.0, n=100, num_segments=8, xi_pieces=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    t_grid = np.linspace(0, T, num_segments + 1)

    if xi_pieces is None:
        xi_pieces = np.random.uniform(0.01, 0.16, num_segments)
    else:
        xi_pieces = np.array(xi_pieces)

    t_full = np.linspace(0, T, int(T * n) + 1)
    xi_curve = np.zeros_like(t_full)

    for i in range(num_segments):
        start = np.searchsorted(t_full, t_grid[i])
        end = np.searchsorted(t_full, t_grid[i + 1]) if i < num_segments - 1 else len(t_full)
        xi_curve[start:end] = xi_pieces[i]

    return xi_curve, t_full, xi_pieces




