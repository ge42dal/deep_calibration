import numpy as np
import os 
from utils import g, b, cov, generate_piecewise_forward_variance, generate_rDonsker_Y_cholesky, bsinv
#import matplotlib.pyplot as plt
#import time
#from matplotlib.font_manager import FontProperties
import pandas as pd


class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.
    """

    def __init__(self, n=100, N=1000, T=1.00, a=-0.4, method='hybrid'):
        """
        Constructor for class.
        """
        # Basic assignments
        self.T = T  # Maturity
        self.n = n  # Granularity (steps per year)
        self.s = int(self.n * self.T)  # Steps
        self.dt = self.T/self.s  # Step size
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis, :] # Time Grid
        self.a = a  # Alpha
        self.N = N  # Paths
        self.method = method  # 'hybrid' or 'rdonsker'

        # Construct hybrid scheme correlation structure for kappa = 1 (for hybrid)
        self.e = np.array([0, 0])
        self.c = cov(self.a, self.n) 

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW=None):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        if self.method == 'hybrid':
            Y1 = np.zeros((self.N, 1 + self.s))  # Exact integrals
            Y2 = np.zeros((self.N, 1 + self.s))  # Riemann sums

            # Construct Y1 through exact integral
            for i in np.arange(1, 1 + self.s, 1):
                Y1[:, i] = dW[:, i-1, 1]  # Assumes kappa = 1

            # Construct arrays for convolution
            G = np.zeros(1 + self.s)  # Gamma
            for k in np.arange(2, 1 + self.s, 1):
                G[k] = g(b(k, self.a) * self.dt, self.a)

            X = dW[:, :, 0]  # Xi

            # Initialise convolution result, GX
            GX = np.zeros((self.N, len(X[0, :]) + len(G) - 1))

            # Compute convolution, FFT not used for small n
            # Possible to compute for all paths in C-layer?
            for i in range(self.N):
                GX[i, :] = np.convolve(G, X[i, :])

            # Extract appropriate part of convolution
            Y2 = GX[:, :1 + self.s]

            # Finally contruct and return full process
            Y = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
            return Y

        elif self.method == 'rdonsker':
            # Generate paths using rDonsker method
            H = self.a + 0.5
            Y = generate_rDonsker_Y_cholesky(self.N, H, self.T, self.s)
            return Y
        else:
            raise ValueError("Method must be either 'hybrid' or 'rdonsker'.")
        

    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho=0.0):
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi, eta=1.0):
        """
        rBergomi variance process with support for scalar, 1D, or full 2D xi(t)
        """

        assert Y.shape[1] == self.t.shape[1], f"Y.shape={Y.shape}, self.t.shape={self.t.shape}"

        self.eta = eta
        a = self.a

        if np.isscalar(xi):
            # Constant forward variance
            xi_t = xi * np.ones_like(self.t)

        elif isinstance(xi, (list, np.ndarray)):
            xi = np.array(xi)
            if xi.ndim == 1:
                if len(xi) < self.s: # shorter xi -> interpolate
                    nodes = np.linspace(0, self.T, len(xi))
                    xi_t = np.interp(self.t[0], nodes, xi)[np.newaxis, :]
                elif xi.shape == self.t.shape[1:] or xi.shape == self.t[0].shape:
                    # direct match with model discretisation (n_steps+1,) -> reshape
                    xi_t = xi[np.newaxis, :]
                else:
                    raise ValueError("Invalid 1D xi shape.")
            elif xi.shape == self.t.shape:
                xi_t = xi
            else:
                raise ValueError("Invalid shape for xi. Expected scalar, 1D, or matching self.t.")
        else:
            raise TypeError("xi must be a scalar, list, or numpy array.")

        V = xi_t * np.exp(eta * Y - 0.5 * eta**2 * self.t**(2 * a + 1))
        return V

    def S(self, V, dB, S0=1):
        """
        rBergomi price process.
        """
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Construct non-anticipative Riemann increments
        increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(integral)
        return S

    def S1(self, V, dW1, rho, S0=1):
        """
        rBergomi parallel price process.
        """
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = rho * np.sqrt(V[:, :-1]) * \
            dW1[:, :, 0] - 0.5 * rho**2 * V[:, :-1] * dt

        # Cumsum is a little slower than Python loop.
        integral = np.cumsum(increments, axis=1)

        S = np.zeros_like(V)
        S[:, 0] = S0 
        S[:, 1:] = S0 * np.exp(integral) # Spot martingale condition -> risk-neutrality
        return S

def simulate_one(args):
    i, method, n_paths, n_steps, T = args

    # Sample model parameters
    xi_pieces = np.random.uniform(0.01, 0.16, 8)
    nu = np.random.uniform(0.5, 4.0)
    rho = np.random.uniform(-0.95, -0.1)
    H = np.random.uniform(0.025, 0.5)
    a = H - 0.5

    # Forward variance curve using utils
    xi_curve, _, _ = generate_piecewise_forward_variance(
        T=T, n=n_steps, num_segments=8, xi_pieces=xi_pieces
    )

    # rBergomi simulation
    model = rBergomi(n=n_steps, N=n_paths, T=T, a=a, method=method)
    dW1 = model.dW1()
    Y = model.Y(dW1)
    V = model.V(Y, xi=xi_curve, eta=nu)
    dW2 = model.dW2()
    dB = model.dB(dW1, dW2, rho=rho)
    S = model.S(V, dB)

    # Time/maturity setup
    maturities = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    strikes = np.round(np.linspace(0.5, 1.5, 11), 2)
    t_grid = model.t[0]
    maturity_indices = [np.searchsorted(t_grid, t) for t in maturities]

    # Output dict
    results = {}

    # Store true model parameters
    for j, val in enumerate(xi_pieces):
        results[f'xi_{j}'] = float(val)
    results.update({'nu': nu, 'rho': rho, 'H': H})

    # Implied vol surface (MC)
    for t, idx in zip(maturities, maturity_indices):
        S_T = S[:, idx]
        F = np.mean(S_T)
        for K in strikes:
            call_price = np.mean(np.maximum(S_T - K, 0))
            try:
                iv = bsinv(call_price, F, K, t)
            except Exception:
                iv = np.nan
            results[f"iv_T{t:.1f}_K{K:.2f}"] = iv

    return results



    
def run_simulation_batch(method='hybrid', n_param=80000, n_paths=60000, n_steps=252, T=2.0, output_dir='results'):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import multiprocessing as mp

    os.makedirs(output_dir, exist_ok=True)

    # Prepare argument list for parallel calls
    args_list = [(i, method, n_paths, n_steps, T) for i in range(n_param)]

    # Parallel execution
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_one, args_list), total=n_param))

    # Flatten and save
    df = pd.DataFrame(results)
    df.to_parquet(os.path.join(output_dir, f'simulated_paths_{method}.parquet'))

    print(f"Saved {n_param} simulations using method '{method}' to {output_dir}/")




if __name__ == "__main__":

    run_simulation_batch(method='hybrid')

    run_simulation_batch(method='rdonsker')
    hybrid_sim = pd.read_parquet('results/simulated_paths_hybrid.parquet')
    rdonsker_sim = pd.read_parquet('results/simulated_paths_rdonsker.parquet')
    rdonsker_sim.to_csv('results/simulated_paths_rdonsker.csv', index=False)
    hybrid_sim.to_csv('results/simulated_paths_hybrid.csv', index=False)

    print(hybrid_sim.head())
    print(rdonsker_sim.head())





    
    # # example usage # rdonsker 
    # start_time_donsker = time.time()
    # model_rdonsker = rBergomi(n=252, N=10000, T=1.0, a=-0.4, method='rdonsker')
    # dW1 = model_rdonsker.dW1()
    # Y_donsker = model_rdonsker.Y(dW1)
    # dW2 = model_rdonsker.dW2()
    # dB = model_rdonsker.dB(dW1, dW2, rho=0.0)
    # xi_curve, t_grid, xi_pieces = generate_piecewise_forward_variance(T=model_rdonsker.T, n=model_rdonsker.n)
    # V_donsker = model_rdonsker.V(Y_donsker, xi_curve)
    # S_donsker = model_rdonsker.S(V_donsker, dB)
    # diff_donsker = time.time() - start_time_donsker
    # print(f"%--------------- time: %s seconds ---------------%{diff_donsker}")

    # start_time_hybrid = time.time()
    # model_hybrid = rBergomi(n=252, N=10000, T=1.0, a=-0.4, method='hybrid')
    # Y_hybrid = model_hybrid.Y(dW1)
    # V_hybrid = model_rdonsker.V(Y_hybrid, xi_curve)
    # S_hybrid = model_rdonsker.S(V_hybrid, dB)
    # diff_hybrid = time.time() - start_time_hybrid
    # print(f"%--------------- time: %s seconds ---------------%{diff_hybrid}")
    # font = FontProperties(family='Times New Roman')

    # # diff = time.time() - start_time
    # # print(f"%--------------- time: %s seconds ---------------%{time.time() - start_time}")
    # # print("Generated paths for rBergomi model:")
    # # print(S)
    #  # asset path hybrid 
    # plt.figure(figsize=(10, 6))
    # plt.title('rBergomi Asset Paths Hybrid', fontproperties=font, fontsize=14)
    # plt.xlabel('Time', fontproperties=font)
    # plt.ylabel('Price', fontproperties=font)
    # plt.text(0.5, 0.95, f'Time taken: {diff_hybrid} seconds', 
    #     horizontalalignment='left',
    #     fontproperties=font,
    #     verticalalignment='bottom',
    #     transform=plt.gca().transAxes)

    # for i in range(model_hybrid.N):
    #     plt.plot(model_hybrid.t[0], S_hybrid[i], label=f'rBergomi Path{i}')
    
    # # variance path hybrid

    # plt.figure(figsize=(10, 6))
    # plt.title('rBergomi Variance Paths Hybrid', fontproperties=font, fontsize=14)
    # plt.xlabel('Time', fontproperties=font)
    # plt.ylabel('Var', fontproperties=font)
    # plt.text(0.5, 0.95, f'Time taken: {diff_hybrid} seconds', 
    #     horizontalalignment='left',
    #     fontproperties=font,
    #     verticalalignment='bottom',
    #     transform=plt.gca().transAxes)

    # for i in range(model_hybrid.N):
    #     plt.plot(model_hybrid.t[0], V_hybrid[i], label=f'rBergomi Path{i}')
        
    # plt.figure(figsize=(10, 6))
    # plt.title('rBergomi Asset Paths rdonsker', fontproperties=font, fontsize=14)
    # plt.xlabel('Time', fontproperties=font)
    # plt.ylabel('Price', fontproperties=font)
    # plt.text(0.5, 0.95, f'Time taken: {diff_donsker} seconds',
    #          horizontalalignment='left',
    #          fontproperties=font,
    #          verticalalignment='bottom',
    #          transform=plt.gca().transAxes)

    # for i in range(model_rdonsker.N):
    #     plt.plot(model_rdonsker.t[0], S_donsker[i], label=f'rBergomi Path{i}')
    
    # # variance path donsker

    # plt.figure(figsize=(10, 6))
    # plt.title('rBergomi Variance Paths rDonsker', fontproperties=font, fontsize=14)
    # plt.xlabel('Time', fontproperties=font)
    # plt.ylabel('Var', fontproperties=font)
    # plt.text(0.5, 0.95, f'Time taken: {diff_hybrid} seconds', 
    #     horizontalalignment='left',
    #     fontproperties=font,
    #     verticalalignment='bottom',
    #     transform=plt.gca().transAxes)

    # for i in range(model_hybrid.N):
    #     plt.plot(model_hybrid.t[0], V_donsker[i], label=f'rBergomi Path{i}')
    
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(xi_curve, label='Forward Variance Curve')
    # plt.title(f'Piecewise Forward Variance Curve, T={model_rdonsker.T}, n={model_rdonsker.n}', fontsize=14, fontproperties=font)
    # plt.show()

 