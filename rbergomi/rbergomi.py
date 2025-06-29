import numpy as np
import os 
from utils import g, b, cov, generate_piecewise_forward_variance, generate_rDonsker_Y_cholesky, bsinv
import matplotlib.pyplot as plt
#import time
from matplotlib.font_manager import FontProperties
import pandas as pd
import re
import glob

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
    i, method, n_paths, n_steps, T, xi_pieces = args
    np.random.seed(i + 12345)


    # Sample model parameters
    xi_pieces = np.random.uniform(0.01, 0.16, 8)
    nu = np.random.uniform(0.5, 4.0)
    rho = np.random.uniform(-0.95, -0.1)
    H = np.random.uniform(0.025, 0.5)
    a = H - 0.5

    # fixed parameters for testing
    # nu = 1.5
    # rho = -0.7
    # H = 0.1
    # a = H - 0.5

    xi_curve, _, _ = generate_piecewise_forward_variance(
    T=T, n=n_steps, num_segments=8, xi_pieces=xi_pieces)


    # rBergomi simulation
    model = rBergomi(n=n_steps, N=n_paths, T=T, a=a, method=method)
    dW1 = model.dW1()
    Y = model.Y(dW1)
    V = model.V(Y, xi=xi_curve, eta=nu)
    dW2 = model.dW2()
    dB = model.dB(dW1, dW2, rho=rho)
    S = model.S(V, dB)

    # Time/maturity setup
    # smaller qty maturities for testing
    #maturities = [0.3, 0.6, 1.2, 2.0]
    maturities = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    strikes = np.round(np.linspace(0.5, 1.5, 11), 2)
    # smaller qty strikes for testing 
    #strikes = [0.7, 0.9, 1.0, 1.1, 1.3]
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



    
def run_simulation_batch(method='hybrid', n_param=500, n_paths=2000, n_steps=100,
                         T=2.0, output_dir='results', xi_pieces=None, output_file=None):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count, Manager

    os.makedirs(output_dir, exist_ok=True)

    args_list = [(i, method, n_paths, n_steps, T, xi_pieces) for i in range(n_param)]

    # Shared progress bar
    with Manager() as manager:
        pbar = tqdm(total=n_param)
        lock = manager.Lock()

        def update_progress(_):
            with lock:
                pbar.update()

        with Pool(processes=cpu_count()) as pool:
            results = []
            for args in args_list:
                res = pool.apply_async(simulate_one, args=(args,), callback=update_progress)
                results.append(res)

            # Wait for all
            results = [r.get() for r in results]

        pbar.close()

    df = pd.DataFrame(results)

    if output_file is None:
        output_file = f'simulated_paths_{method}.parquet'

    df.to_parquet(os.path.join(output_dir, output_file))
    print(f"Saved {n_param} simulations using method '{method}' to {output_dir}/{output_file}")

def analyze_variance_decay(path_root='results/var_decay', method='hybrid', strikes=None, maturities=None):
    """
    Analyzes variance of implied vols across different path counts for given method.
    """

    files = sorted([f for f in os.listdir(path_root) if f.startswith(f'sim_{method}_npaths')])
    data = {}

    for f in files:
        match = re.search(r"npaths(\d+)", f)
        if match:
            path_count = int(match.group(1))
        else:
            continue
        df = pd.read_parquet(os.path.join(path_root, f))

        # Focus on IV columns only
        iv_cols = [col for col in df.columns if col.startswith('iv_')]
        if strikes or maturities:
            iv_cols = [
                c for c in iv_cols if any(f"K{strike:.2f}" in c for strike in strikes)
                and any(f"T{maturity:.1f}" in c for maturity in maturities)
            ]
        iv_std = df[iv_cols].std().mean()
        data[path_count] = iv_std

    # Plot
    plt.figure(figsize=(8, 5))
    x, y = zip(*sorted(data.items()))
    plt.plot(x, y, marker='o')
    plt.xlabel('Number of Paths')
    plt.ylabel('Avg Std of Implied Vols')
    plt.title(f'Variance Decay of Implied Vols ({method} method)')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for i in range(8):  # 8 batches × 10k = 80k total
        np.random.seed(1234 + i)  # Unique seed per batch
        output_name = f'sim_hybrid_batch{i}.parquet'
        print(f"Running batch {i+1}/8 with seed {1234 + i}")
        run_simulation_batch(
            method='hybrid',
            n_param=10000,
            n_paths=60000,
            n_steps=252,
            T=2.0,
            output_dir='results/full_sim',
            output_file=output_name
        )

    all_batches = sorted(glob.glob('results/full_sim/sim_hybrid_batch*.parquet'))
    all_dfs = [pd.read_parquet(f) for f in all_batches]
    combined_hybrid = pd.concat(all_dfs, ignore_index=True)
    combined_hybrid.to_parquet('results/full_sim/sim_hybrid_all.parquet')

    for i in range(8):  # 8 batches × 10k = 80k total
        np.random.seed(1234 + i)  # Unique seed per batch
        output_name = f'sim_rdonsker_batch{i}.parquet'
        print(f"Running batch {i+1}/8 with seed {1234 + i}")
        run_simulation_batch(
            method='rdonsker',
            n_param=10000,
            n_paths=60000,
            n_steps=252,
            T=2.0,
            output_dir='results/full_sim',
            output_file=output_name
        )

    all_batches = sorted(glob.glob('results/full_sim/sim_rdonsker_batch*.parquet'))
    all_dfs = [pd.read_parquet(f) for f in all_batches]
    combined_donsker = pd.concat(all_dfs, ignore_index=True)
    combined_donsker.to_parquet('results/full_sim/sim_rdonsker_all.parquet')
    

    # np.random.seed(42)
    # # Variance decay simulations
    # output_dir = 'results/var_decay'
    # os.makedirs(output_dir, exist_ok=True)

    # for n_paths in [500, 1000, 2000, 5000]:
    #     run_simulation_batch(method='hybrid', n_param=250, n_paths=n_paths, output_dir=output_dir)
    #     # Rename output file to capture n_paths
    #     old_path = os.path.join(output_dir, f'simulated_paths_hybrid.parquet')
    #     new_path = os.path.join(output_dir, f'sim_hybrid_npaths{n_paths}.parquet')
    #     os.rename(old_path, new_path)

    # # Analyze and visualize variance decay
    # analyze_variance_decay(path_root=output_dir)

    # for n_paths in [500, 1000, 2000, 5000]:
    #     run_simulation_batch(method='rdonsker', n_param=250, n_paths=n_paths, output_dir=output_dir)
    #     # Rename output file to capture n_paths
    #     old_path = os.path.join(output_dir, f'simulated_paths_rdonsker.parquet')
    #     new_path = os.path.join(output_dir, f'sim_rdonsker_npaths{n_paths}.parquet')
    #     os.rename(old_path, new_path)

    # analyze_variance_decay(method='rdonsker', path_root=output_dir)






    
    # # # example usage # rdonsker 
    # # start_time_donsker = time.time()
    # # model_rdonsker = rBergomi(n=252, N=10000, T=1.0, a=-0.4, method='rdonsker')
    # # dW1 = model_rdonsker.dW1()
    # # Y_donsker = model_rdonsker.Y(dW1)
    # # dW2 = model_rdonsker.dW2()
    # # dB = model_rdonsker.dB(dW1, dW2, rho=0.0)
    # # xi_curve, t_grid, xi_pieces = generate_piecewise_forward_variance(T=model_rdonsker.T, n=model_rdonsker.n)
    # # V_donsker = model_rdonsker.V(Y_donsker, xi_curve)
    # # S_donsker = model_rdonsker.S(V_donsker, dB)
    # # diff_donsker = time.time() - start_time_donsker
    # # print(f"%--------------- time: %s seconds ---------------%{diff_donsker}")

    # # start_time_hybrid = time.time()
    # # model_hybrid = rBergomi(n=252, N=10000, T=1.0, a=-0.4, method='hybrid')
    # # Y_hybrid = model_hybrid.Y(dW1)
    # # V_hybrid = model_rdonsker.V(Y_hybrid, xi_curve)
    # # S_hybrid = model_rdonsker.S(V_hybrid, dB)
    # # diff_hybrid = time.time() - start_time_hybrid
    # # print(f"%--------------- time: %s seconds ---------------%{diff_hybrid}")
    # # font = FontProperties(family='Times New Roman')

    # # # diff = time.time() - start_time
    # # # print(f"%--------------- time: %s seconds ---------------%{time.time() - start_time}")
    # # # print("Generated paths for rBergomi model:")
    # # # print(S)
    # #  # asset path hybrid 
    # # plt.figure(figsize=(10, 6))
    # # plt.title('rBergomi Asset Paths Hybrid', fontproperties=font, fontsize=14)
    # # plt.xlabel('Time', fontproperties=font)
    # # plt.ylabel('Price', fontproperties=font)
    # # plt.text(0.5, 0.95, f'Time taken: {diff_hybrid} seconds', 
    # #     horizontalalignment='left',
    # #     fontproperties=font,
    # #     verticalalignment='bottom',
    # #     transform=plt.gca().transAxes)

    # # for i in range(model_hybrid.N):
    # #     plt.plot(model_hybrid.t[0], S_hybrid[i], label=f'rBergomi Path{i}')
    
    # # # variance path hybrid

    # # plt.figure(figsize=(10, 6))
    # # plt.title('rBergomi Variance Paths Hybrid', fontproperties=font, fontsize=14)
    # # plt.xlabel('Time', fontproperties=font)
    # # plt.ylabel('Var', fontproperties=font)
    # # plt.text(0.5, 0.95, f'Time taken: {diff_hybrid} seconds', 
    # #     horizontalalignment='left',
    # #     fontproperties=font,
    # #     verticalalignment='bottom',
    # #     transform=plt.gca().transAxes)

    # # for i in range(model_hybrid.N):
    # #     plt.plot(model_hybrid.t[0], V_hybrid[i], label=f'rBergomi Path{i}')
        
    # # plt.figure(figsize=(10, 6))
    # # plt.title('rBergomi Asset Paths rdonsker', fontproperties=font, fontsize=14)
    # # plt.xlabel('Time', fontproperties=font)
    # # plt.ylabel('Price', fontproperties=font)
    # # plt.text(0.5, 0.95, f'Time taken: {diff_donsker} seconds',
    # #          horizontalalignment='left',
    # #          fontproperties=font,
    # #          verticalalignment='bottom',
    # #          transform=plt.gca().transAxes)

    # # for i in range(model_rdonsker.N):
    # #     plt.plot(model_rdonsker.t[0], S_donsker[i], label=f'rBergomi Path{i}')
    
    # # # variance path donsker

    # # plt.figure(figsize=(10, 6))
    # # plt.title('rBergomi Variance Paths rDonsker', fontproperties=font, fontsize=14)
    # # plt.xlabel('Time', fontproperties=font)
    # # plt.ylabel('Var', fontproperties=font)
    # # plt.text(0.5, 0.95, f'Time taken: {diff_hybrid} seconds', 
    # #     horizontalalignment='left',
    # #     fontproperties=font,
    # #     verticalalignment='bottom',
    # #     transform=plt.gca().transAxes)

    # # for i in range(model_hybrid.N):
    # #     plt.plot(model_hybrid.t[0], V_donsker[i], label=f'rBergomi Path{i}')
    
    
    # # plt.figure(figsize=(10, 6))
    # # plt.plot(xi_curve, label='Forward Variance Curve')
    # # plt.title(f'Piecewise Forward Variance Curve, T={model_rdonsker.T}, n={model_rdonsker.n}', fontsize=14, fontproperties=font)
    # # plt.show()

 