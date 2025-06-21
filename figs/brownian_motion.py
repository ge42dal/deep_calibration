import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


T = 1.0  
mu = 0
sigma = 1

def simulate_brownian(T, N):
    dt = T / N
    t = np.linspace(0, T, N+1)
    dz = np.random.normal(loc=0, scale=np.sqrt(dt), size=N)
    z = np.concatenate([[0], np.cumsum(dz)])
    return t, z

# two different time step sizes
t1, W1 = simulate_brownian(T, 50)   # relatively large delta
t2, W2 = simulate_brownian(T, 1000) # smaller delta


fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t1, W1, color='black')
axes[0].set_ylabel('z')
axes[0].set_title('Relatively large value of $\Delta t$')

axes[1].plot(t2, W2, color='black')
axes[1].set_ylabel('z')
axes[1].set_xlabel('t')
axes[1].set_title('Smaller value of $\Delta t$')

plt.tight_layout()
# plt.show()

