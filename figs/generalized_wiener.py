import numpy as np
import matplotlib.pyplot as plt




T = 1.0         
N = 500         
dt = T / N      
a = 1.5        
b = 1.3       

# Time vector
t = np.linspace(0, T, N)

# Wiener process (Brownian motion)
dZ = np.random.normal(0, np.sqrt(dt), size=N)
Z = np.cumsum(dZ)

# Generalized Wiener process: dx = a*dt + b*dZ
X = np.cumsum(a * dt + b * dZ)

# Deterministic trend: dx = a*dt
X_deterministic = np.cumsum(a * dt * np.ones(N))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(t, X, label="Generalized Wiener process: $dx = a \\, dt + b \\, dz$", color='black')
plt.plot(t, Z, label="Wiener process: $dz$", color='black', alpha=0.3)
plt.plot(t, X_deterministic, label="Deterministic trend: $dx = a \\, dt$", linestyle='--', color='black')
plt.xlabel("Time")
plt.ylabel("Value of variable, $x$")
plt.title("Generalized Wiener Process with $a = 1.5$ and $b = 1.3$")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

plt.savefig('generalized_wiener_process.png')