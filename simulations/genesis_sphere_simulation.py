import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.02
omega = 1
beta = 0.8
epsilon = 0.1
v0 = 1.0
p0 = 1.0

# Time domain
t = np.linspace(-12, 12, 1000)

# Core functions
S = 1 / (1 + np.sin(omega * t)**2)
D = 1 + alpha * t**2
rho = S * D
Tf = 1 / (1 + beta * (np.abs(t) + epsilon))

# Derived functions
velocity = v0 * Tf
pressure = p0 * rho

# Plot everything
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, S, label="S(t) - Projection")
plt.plot(t, D, label="D(t) - Expansion")
plt.title("Projection & Expansion")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, rho, color='darkred', label="ρ(t) - Time-Density")
plt.title("Space-Time Density")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, Tf, color='blue', label="Tf(t) - Temporal Flow")
plt.title("Temporal Flow Modulation")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, velocity, color='green', label="v(t) - Modulated Velocity")
plt.plot(t, pressure, color='purple', label="p(t) - Modulated Pressure")
plt.title("Derived Quantities")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('../docs/simulation_output.png', dpi=300)  # Save a high-quality image
plt.show()
