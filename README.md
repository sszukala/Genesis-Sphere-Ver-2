# 🌌 Genesis-Sphere: A Framework for Space-Time Density and Temporal Flow

**Author**: Shannon Szukala  
**Date**: April 17, 2025

---

## 🧠 Overview

**Genesis-Sphere** is a theoretical framework that extends general relativity by introducing two novel concepts:

- **Time-Density Geometry**: A model of space-time density that evolves based on sinusoidal and quadratic scaling.
- **Temporal Flow Ratio**: A mathematical formulation to simulate how time slows down or normalizes near singularities.

The goal is to provide a more accessible and visualizable way to study cosmic events like the **Big Bang**, **black holes**, and **cyclic universes**.

---

## 📐 Key Functions

### 1. Time-Density Geometry Function

$$
\rho(t) = \frac{1}{1 + \sin^2(\omega t)} \cdot (1 + \alpha t^2)
$$

- **Sinusoidal Projection Term**: Smooths density behavior over time.
- **Dimension Expansion Term**: Models growth of spatial complexity.

---

### 2. Temporal Flow Ratio Function

$$
Tf(t) = \frac{1}{1 + \beta(|t| + \epsilon)}
$$

- Near $t = 0$, this function sharply reduces, mimicking **time dilation near singularities**.
- As $t \rightarrow \infty$, it smoothly approaches 1, simulating **normalized time flow**.

---

### 3. Derived Modulations

- **Modulated Velocity**  
  $$
  v(t) = v_0 \cdot Tf(t)
  $$

- **Modulated Pressure**  
  $$
  p(t) = p_0 \cdot \rho(t)
  $$

These scale velocity and pressure over time relative to time-density and flow modulation.

---

## 📊 Sample Visualization (Python)

Use this Python script to plot the behavior of $\rho(t)$ and $Tf(t)$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.02
omega = 1
beta = 0.8
epsilon = 0.1
t = np.linspace(-12, 12, 500)

# Functions
S = 1 / (1 + np.sin(omega * t)**2)
D = 1 + alpha * t**2
rho = S * D
Tf = 1 / (1 + beta * (np.abs(t) + epsilon))

# Plot
plt.plot(t, rho, label="ρ(t) - Time-Density")
plt.plot(t, Tf, label="Tf(t) - Temporal Flow")
plt.title("Genesis-Sphere Functions")
plt.xlabel("Time (t)")
plt.ylabel("Function Value")
plt.legend()
plt.grid(True)
plt.show()
```

## 🎬 Animation Features

Genesis-Sphere includes animation capabilities to visualize how the model's key functions evolve over time:

1. **Static Visualization**: Basic plots showing function relationships using `genesis_sphere_simulation.py`

2. **Video Animation**: Dynamic visualization of function evolution using `genesis_sphere_animation_fallback.py`

3. **Frame Sequences**: Individual image frames showing the progression over time

Note: Video animation requires FFmpeg to be installed. The framework will automatically check for FFmpeg and offers an alternative frame-by-frame visualization if not found.

```bash
# Run from the simulations directory
python genesis_sphere_animation_fallback.py
```

## 🌐 3D/4D Visualizations

The framework now includes advanced 3D and 4D visualizations to provide deeper insights into the Genesis-Sphere model:

1. **3D Surface Plot**: Shows space-time density variation across time and frequency parameters
   
2. **3D Parametric Curve**: Traces the evolution of the system in 3D space (time, density, velocity)
   
3. **4D Visualization**: Uses 3D coordinates with color as the fourth dimension to represent pressure
   
4. **Space-Time Folding**: Visualizes how space-time might fold near a singularity

```bash
# Generate all 3D/4D visualizations
python simulations/genesis_sphere_3d_visualization.py
```

![3D Surface Example](output/3d_density_surface.png)

---

## 🔗 Getting Started

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run the simulations from the `simulations` folder

For full setup instructions, see the `roadmap.md` file.
