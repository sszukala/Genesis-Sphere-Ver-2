import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_3d_visualization():
    """Generate 3D visualizations of the Genesis-Sphere model"""
    # Parameters
    alpha = 0.02
    omega = 1
    beta = 0.8
    epsilon = 0.1
    v0 = 1.0
    p0 = 1.0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 3D Surface Plot: Time-Density as a surface
    print("Generating 3D surface plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for time and omega 
    t = np.linspace(-10, 10, 100)
    w = np.linspace(0.5, 2, 100)
    T, W = np.meshgrid(t, w)
    
    # Calculate density for each point in the grid
    S = 1 / (1 + np.sin(W * T)**2)
    D = 1 + alpha * T**2
    rho = S * D
    
    # Plot the surface
    surf = ax.plot_surface(T, W, rho, cmap=cm.viridis, alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add labels and colorbar
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Frequency (ω)')
    ax.set_zlabel('Space-Time Density (ρ)')
    ax.set_title('3D Surface: Space-Time Density Variation')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(os.path.join(output_dir, '3d_density_surface.png'), dpi=300)
    print(f"Saved 3D surface plot to {os.path.join(output_dir, '3d_density_surface.png')}")
    plt.close()
    
    # 2. 3D Parametric Curve: Evolution trace in 3D
    print("Generating 3D parametric curve...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time domain
    t = np.linspace(-12, 12, 1000)
    
    # Calculate functions
    S = 1 / (1 + np.sin(omega * t)**2)
    D = 1 + alpha * t**2
    rho = S * D
    Tf = 1 / (1 + beta * (np.abs(t) + epsilon))
    velocity = v0 * Tf
    pressure = p0 * rho
    
    # Plot the 3D parametric curve (time, density, velocity)
    ax.plot(t, rho, velocity, color='blue', linewidth=2)
    
    # Add points at regular intervals for clarity
    stride = 50
    ax.scatter(t[::stride], rho[::stride], velocity[::stride], 
               color='red', s=50, alpha=0.6)
    
    # Add labels
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space-Time Density (ρ)')
    ax.set_zlabel('Velocity (v)')
    ax.set_title('3D Parametric Curve: Evolution of Genesis-Sphere System')
    
    plt.savefig(os.path.join(output_dir, '3d_parametric_curve.png'), dpi=300)
    print(f"Saved 3D parametric curve to {os.path.join(output_dir, '3d_parametric_curve.png')}")
    plt.close()
    
    # 3. 4D Visualization: 3D + Color for the fourth dimension 
    print("Generating 4D visualization (3D + color dimension)...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points with color representing pressure (4th dimension)
    scatter = ax.scatter(t[::5], rho[::5], velocity[::5], 
                        c=pressure[::5], cmap=cm.plasma,
                        s=10, alpha=0.8)
    
    # Add labels and colorbar
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Space-Time Density (ρ)')
    ax.set_zlabel('Velocity (v)')
    ax.set_title('4D Visualization: Color represents Pressure')
    fig.colorbar(scatter, ax=ax, label='Pressure (p)')
    
    plt.savefig(os.path.join(output_dir, '4d_visualization.png'), dpi=300)
    print(f"Saved 4D visualization to {os.path.join(output_dir, '4d_visualization.png')}")
    plt.close()
    
    # 4. 3D Time-Space Folding Visualization
    print("Generating 3D time-space folding visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create time-space grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distortion based on radial distance (r = sqrt(x^2 + y^2))
    R = np.sqrt(X**2 + Y**2)
    
    # Apply temporal flow distortion to Z coordinate
    Z = np.sin(R) / (1 + beta * (R + epsilon))
    
    # Plot the warped surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add labels and colorbar
    ax.set_xlabel('X-Space')
    ax.set_ylabel('Y-Space')
    ax.set_zlabel('Temporal Distortion')
    ax.set_title('3D Visualization: Space-Time Folding near Singularity')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(os.path.join(output_dir, '3d_spacetime_folding.png'), dpi=300)
    print(f"Saved 3D space-time folding visualization to {os.path.join(output_dir, '3d_spacetime_folding.png')}")
    plt.close()
    
    print("All 3D/4D visualizations completed successfully!")
    return output_dir

if __name__ == "__main__":
    print("Generating Genesis-Sphere 3D/4D visualizations...")
    output_dir = generate_3d_visualization()
    print(f"All visualization files saved to {output_dir}")
