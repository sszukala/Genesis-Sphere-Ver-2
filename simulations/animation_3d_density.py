import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from animation_helper import generate_animation_with_progress, save_frame

def generate_animation():
    """Generate 3D density surface animation with rotating perspective"""
    print("Generating 3D density surface animation...")
    
    # Parameters
    alpha = 0.02      # Spatial dimension expansion coefficient
    omega = 1         # Base angular frequency
    
    # Create output directories
    output_dir = os.path.join('..', 'output')
    frames_dir = os.path.join(output_dir, 'density_frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create a meshgrid for time and omega
    t = np.linspace(-10, 10, 100)
    w = np.linspace(0.5, 2, 100)
    T, W = np.meshgrid(t, w)
    
    # Calculate density for each point in the grid
    S = 1 / (1 + np.sin(W * T)**2)
    D = 1 + alpha * T**2
    rho = S * D
    
    # Function to generate each frame
    def generate_frame(i, total_frames):
        # Create a new figure for each frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set a new viewing angle that rotates around
        elevation = 30
        azimuth = i * 3  # Rotate 3 degrees per frame
        ax.view_init(elevation, azimuth)
        
        # Plot the surface
        surf = ax.plot_surface(T, W, rho, cmap=cm.viridis, alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Add labels
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Frequency (ω)')
        ax.set_zlabel('Space-Time Density (ρ)')
        ax.set_title('3D Surface: Space-Time Density Variation')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Save the frame
        frame_file = save_frame(i, total_frames, fig, frames_dir)
        plt.close(fig)
        
        return frame_file
    
    # Generate the animation with progress tracking
    output_file = os.path.join(output_dir, '3d_density_animation.mp4')
    return generate_animation_with_progress(frames_dir, generate_frame, output_file)

if __name__ == "__main__":
    print("Generating 3D density surface animation...")
    output = generate_animation()
    print(f"Animation process completed. Result available at: {output}")
