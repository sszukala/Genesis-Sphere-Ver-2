import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import subprocess

def check_ffmpeg():
    """Check if FFmpeg is available in the system or common installation locations"""
    if shutil.which('ffmpeg'):
        return True
    
    # Common installation locations on Windows
    common_locations = [
        os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'
    ]
    
    for location in common_locations:
        if os.path.exists(location):
            print(f"FFmpeg found at {location} but not in system PATH")
            print("Temporarily adding to PATH for this session...")
            os.environ['PATH'] += os.pathsep + os.path.dirname(location)
            return True
    
    return False

def generate_animation():
    """Generate 4D visualization animation with evolving color mapping and rotation"""
    print("Generating 4D visualization animation...")
    
    # Parameters
    alpha = 0.02
    omega = 1
    beta = 0.8
    epsilon = 0.1
    v0 = 1.0
    
    # Create output directories
    output_dir = os.path.join('..', 'output')
    frames_dir = os.path.join(output_dir, '4d_viz_frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Time domain
    t = np.linspace(-12, 12, 500)  # Using fewer points for better scatter plot performance
    
    # Calculate basic functions
    S = 1 / (1 + np.sin(omega * t)**2)
    D = 1 + alpha * t**2
    rho = S * D
    Tf = 1 / (1 + beta * (np.abs(t) + epsilon))
    velocity = v0 * Tf
    
    # Total number of frames
    frames = 120
    
    # Generate each frame
    for i in range(frames):
        # Create a new figure for each frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Vary the pressure parameter for each frame to create a time-evolving 4D visualization
        # This creates a "wave" of pressure changes moving through the system
        p0_factor = 1.0 + 0.5 * np.sin(i * 2 * np.pi / frames)  # Pressure oscillates between 0.5 and 1.5
        
        # Calculate pressure with the varying factor
        pressure = p0_factor * rho
        
        # Plot 3D points with color representing pressure (4th dimension)
        scatter = ax.scatter(t[::2], rho[::2], velocity[::2], 
                            c=pressure[::2], cmap=cm.plasma,
                            s=10, alpha=0.8)
        
        # Rotate view for each frame
        elevation = 30
        azimuth = i * 3 % 360  # Rotate full circle
        ax.view_init(elevation, azimuth)
        
        # Set consistent axis limits
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(rho.min(), rho.max())
        ax.set_zlim(velocity.min(), velocity.max())
        
        # Update the colorbar range to show the pressure oscillation
        current_max = pressure.max()
        current_min = pressure.min()
        scatter.set_clim(current_min, current_max)
        
        # Add labels and colorbar
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Space-Time Density (ρ)')
        ax.set_zlabel('Velocity (v)')
        ax.set_title(f'4D Visualization: Pressure Wave (p0 = {p0_factor:.2f})')
        fig.colorbar(scatter, ax=ax, label='Pressure (p)')
        
        # Save this frame
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_file, dpi=150)
        plt.close(fig)
        
        print(f"Saved frame {i+1}/{frames}")
    
    # Check if FFmpeg is available
    if check_ffmpeg():
        try:
            # Compile the frames into a video using FFmpeg
            output_file = os.path.join(output_dir, '4d_visualization_animation.mp4')
            
            # FFmpeg command to create a video from the frames
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite existing files
                '-framerate', '30',  # Frames per second
                '-i', os.path.join(frames_dir, 'frame_%03d.png'),
                '-c:v', 'libx264',
                '-profile:v', 'high',
                '-crf', '20',  # Quality (lower is better)
                '-pix_fmt', 'yuv420p',
                output_file
            ]
            
            print(f"Running FFmpeg to create video: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            print(f"Animation saved successfully to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error creating video with FFmpeg: {e}")
            print(f"Individual frames can be found in {frames_dir}")
    else:
        print("FFmpeg not found. Cannot create video.")
        print(f"Individual frames can be found in {frames_dir}")
    
    return frames_dir

if __name__ == "__main__":
    print("Generating 4D visualization animation...")
    output = generate_animation()
    print(f"Animation process completed. Result available at: {output}")
