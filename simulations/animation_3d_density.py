import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
import subprocess
from matplotlib.animation import FuncAnimation

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
    
    # Create a figure for the animation frames
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial surface plot
    surf = ax.plot_surface(T, W, rho, cmap=cm.viridis, alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add labels and colorbar
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Frequency (ω)')
    ax.set_zlabel('Space-Time Density (ρ)')
    ax.set_title('3D Surface: Space-Time Density Variation')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Function to update the plot for each animation frame
    def update(frame):
        # Clear previous plot
        ax.clear()
        
        # Set a new viewing angle that rotates around
        elevation = 30
        azimuth = frame * 3  # Rotate 3 degrees per frame
        ax.view_init(elevation, azimuth)
        
        # Re-plot the surface
        surf = ax.plot_surface(T, W, rho, cmap=cm.viridis, alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Add labels again (since we cleared the axes)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Frequency (ω)')
        ax.set_zlabel('Space-Time Density (ρ)')
        ax.set_title('3D Surface: Space-Time Density Variation')
        
        return [surf]
    
    # Total number of frames (120 = 360 degrees rotation at 3 degrees per frame)
    frames = 120
    
    # Generate and save each frame
    for i in range(frames):
        # Update the figure for this frame
        update(i)
        
        # Save the frame
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_file, dpi=150)
        print(f"Saved frame {i+1}/{frames}")
    
    plt.close(fig)
    
    # Check if FFmpeg is available
    if check_ffmpeg():
        try:
            # Compile the frames into a video using FFmpeg
            output_file = os.path.join(output_dir, '3d_density_animation.mp4')
            
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
    print("Generating 3D density surface animation...")
    output = generate_animation()
    print(f"Animation process completed. Result available at: {output}")
