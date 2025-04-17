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
    """Generate animation showing space-time folding with changing beta parameter"""
    print("Generating space-time folding animation...")
    
    # Base parameters
    epsilon = 0.1
    
    # Create output directories
    output_dir = os.path.join('..', 'output')
    frames_dir = os.path.join(output_dir, 'folding_frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create time-space grid (static for all frames)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)  # Radial distance
    
    # Total number of frames
    frames = 120
    
    # Generate each frame with changing beta and view angle
    for i in range(frames):
        # Create a new figure for each frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Vary the beta parameter to show different levels of "folding"
        # Start with a small value and gradually increase to show more dramatic folding
        beta_factor = 0.2 + (2.0 * i / frames)  # Range from 0.2 to 2.2
        
        # Apply temporal flow distortion to Z coordinate using current beta
        Z = np.sin(R) / (1 + beta_factor * (R + epsilon))
        
        # Plot the warped surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8,
                               linewidth=0, antialiased=True)
        
        # Rotate view slightly for each frame
        elevation = 30
        azimuth = i * 3 % 360  # Rotate full circle
        ax.view_init(elevation, azimuth)
        
        # Set consistent axis limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-1, 1)
        
        # Add labels and colorbar
        ax.set_xlabel('X-Space')
        ax.set_ylabel('Y-Space')
        ax.set_zlabel('Temporal Distortion')
        ax.set_title(f'Space-Time Folding (β = {beta_factor:.2f})')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Save this frame
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_file, dpi=150)
        plt.close(fig)
        
        print(f"Saved frame {i+1}/{frames}")
    
    # Check if FFmpeg is available
    if check_ffmpeg():
        try:
            # Compile the frames into a video using FFmpeg
            output_file = os.path.join(output_dir, 'spacetime_folding_animation.mp4')
            
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
    print("Generating space-time folding animation...")
    output = generate_animation()
    print(f"Animation process completed. Result available at: {output}")
