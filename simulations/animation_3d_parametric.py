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
    """Generate 3D parametric curve animation showing evolution and rotation"""
    print("Generating 3D parametric curve animation...")
    
    # Parameters
    alpha = 0.02
    omega = 1
    beta = 0.8
    epsilon = 0.1
    v0 = 1.0
    p0 = 1.0
    
    # Create output directories
    output_dir = os.path.join('..', 'output')
    frames_dir = os.path.join(output_dir, 'parametric_frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Time domain
    t = np.linspace(-12, 12, 1000)
    
    # Calculate functions
    S = 1 / (1 + np.sin(omega * t)**2)
    D = 1 + alpha * t**2
    rho = S * D
    Tf = 1 / (1 + beta * (np.abs(t) + epsilon))
    velocity = v0 * Tf
    pressure = p0 * rho
    
    # Total number of frames
    frames = 120
    
    # Generate each frame
    for i in range(frames):
        # Create a new figure for each frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate how much of the curve to show in this frame
        # First half of animation: gradually reveal the curve
        # Second half: rotate the fully revealed curve
        if i < frames // 2:
            # Gradually reveal the curve
            points_to_show = int((i + 1) * len(t) / (frames // 2))
            
            # Plot the revealed portion of the curve
            ax.plot(t[:points_to_show], rho[:points_to_show], velocity[:points_to_show], 
                   color='blue', linewidth=2)
            
            # Add points at regular intervals for clarity
            stride = max(1, points_to_show // 20)
            ax.scatter(t[:points_to_show:stride], rho[:points_to_show:stride], 
                      velocity[:points_to_show:stride], color='red', s=50, alpha=0.6)
            
            # Keep the view angle fixed during reveal
            ax.view_init(30, 45)
        else:
            # Rotate the full curve for the second half of animation
            ax.plot(t, rho, velocity, color='blue', linewidth=2)
            
            # Add points at regular intervals for clarity
            stride = len(t) // 20
            ax.scatter(t[::stride], rho[::stride], velocity[::stride], 
                      color='red', s=50, alpha=0.6)
            
            # Rotate the view
            rotation_frame = i - frames // 2
            elevation = 30
            azimuth = 45 + rotation_frame * 3  # Start at 45 degrees and rotate
            ax.view_init(elevation, azimuth)
        
        # Set axis limits to ensure consistent view
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(rho.min(), rho.max())
        ax.set_zlim(velocity.min(), velocity.max())
        
        # Add labels
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Space-Time Density (ρ)')
        ax.set_zlabel('Velocity (v)')
        ax.set_title('3D Parametric Curve: Evolution of Genesis-Sphere System')
        
        # Save this frame
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_file, dpi=150)
        plt.close(fig)
        
        print(f"Saved frame {i+1}/{frames}")
    
    # Check if FFmpeg is available
    if check_ffmpeg():
        try:
            # Compile the frames into a video using FFmpeg
            output_file = os.path.join(output_dir, '3d_parametric_animation.mp4')
            
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
    print("Generating 3D parametric curve animation...")
    output = generate_animation()
    print(f"Animation process completed. Result available at: {output}")
