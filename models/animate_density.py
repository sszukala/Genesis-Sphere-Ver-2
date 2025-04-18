"""
Animated Simulations of the Genesis-Sphere Model

This script demonstrates how to create animations showing the behavior
of the Genesis-Sphere model as parameters change over time.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from genesis_model import GenesisSphereModel


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


def create_parameter_evolution_animation(param_name, start_value, end_value, 
                                        frames=100, fps=30, 
                                        output_file="../output/model_animations/param_evolution.mp4"):
    """
    Create an animation showing how a model parameter affects the system behavior.
    
    Parameters:
    -----------
    param_name : str
        Parameter to animate ('alpha', 'beta', 'omega', or 'epsilon')
    start_value : float
        Starting value for the parameter
    end_value : float
        Ending value for the parameter
    frames : int
        Number of frames in the animation
    fps : int
        Frames per second for the output video
    output_file : str
        File path for the output video
    """
    # Make sure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create a frames directory for fallback
    frames_dir = os.path.join(output_dir, f"{param_name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize the model with default parameters
    model = GenesisSphereModel()
    
    # Set up the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Set up time domain
    t = np.linspace(-12, 12, 1000)
    
    # Create empty lines for each plot
    lines = []
    for ax in axes:
        line, = ax.plot([], [], lw=2)
        lines.append(line)
        ax.grid(True)
    
    # Configure axes
    axes[0].set_xlim(-12, 12)
    axes[0].set_ylim(0, 3)
    axes[0].set_title("Space-Time Density ρ(t)")
    
    axes[1].set_xlim(-12, 12)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Temporal Flow Tf(t)")
    
    axes[2].set_xlim(-12, 12)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("Modulated Velocity v(t)")
    
    axes[3].set_xlim(-12, 12)
    axes[3].set_ylim(0, 3)
    axes[3].set_title("Modulated Pressure p(t)")
    
    # Parameter display text
    param_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12)
    
    # Title for the figure
    fig.suptitle(f"Genesis-Sphere Model: {param_name} Evolution", fontsize=14)
    
    def init():
        """Initialize the animation"""
        for line in lines:
            line.set_data([], [])
        param_text.set_text("")
        return lines + [param_text]
    
    def update(frame):
        """Update function for animation"""
        # Calculate current parameter value based on frame
        param_value = start_value + (end_value - start_value) * frame / (frames - 1)
        
        # Update the model parameter
        setattr(model, param_name, param_value)
        
        # Calculate model values
        results = model.evaluate_all(t)
        
        # Update each line
        lines[0].set_data(t, results['density'])
        lines[1].set_data(t, results['temporal_flow'])
        lines[2].set_data(t, results['velocity'])
        lines[3].set_data(t, results['pressure'])
        
        # Update parameter display
        param_text.set_text(f"{param_name} = {param_value:.4f}")
        
        # For fallback mode, also save the frame
        if not has_ffmpeg:
            # Save this frame
            frame_file = os.path.join(frames_dir, f'frame_{frame:03d}.png')
            fig.savefig(frame_file, dpi=150)
            if frame % 10 == 0:
                print(f"Saved frame {frame+1}/{frames}")
        
        return lines + [param_text]
    
    # Create animation
    animation = FuncAnimation(fig, update, frames=frames, init_func=init, 
                             blit=True, interval=1000/fps)
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    
    if has_ffmpeg:
        try:
            # Save animation as MP4
            animation.save(output_file, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Animation saved to {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Falling back to saving individual frames...")
            has_ffmpeg = False
    
    if not has_ffmpeg:
        # Save frames individually if FFmpeg is not available
        for i in range(frames):
            update(i)
        print(f"Individual frames saved to {frames_dir}")
        print("To create a video, install FFmpeg:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add the bin directory to your PATH environment variable")
        print("3. Restart your terminal/command prompt after installation")
    
    plt.close(fig)


def create_time_window_animation(output_file="../output/model_animations/time_window.mp4",
                                frames=150, fps=30, window_size=6):
    """
    Create an animation showing a sliding window view of the model functions over time.
    
    Parameters:
    -----------
    output_file : str
        File path for the output video
    frames : int
        Number of frames in the animation
    fps : int
        Frames per second for the output video
    window_size : float
        Size of the time window to display at each frame
    """
    # Make sure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create a frames directory for fallback
    frames_dir = os.path.join(output_dir, "time_window_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize the model
    model = GenesisSphereModel()
    
    # Full time domain
    t_full = np.linspace(-15, 15, 2000)
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Configure axes
    titles = ["Projection & Expansion", "Space-Time Density", 
             "Temporal Flow", "Derived Quantities"]
    
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(-window_size/2, window_size/2)
    
    axes[0].set_ylim(0, 3)
    axes[1].set_ylim(0, 3)
    axes[2].set_ylim(0, 1.1)
    axes[3].set_ylim(0, 3)
    
    # Pre-calculate all model values for efficiency
    results = model.evaluate_all(t_full)
    
    # Set up the lines
    lines = []
    lines.append(axes[0].plot([], [], label="S(t) - Projection")[0])
    lines.append(axes[0].plot([], [], label="D(t) - Expansion")[0])
    lines.append(axes[1].plot([], [], color='darkred', label="ρ(t) - Time-Density")[0])
    lines.append(axes[2].plot([], [], color='blue', label="Tf(t) - Temporal Flow")[0])
    lines.append(axes[3].plot([], [], color='green', label="v(t) - Velocity")[0])
    lines.append(axes[3].plot([], [], color='purple', label="p(t) - Pressure")[0])
    
    # Add legends
    for ax in axes:
        ax.legend()
    
    # Current time indicator
    time_indicators = []
    for ax in axes:
        indicator = ax.axvline(x=0, color='r', linestyle='-', alpha=0.5)
        time_indicators.append(indicator)
    
    # Time display text
    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12)
    
    # Title
    fig.suptitle("Genesis-Sphere Model: Time Evolution", fontsize=14)
    
    def init():
        """Initialize the animation"""
        for line in lines:
            line.set_data([], [])
        time_text.set_text("")
        return lines + time_indicators + [time_text]
    
    def update(frame):
        """Update function for animation"""
        # Calculate current center time based on frame
        total_range = t_full[-1] - t_full[0] - window_size
        center_time = t_full[0] + window_size/2 + total_range * frame / (frames - 1)
        
        # Calculate window boundaries
        t_min = center_time - window_size/2
        t_max = center_time + window_size/2
        
        # Find indices for the window
        indices = (t_full >= t_min) & (t_full <= t_max)
        t_window = t_full[indices]
        
        # Update each line
        lines[0].set_data(t_window, results['projection'][indices])
        lines[1].set_data(t_window, results['expansion'][indices])
        lines[2].set_data(t_window, results['density'][indices])
        lines[3].set_data(t_window, results['temporal_flow'][indices])
        lines[4].set_data(t_window, results['velocity'][indices])
        lines[5].set_data(t_window, results['pressure'][indices])
        
        # Update time indicators
        for indicator in time_indicators:
            indicator.set_xdata([center_time, center_time])
        
        # Update time display
        time_text.set_text(f"Time: {center_time:.2f}")
        
        # Update axis limits to follow the window
        for ax in axes:
            ax.set_xlim(t_min, t_max)
        
        # For fallback mode, also save the frame
        if not has_ffmpeg:
            # Save this frame
            frame_file = os.path.join(frames_dir, f'frame_{frame:03d}.png')
            fig.savefig(frame_file, dpi=150)
            if frame % 15 == 0:
                print(f"Saved frame {frame+1}/{frames}")
        
        return lines + time_indicators + [time_text]
    
    # Create animation
    animation = FuncAnimation(fig, update, frames=frames, init_func=init, 
                             blit=True, interval=1000/fps)
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    
    if has_ffmpeg:
        try:
            # Save animation as MP4
            animation.save(output_file, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Animation saved to {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Falling back to saving individual frames...")
            has_ffmpeg = False
    
    if not has_ffmpeg:
        # Save frames individually if FFmpeg is not available
        for i in range(frames):
            update(i)
        print(f"Individual frames saved to {frames_dir}")
        print("To create a video, install FFmpeg:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add the bin directory to your PATH environment variable")
        print("3. Restart your terminal/command prompt after installation")
    
    plt.close(fig)


def create_3d_density_animation(output_file="../output/model_animations/3d_density_evolution.mp4",
                               frames=120, fps=30):
    """
    Create a 3D animation showing how space-time density varies with parameter changes.
    
    Parameters:
    -----------
    output_file : str
        File path for the output video
    frames : int
        Number of frames in the animation
    fps : int
        Frames per second for the output video
    """
    # Make sure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create a frames directory for fallback
    frames_dir = os.path.join(output_dir, "3d_density_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Import 3D specific modules
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    
    # Initialize model
    model = GenesisSphereModel()
    
    # Create meshgrid for time and omega
    t = np.linspace(-10, 10, 100)
    w = np.linspace(0.5, 2, 100)
    T, W = np.meshgrid(t, w)
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    
    # Generate each frame
    for i in range(frames):
        # Create a new figure for each frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate frame-specific parameters
        # Animation strategy: Cycle through parameter changes and view angles
        cycle_position = i % 60
        
        if cycle_position < 30:
            # First half: Vary alpha
            alpha_val = 0.01 + 0.09 * (cycle_position / 30)
            model.alpha = alpha_val
            param_str = f"α = {alpha_val:.3f}"
        else:
            # Second half: Vary omega
            omega_val = 0.5 + 1.5 * ((cycle_position - 30) / 30)
            model.omega = omega_val
            param_str = f"ω = {omega_val:.3f}"
        
        # Calculate new density
        S = 1 / (1 + np.sin(model.omega * T)**2)
        D = 1 + model.alpha * T**2
        rho = S * D
        
        # Create new surface
        surf = ax.plot_surface(T, W, rho, cmap=cm.viridis, alpha=0.8,
                              linewidth=0, antialiased=True)
        
        # Set labels
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Frequency (ω)')
        ax.set_zlabel('Space-Time Density (ρ)')
        ax.set_title(f'Evolution of Space-Time Density ({param_str})')
        
        # Update view angle for rotation effect
        elevation = 30
        azimuth = i * 3 % 360
        ax.view_init(elevation, azimuth)
        
        # Set consistent axis limits
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(w.min(), w.max())
        ax.set_zlim(0, 3)
        
        # Save the frame for fallback or to combine into animation
        frame_file = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_file, dpi=150)
        if i % 10 == 0:
            print(f"Saved frame {i+1}/{frames}")
        
        plt.close(fig)
    
    if has_ffmpeg:
        try:
            # Compile frames into a video using FFmpeg directly
            import subprocess
            
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite existing files
                '-framerate', str(fps),  # Frames per second
                '-i', os.path.join(frames_dir, 'frame_%03d.png'),
                '-c:v', 'libx264',
                '-profile:v', 'high',
                '-crf', '20',  # Quality (lower is better)
                '-pix_fmt', 'yuv420p',
                output_file
            ]
            
            subprocess.run(cmd, check=True)
            print(f"3D animation saved to {output_file}")
        except Exception as e:
            print(f"Error creating video with FFmpeg: {e}")
            print(f"Individual frames can be found in {frames_dir}")
    else:
        print(f"FFmpeg not found. Individual frames saved to {frames_dir}")
        print("To create a video, install FFmpeg:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add the bin directory to your PATH environment variable")
        print("3. Restart your terminal/command prompt after installation")


def main():
    """Run all animation generation functions"""
    print("Genesis-Sphere Animation Generator")
    print("=================================")
    
    # Check FFmpeg availability first
    has_ffmpeg = check_ffmpeg()
    if not has_ffmpeg:
        print("FFmpeg not found. Animations will be saved as individual frames instead.")
        print("To install FFmpeg (recommended for MP4 creation):")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add the bin directory to your PATH environment variable")
        print("3. Restart your terminal/command prompt after installation")
        print("\nContinuing with fallback mode...\n")
    
    # Create output directory
    os.makedirs("../output/model_animations", exist_ok=True)
    
    # Generate parameter evolution animations
    create_parameter_evolution_animation('alpha', 0.01, 0.1, 
                                        output_file="../output/model_animations/alpha_evolution.mp4")
    
    create_parameter_evolution_animation('beta', 0.2, 2.0,
                                        output_file="../output/model_animations/beta_evolution.mp4")
    
    create_parameter_evolution_animation('omega', 0.5, 2.5,
                                        output_file="../output/model_animations/omega_evolution.mp4")
    
    # Generate time window animation
    create_time_window_animation()
    
    # Generate 3D density animation
    create_3d_density_animation()
    
    print("All animations generated successfully!")


if __name__ == "__main__":
    main()
