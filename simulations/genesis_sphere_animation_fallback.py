import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import shutil
from pathlib import Path

def create_animation():
    # Parameters
    alpha = 0.02
    omega = 1
    beta = 0.8
    epsilon = 0.1
    v0 = 1.0
    p0 = 1.0

    # Time domain for full calculation
    t_full = np.linspace(-12, 12, 1000)
    
    # Core functions calculated over full range
    S_full = 1 / (1 + np.sin(omega * t_full)**2)
    D_full = 1 + alpha * t_full**2
    rho_full = S_full * D_full
    Tf_full = 1 / (1 + beta * (np.abs(t_full) + epsilon))
    velocity_full = v0 * Tf_full
    pressure_full = p0 * rho_full

    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Genesis-Sphere Dynamic Visualization', fontsize=16)
    
    # Initial empty plots
    proj_exp_plot, = axs[0, 0].plot([], [], label="S(t) - Projection")
    proj_exp_plot2, = axs[0, 0].plot([], [], label="D(t) - Expansion")
    axs[0, 0].set_title("Projection & Expansion")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].set_xlim(-12, 12)
    axs[0, 0].set_ylim(0, 3)
    
    density_plot, = axs[0, 1].plot([], [], color='darkred', label="ρ(t) - Time-Density")
    axs[0, 1].set_title("Space-Time Density")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_xlim(-12, 12)
    axs[0, 1].set_ylim(0, 3)
    
    temp_flow_plot, = axs[1, 0].plot([], [], color='blue', label="Tf(t) - Temporal Flow")
    axs[1, 0].set_title("Temporal Flow Modulation")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].set_xlim(-12, 12)
    axs[1, 0].set_ylim(0, 1.1)
    
    vel_plot, = axs[1, 1].plot([], [], color='green', label="v(t) - Modulated Velocity")
    pres_plot, = axs[1, 1].plot([], [], color='purple', label="p(t) - Modulated Pressure")
    axs[1, 1].set_title("Derived Quantities")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].set_xlim(-12, 12)
    axs[1, 1].set_ylim(0, 3)
    
    # Timeline indicator
    timeline, = plt.plot([], [], 'ro')
    
    # Text display for current time
    time_text = fig.text(0.5, 0.04, '', ha='center')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    def init():
        """Initialize animation"""
        proj_exp_plot.set_data([], [])
        proj_exp_plot2.set_data([], [])
        density_plot.set_data([], [])
        temp_flow_plot.set_data([], [])
        vel_plot.set_data([], [])
        pres_plot.set_data([], [])
        timeline.set_data([], [])
        time_text.set_text('')
        
        return proj_exp_plot, proj_exp_plot2, density_plot, temp_flow_plot, vel_plot, pres_plot, timeline, time_text
    
    def animate(i):
        """Animate function for each frame"""
        # Define visible window (growing portion of the data)
        window_size = 300  # Number of points to show
        frame_step = 3     # How many points to add per frame
        
        idx = min(i * frame_step + window_size, len(t_full))
        start_idx = max(0, idx - window_size)
        
        # Current time point (for timeline indicator)
        current_t = t_full[min(start_idx + window_size - 1, len(t_full) - 1)]
        
        # Get visible data
        t = t_full[start_idx:idx]
        S = S_full[start_idx:idx]
        D = D_full[start_idx:idx]
        rho = rho_full[start_idx:idx]
        Tf = Tf_full[start_idx:idx]
        velocity = velocity_full[start_idx:idx]
        pressure = pressure_full[start_idx:idx]
        
        # Update plots
        proj_exp_plot.set_data(t, S)
        proj_exp_plot2.set_data(t, D)
        density_plot.set_data(t, rho)
        temp_flow_plot.set_data(t, Tf)
        vel_plot.set_data(t, velocity)
        pres_plot.set_data(t, pressure)
        
        # Update timeline indicator on all plots
        timeline.set_data([current_t], [0])
        
        # Update time text
        time_text.set_text(f'Time: {current_t:.2f}')
        
        return proj_exp_plot, proj_exp_plot2, density_plot, temp_flow_plot, vel_plot, pres_plot, timeline, time_text
    
    # Create animation
    num_frames = (len(t_full) - 300) // 3 + 1  # Calculate total frames based on window size and step
    ani = animation.FuncAnimation(
        fig, animate, frames=num_frames, init_func=init, 
        interval=30, blit=True
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    return ani, fig, output_dir, t_full, S_full, D_full, rho_full, Tf_full, velocity_full, pressure_full

def check_ffmpeg():
    """Check if FFmpeg is available in the system or common installation locations"""
    # Check in PATH
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
            # Found FFmpeg, but it's not in PATH
            print(f"FFmpeg found at {location} but not in system PATH")
            print("Temporarily adding to PATH for this session...")
            os.environ['PATH'] += os.pathsep + os.path.dirname(location)
            return True
            
    return False

def save_frames_simple(output_dir, t_full, S_full, D_full, rho_full, Tf_full, velocity_full, pressure_full, num_frames=20):
    """Save frames directly without using animation object"""
    print(f"Saving {num_frames} frames as image sequence...")
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Select time points to save (evenly spaced)
    indices = np.linspace(0, len(t_full) - 1, num_frames, dtype=int)
    
    for i, idx in enumerate(indices):
        # Create a new figure for each frame
        current_t = t_full[idx]
        
        # Create a window centered around the current time point
        window_half = 150
        start_idx = max(0, idx - window_half)
        end_idx = min(len(t_full), idx + window_half)
        
        # Get data for this window
        t = t_full[start_idx:end_idx]
        S = S_full[start_idx:end_idx]
        D = D_full[start_idx:end_idx]
        rho = rho_full[start_idx:end_idx]
        Tf = Tf_full[start_idx:end_idx]
        velocity = velocity_full[start_idx:end_idx]
        pressure = pressure_full[start_idx:end_idx]
        
        # Create a new figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Genesis-Sphere Visualization (t = {current_t:.2f})', fontsize=16)
        
        # Plot data
        axs[0, 0].plot(t, S, label="S(t) - Projection")
        axs[0, 0].plot(t, D, label="D(t) - Expansion")
        axs[0, 0].set_title("Projection & Expansion")
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        axs[0, 0].set_xlim(-12, 12)
        axs[0, 0].set_ylim(0, 3)
        
        axs[0, 1].plot(t, rho, color='darkred', label="ρ(t) - Time-Density")
        axs[0, 1].set_title("Space-Time Density")
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        axs[0, 1].set_xlim(-12, 12)
        axs[0, 1].set_ylim(0, 3)
        
        axs[1, 0].plot(t, Tf, color='blue', label="Tf(t) - Temporal Flow")
        axs[1, 0].set_title("Temporal Flow Modulation")
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_xlim(-12, 12)
        axs[1, 0].set_ylim(0, 1.1)
        
        axs[1, 1].plot(t, velocity, color='green', label="v(t) - Modulated Velocity")
        axs[1, 1].plot(t, pressure, color='purple', label="p(t) - Modulated Pressure")
        axs[1, 1].set_title("Derived Quantities")
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_xlim(-12, 12)
        axs[1, 1].set_ylim(0, 3)
        
        # Add a vertical line at current time
        for ax in axs.flat:
            ax.axvline(x=current_t, color='r', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved frame {i+1}/{num_frames}")
    
    print(f"All frames saved to {frames_dir}")
    return frames_dir

if __name__ == "__main__":
    print("Creating Genesis-Sphere animation...")
    ani, fig, output_dir, t_full, S_full, D_full, rho_full, Tf_full, velocity_full, pressure_full = create_animation()
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    
    # Flag to track if we need to show animation interactively
    show_interactive = True
    
    if has_ffmpeg:
        try:
            # Save animation as MP4
            output_file = os.path.join(output_dir, 'genesis_sphere_animation.mp4')
            print(f"Saving animation to {output_file}...")
            
            # Additional writer-specific arguments for higher quality
            writer = animation.FFMpegWriter(
                fps=30, 
                metadata=dict(title='Genesis-Sphere Animation', artist='Genesis-Sphere Project'),
                bitrate=1800
            )
            
            ani.save(output_file, writer=writer)
            print(f"Animation saved successfully to {output_file}")
            
            # If we've successfully saved the animation, don't show interactive display
            # to avoid potential rendering issues
            show_interactive = False
            print("\nAnimation saved to file. Skipping interactive display to avoid rendering issues.")
            print(f"You can view the animation at: {os.path.abspath(output_file)}")
            
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("\nTroubleshooting guide:")
            print("1. Make sure FFmpeg is installed and available in your system PATH")
            print("2. Try running 'ffmpeg -version' in your command prompt/terminal")
            print("3. If FFmpeg is not installed, see instructions below\n")
            
            # Use the safer fallback method
            print("Using fallback method to save frames...")
            frames_dir = save_frames_simple(output_dir, t_full, S_full, D_full, rho_full, Tf_full, 
                                           velocity_full, pressure_full, num_frames=20)
    else:
        print("\nFFmpeg not found in system PATH or common installation locations.")
        print("Using simpler frame-by-frame output instead of video.")
        
        # Use the safer fallback method directly
        frames_dir = save_frames_simple(output_dir, t_full, S_full, D_full, rho_full, Tf_full, 
                                       velocity_full, pressure_full, num_frames=20)
        
        print("\nTo install FFmpeg for future use:")
        print("1. Windows: Run the provided install_ffmpeg_windows.bat script")
        print("2. Restart your command prompt/PowerShell after installation")
        print("3. Verify installation with 'ffmpeg -version'")
    
    # Only show interactive display if we haven't successfully saved the animation
    if show_interactive:
        try:
            print("\nAttempting to display animation in interactive window...")
            plt.show()
        except Exception as e:
            print(f"Error displaying interactive animation: {e}")
            print("This is a known issue with animation rendering in some environments.")
            print("However, frames or video have been saved to the output directory.")
    
    print("Done.")
