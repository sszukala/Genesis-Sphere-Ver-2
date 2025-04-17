import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
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
    
    return ani, fig, output_dir

def check_ffmpeg():
    """Check if FFmpeg is available in the system"""
    import shutil
    return shutil.which('ffmpeg') is not None

def save_frames(ani, output_dir, num_frames=20):
    """Save individual frames instead of a video"""
    print(f"Saving {num_frames} frames as image sequence...")
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Calculate frame indices to save (evenly spaced)
    total_frames = ani._fig.axes[0].get_lines()[0].get_animated()
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i, idx in enumerate(indices):
        # Draw the frame
        ani._draw_next_frame(idx, None)
        # Save as PNG
        filename = os.path.join(frames_dir, f'frame_{i:03d}.png')
        ani._fig.savefig(filename, dpi=150)
        print(f"Saved frame {i+1}/{num_frames}")
    
    print(f"All frames saved to {frames_dir}")

if __name__ == "__main__":
    print("Creating Genesis-Sphere animation...")
    ani, fig, output_dir = create_animation()
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    
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
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("\nTroubleshooting guide:")
            print("1. Make sure FFmpeg is installed and available in your system PATH")
            print("2. Try running 'ffmpeg -version' in your command prompt/terminal")
            print("3. If FFmpeg is not installed, see instructions below\n")
            
            # Offer to save frames instead
            save_frames_prompt = input("Would you like to save animation frames as images instead? (y/n): ")
            if save_frames_prompt.lower() in ['y', 'yes']:
                save_frames(ani, output_dir)
    else:
        print("\nFFmpeg not found in system PATH. Cannot save as video.")
        print("FFmpeg Installation Instructions:")
        print("  - Windows: Download from https://ffmpeg.org/download.html and add to PATH")
        print("  - macOS: Using Homebrew: 'brew install ffmpeg'")
        print("  - Linux: Using apt: 'sudo apt install ffmpeg'")
        
        # Offer to save frames instead
        save_frames_prompt = input("Would you like to save animation frames as images instead? (y/n): ")
        if save_frames_prompt.lower() in ['y', 'yes']:
            save_frames(ani, output_dir)
    
    # Show the animation
    print("\nDisplaying animation in interactive window...")
    plt.show()
    print("Done.")
