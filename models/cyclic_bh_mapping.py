"""
Cyclic Black Hole Mapping

This module extends the Genesis-Sphere model to demonstrate the relationship
between black hole physics (particularly Kerr-Newman dynamics) and cyclic universe
models, showing how the Genesis-Sphere formulation naturally handles both.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import shutil
from tqdm import tqdm  # Import tqdm for progress bars

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'validation'))

from genesis_model import GenesisSphereModel
from black_hole_validation import generate_kerr_newman_time_dilation

# Create output directory
output_dir = os.path.join(parent_dir, 'output', 'cyclic_bh')
os.makedirs(output_dir, exist_ok=True)

# Helper function to check for FFmpeg
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

class CyclicBlackHoleModel:
    """
    Model that demonstrates the connection between Genesis-Sphere formulation,
    black hole physics, and cyclic universe models.
    """
    
    def __init__(self, alpha=0.02, beta=0.8, omega=1.0, epsilon=0.1, spin=0.9, charge=0.4):
        """Initialize the model with parameters for both Genesis-Sphere and black hole physics"""
        self.gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.epsilon = epsilon
        self.spin = spin
        self.charge = charge
        
    def generate_bh_data(self, r_min=0.1, r_max=10, num_points=200):
        """Generate black hole time dilation data for comparison"""
        return generate_kerr_newman_time_dilation(
            r_min=r_min, r_max=r_max, num_points=num_points, 
            spin=self.spin, charge=self.charge
        )
        
    def generate_cyclic_universe_data(self, t_min=-20, t_max=20, num_points=1000, 
                                     num_cycles=3, cycle_period=10):
        """
        Generate data for a cyclic universe model using Genesis-Sphere formulation
        
        Parameters:
        -----------
        t_min, t_max : float
            Time range
        num_points : int
            Number of time points to generate
        num_cycles : int
            Number of complete cycles to model
        cycle_period : float
            Period of each cycle
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the cyclic universe data
        """
        print(f"Generating cyclic universe data ({num_points} points, {num_cycles} cycles)...")
        
        # Create time array
        t = np.linspace(t_min, t_max, num_points)
        
        # Calculate Genesis-Sphere functions with modified omega for cycles
        cycle_omega = 2 * np.pi / cycle_period
        
        # Create a model with the cycle frequency
        cyclic_gs = GenesisSphereModel(
            alpha=self.alpha, 
            beta=self.beta,
            omega=cycle_omega,  # This creates the cyclic behavior
            epsilon=self.epsilon
        )
        
        # Get all Genesis-Sphere functions
        gs_data = cyclic_gs.evaluate_all(t)
        
        # Add cycle phase information
        cycle_phase = (t % cycle_period) / cycle_period
        current_cycle = np.floor(t / cycle_period)
        
        # Create DataFrame with results
        df = pd.DataFrame({
            'time': t,
            'density': gs_data['density'],
            'temporal_flow': gs_data['temporal_flow'],
            'cycle_phase': cycle_phase,
            'current_cycle': current_cycle,
            'velocity': gs_data['velocity'],
            'pressure': gs_data['pressure']
        })
        
        print(f"✓ Cyclic universe data generated successfully")
        return df
        
    def map_bh_to_cyclic_universe(self, bh_df, cycle_period=10):
        """
        Map black hole dynamics to cyclic universe phases
        
        Parameters:
        -----------
        bh_df : pd.DataFrame
            Black hole data from generate_bh_data
        cycle_period : float
            Period of each cosmic cycle
            
        Returns:
        --------
        pd.DataFrame
            Mapping between black hole and cyclic universe data
        """
        # Extract black hole parameters
        r = bh_df['r'].values
        time_dilation = bh_df['time_dilation'].values
        
        # Map radial distance to cycle phase
        # Smaller r corresponds to later cycle phase (approaching singularity)
        cycle_phase = 1 - np.log(r) / np.log(r.min()) 
        cycle_phase = np.clip(cycle_phase, 0, 1)  # Keep in [0,1] range
        
        # Map to time in cycle
        cycle_time = cycle_phase * cycle_period
        
        # Use Genesis-Sphere model to calculate cycle properties
        cycle_omega = 2 * np.pi / cycle_period
        cyclic_gs = GenesisSphereModel(
            alpha=self.alpha, 
            beta=self.beta,
            omega=cycle_omega,
            epsilon=self.epsilon
        )
        
        gs_data = cyclic_gs.evaluate_all(cycle_time)
        
        # Create DataFrame with mapping
        df = pd.DataFrame({
            'radial_distance': r,
            'bh_time_dilation': time_dilation,
            'cycle_phase': cycle_phase,
            'cycle_time': cycle_time,
            'gs_density': gs_data['density'],
            'gs_temporal_flow': gs_data['temporal_flow']
        })
        
        return df
    
    def visualize_bh_cyclic_mapping(self, bh_df=None, cycle_df=None, cycle_period=10):
        """
        Visualize the mapping between black hole physics and cyclic universe model
        
        Parameters:
        -----------
        bh_df : pd.DataFrame, optional
            Black hole data, if None it will be generated
        cycle_df : pd.DataFrame, optional
            Cyclic universe data, if None it will be generated
        cycle_period : float
            Period of each cosmic cycle
            
        Returns:
        --------
        str
            Path to the saved figure
        """
        # Generate data if not provided
        if bh_df is None:
            bh_df = self.generate_bh_data()
        
        if cycle_df is None:
            cycle_df = self.generate_cyclic_universe_data(cycle_period=cycle_period)
        
        # Create mapping
        mapping_df = self.map_bh_to_cyclic_universe(bh_df, cycle_period)
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        
        # 1. Plot black hole time dilation vs radial distance
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(bh_df['r'], bh_df['time_dilation'], 'b-')
        ax1.set_xscale('log')
        ax1.set_xlabel('Radial Distance (r/Rs)')
        ax1.set_ylabel('Time Dilation')
        ax1.set_title('Kerr-Newman Black Hole Time Dilation')
        ax1.grid(True)
        
        # 2. Plot cyclic universe density over multiple cycles
        ax2 = plt.subplot(2, 2, 2)
        sample = cycle_df.iloc[::5]  # Sample for clearer plotting
        scatter = ax2.scatter(sample['time'], sample['density'], 
                     c=sample['cycle_phase'], cmap='viridis', 
                     alpha=0.7, s=15)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Space-Time Density')
        ax2.set_title('Cyclic Universe Density Evolution')
        ax2.grid(True)
        plt.colorbar(scatter, ax=ax2, label='Cycle Phase')
        
        # 3. Plot mapping between BH and cycle
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(mapping_df['cycle_phase'], mapping_df['bh_time_dilation'], 'r-')
        ax3.set_xlabel('Cycle Phase')
        ax3.set_ylabel('Black Hole Time Dilation')
        ax3.set_title('BH Time Dilation vs. Cycle Phase')
        ax3.grid(True)
        
        # 4. Plot comparison between BH time dilation and GS temporal flow
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(mapping_df['cycle_phase'], mapping_df['bh_time_dilation'], 'b-', 
                label='BH Time Dilation')
        ax4.plot(mapping_df['cycle_phase'], mapping_df['gs_temporal_flow'], 'r--', 
                label='GS Temporal Flow')
        ax4.set_xlabel('Cycle Phase')
        ax4.set_ylabel('Time Dilation / Temporal Flow')
        ax4.set_title('BH vs. Genesis-Sphere Comparison')
        ax4.legend()
        ax4.grid(True)
        
        # Set overall title
        plt.suptitle(f'Black Hole Physics and Cyclic Universes in Genesis-Sphere Model\n'
                    f'(α={self.alpha:.3f}, β={self.beta:.3f}, ω={self.omega:.3f}, '
                    f'ε={self.epsilon:.3f}, spin={self.spin:.2f}, charge={self.charge:.2f})',
                    fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        file_path = os.path.join(output_dir, 'bh_cyclic_mapping.png')
        plt.savefig(file_path, dpi=150)
        plt.close(fig)
        
        return file_path
    
    def create_cycle_animation(self, cycle_period=10, num_cycles=2, fps=30, duration=10):
        """
        Create an animation showing the cyclic universe evolution and its relation
        to black hole time dilation effects
        
        Parameters:
        -----------
        cycle_period : float
            Period of each cosmic cycle
        num_cycles : int
            Number of cycles to animate
        fps : int
            Frames per second
        duration : float
            Duration of animation in seconds
            
        Returns:
        --------
        str
            Path to the saved animation or frames directory
        """
        # Generate data
        t_max = num_cycles * cycle_period
        num_points = int(fps * duration)
        cycle_df = self.generate_cyclic_universe_data(-t_max/2, t_max/2, num_points, 
                                                    num_cycles, cycle_period)
        bh_df = self.generate_bh_data()
        mapping_df = self.map_bh_to_cyclic_universe(bh_df, cycle_period)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Initialize plots
        density_line, = axes[0].plot([], [], 'r-', lw=2)
        axes[0].set_xlim(cycle_df['time'].min(), cycle_df['time'].max())
        axes[0].set_ylim(0, cycle_df['density'].max() * 1.1)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Space-Time Density')
        axes[0].set_title('Cyclic Universe Density Evolution')
        axes[0].grid(True)
        
        flow_line, = axes[1].plot([], [], 'b-', lw=2)
        axes[1].set_xlim(cycle_df['time'].min(), cycle_df['time'].max())
        axes[1].set_ylim(0, 1.1)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Temporal Flow')
        axes[1].set_title('Cyclic Universe Temporal Flow')
        axes[1].grid(True)
        
        # Phase space plot (Density vs Flow)
        phase_line, = axes[2].plot([], [], 'g-', lw=1, alpha=0.5)
        phase_point, = axes[2].plot([], [], 'go', ms=8)
        axes[2].set_xlim(0, cycle_df['density'].max() * 1.1)
        axes[2].set_ylim(0, 1.1)
        axes[2].set_xlabel('Space-Time Density')
        axes[2].set_ylabel('Temporal Flow')
        axes[2].set_title('Phase Space Evolution')
        axes[2].grid(True)
        
        # BH mapping plot
        bh_line, = axes[3].plot(mapping_df['cycle_phase'], mapping_df['bh_time_dilation'], 'b-', lw=2)
        bh_point, = axes[3].plot([], [], 'ro', ms=8)
        axes[3].set_xlim(0, 1)
        axes[3].set_ylim(0, 1.1)
        axes[3].set_xlabel('Cycle Phase')
        axes[3].set_ylabel('Time Dilation')
        axes[3].set_title('Corresponding Black Hole State')
        axes[3].grid(True)
        
        # Add time indicator
        time_text = fig.text(0.5, 0.01, '', ha='center')
        
        # Set overall title
        plt.suptitle(f'Genesis-Sphere Cyclic Universe and Black Hole Correspondence\n'
                    f'(α={self.alpha:.3f}, β={self.beta:.3f}, ω={2*np.pi/cycle_period:.3f}, '
                    f'ε={self.epsilon:.3f})',
                    fontsize=14)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        def init():
            """Initialize animation"""
            density_line.set_data([], [])
            flow_line.set_data([], [])
            phase_line.set_data([], [])
            phase_point.set_data([], [])
            bh_point.set_data([], [])
            time_text.set_text('')
            return density_line, flow_line, phase_line, phase_point, bh_point, time_text
        
        def animate(i):
            """Update animation for each frame"""
            # Determine how much of the data to show (growing window)
            end_idx = min(i + 1, len(cycle_df))
            window = min(end_idx, 100)  # Show at most 100 points at a time
            start_idx = max(0, end_idx - window)
            
            # Get current data window
            current_data = cycle_df.iloc[start_idx:end_idx]
            
            # Update density plot
            density_line.set_data(current_data['time'], current_data['density'])
            
            # Update flow plot
            flow_line.set_data(current_data['time'], current_data['temporal_flow'])
            
            # Update phase space plot
            phase_line.set_data(current_data['density'], current_data['temporal_flow'])
            if not current_data.empty:
                phase_point.set_data([current_data['density'].iloc[-1]], 
                                   [current_data['temporal_flow'].iloc[-1]])
            
            # Update BH mapping plot
            if not current_data.empty:
                # Get current cycle phase
                current_phase = current_data['cycle_phase'].iloc[-1]
                bh_point.set_data([current_phase], 
                                 [np.interp(current_phase, mapping_df['cycle_phase'], 
                                           mapping_df['bh_time_dilation'])])
            
            # Update time text
            if not current_data.empty:
                time_text.set_text(f'Time: {current_data["time"].iloc[-1]:.2f} | '
                                 f'Cycle: {current_data["current_cycle"].iloc[-1]:.0f} | '
                                 f'Phase: {current_data["cycle_phase"].iloc[-1]:.2f}')
            
            return density_line, flow_line, phase_line, phase_point, bh_point, time_text
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(cycle_df), interval=1000/fps, blit=True)
        
        # Create output directory
        frames_dir = os.path.join(output_dir, 'cycle_frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Path for video file
        file_path = os.path.join(output_dir, 'cyclic_universe_animation.mp4')
        
        # Check if ffmpeg is available
        has_ffmpeg = check_ffmpeg()
        
        if has_ffmpeg:
            try:
                # Try to save with ffmpeg
                writer = 'ffmpeg'
                print(f"Saving animation with FFmpeg ({len(cycle_df)} frames)...")
                anim.save(file_path, writer=writer, fps=fps, 
                          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
                print(f"✓ Animation video saved to: {file_path}")
                return file_path
            except Exception as e:
                print(f"Error saving animation with FFmpeg: {e}")
                print("Falling back to saving individual frames...")
        else:
            print("FFmpeg not found. Saving individual frames instead...")
        
        # If we got here, FFmpeg failed or is not available
        # Save individual frames
        print(f"Saving {len(cycle_df)} frames to {frames_dir}...")
        progress_bar = tqdm(total=len(cycle_df), desc="Generating frames", unit="frame")
        for i in range(len(cycle_df)):
            # Call the animate function to update the plot
            animate(i)
            # Save the current figure
            frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_path)
            progress_bar.update(1)
        
        progress_bar.close()
        print(f"✓ Animation frames saved to: {frames_dir}")
        print("To create a video, install FFmpeg:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add the bin directory to your PATH environment variable")
        print("3. Restart your terminal/command prompt after installation")
        
        plt.close(fig)
        return frames_dir

def main():
    """Run the Cyclic Black Hole mapping demonstration"""
    print("Genesis-Sphere: Black Hole to Cyclic Universe Mapping")
    print("====================================================")
    
    # Create the model
    model = CyclicBlackHoleModel(
        alpha=0.02,  # Spatial dimension expansion coefficient
        beta=0.8,    # Temporal damping factor
        omega=1.0,   # Base angular frequency
        epsilon=0.1, # Zero-prevention constant
        spin=0.9,    # Black hole spin parameter
        charge=0.4   # Black hole charge parameter
    )
    
    # Generate static visualization
    print("Generating static visualization of black hole to cyclic universe mapping...")
    fig_path = model.visualize_bh_cyclic_mapping()
    print(f"✓ Static visualization saved to: {fig_path}")
    
    # Generate animation
    print("Generating animation of cyclic universe evolution...")
    anim_path = model.create_cycle_animation(cycle_period=8, num_cycles=2)
    print(f"✓ Animation saved to: {anim_path}")
    
    print("\nAnalysis:")
    print("The Genesis-Sphere formulation demonstrates how black hole time dilation")
    print("effects can be directly related to cyclic universe dynamics.")
    print("- Omega (ω) parameter controls oscillation frequency, directly mapping to cosmic cycle period")
    print("- Beta (β) parameter influences time dilation magnitude, similar to black hole gravity strength")
    print("- The temporal flow function 1/(1+β|t|+ε) captures essential behaviors in both contexts")
    
    print("\nSee generated files for detailed visualizations.")

if __name__ == "__main__":
    main()
