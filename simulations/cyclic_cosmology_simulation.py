"""
Cyclic Cosmology Simulation

This script provides interactive simulations to explore the connection
between Genesis-Sphere cyclic cosmology and black hole physics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import shutil
from tqdm import tqdm  # Import tqdm for progress bars

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'models'))

from cyclic_bh_mapping import CyclicBlackHoleModel, check_ffmpeg

def run_parameter_exploration(output_dir):
    """
    Run parameter exploration showing how Genesis-Sphere parameters
    affect cyclic universe behavior and black hole correspondence
    
    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    """
    # Test different omega values (cycle frequencies)
    print("Exploring effect of omega parameter on cycle frequency...")
    
    omega_values = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(len(omega_values), 2, figsize=(12, 4*len(omega_values)))
    
    # Create progress bar
    progress_bar = tqdm(total=len(omega_values), desc="Testing omega values", unit="model")
    
    for i, omega in enumerate(omega_values):
        # Create model with this omega
        model = CyclicBlackHoleModel(omega=omega)
        
        # Generate data
        cycle_period = 2 * np.pi / omega  # Calculate cycle period from omega
        
        cycle_df = model.generate_cyclic_universe_data(
            t_min=-15, t_max=15, 
            cycle_period=cycle_period
        )
        
        # Plot density
        axes[i, 0].plot(cycle_df['time'], cycle_df['density'], 'r-')
        axes[i, 0].set_title(f'Density Evolution (ω={omega:.1f}, Period={cycle_period:.1f})')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Space-Time Density')
        axes[i, 0].grid(True)
        
        # Plot temporal flow
        axes[i, 1].plot(cycle_df['time'], cycle_df['temporal_flow'], 'b-')
        axes[i, 1].set_title(f'Temporal Flow (ω={omega:.1f})')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Temporal Flow')
        axes[i, 1].grid(True)
        
        progress_bar.update(1)
    
    progress_bar.close()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'omega_parameter_effect.png'), dpi=150)
    plt.close()
    
    # Test different beta values (temporal damping)
    print("Exploring effect of beta parameter on time dilation...")
    
    beta_values = [0.4, 0.8, 1.5]
    fig, axes = plt.subplots(len(beta_values), 2, figsize=(12, 4*len(beta_values)))
    
    # Create progress bar
    progress_bar = tqdm(total=len(beta_values), desc="Testing beta values", unit="model")
    
    for i, beta in enumerate(beta_values):
        # Create model with this beta
        model = CyclicBlackHoleModel(beta=beta)
        
        # Generate data
        cycle_df = model.generate_cyclic_universe_data(t_min=-10, t_max=10)
        bh_df = model.generate_bh_data()
        
        # Plot temporal flow
        axes[i, 0].plot(cycle_df['time'], cycle_df['temporal_flow'], 'b-')
        axes[i, 0].set_title(f'Temporal Flow (β={beta:.1f})')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Temporal Flow')
        axes[i, 0].grid(True)
        
        # Plot BH time dilation
        axes[i, 1].plot(bh_df['r'], bh_df['time_dilation'], 'r-')
        axes[i, 1].set_xscale('log')
        axes[i, 1].set_title(f'Black Hole Time Dilation (β={beta:.1f})')
        axes[i, 1].set_xlabel('Radial Distance (r)')
        axes[i, 1].set_ylabel('Time Dilation')
        axes[i, 1].grid(True)
        
        progress_bar.update(1)
    
    progress_bar.close()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'beta_parameter_effect.png'), dpi=150)
    plt.close()
    
    print("✓ Parameter exploration visualizations saved to output directory.")

def run_cycle_bh_mapping_simulation(omega=1.0, beta=0.8, alpha=0.02, epsilon=0.1, 
                                   cycle_period=None, output_dir=None):
    """
    Run a simulation showing the mapping between cyclic cosmology and black hole physics
    
    Parameters:
    -----------
    omega : float
        Angular frequency parameter
    beta : float
        Temporal damping factor
    alpha : float
        Spatial dimension expansion coefficient
    epsilon : float
        Zero-prevention constant
    cycle_period : float, optional
        Explicit cycle period (if None, calculated from omega)
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    dict
        Paths to generated output files
    """
    # If cycle_period is provided, calculate corresponding omega
    if cycle_period is not None:
        omega = 2 * np.pi / cycle_period
    else:
        cycle_period = 2 * np.pi / omega
    
    print(f"Running cyclic universe to black hole mapping simulation...")
    print(f"Parameters: α={alpha:.3f}, β={beta:.3f}, ω={omega:.3f}, ε={epsilon:.3f}")
    print(f"Cycle Period={cycle_period:.3f}")
    
    # Create model
    model = CyclicBlackHoleModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
    
    # Generate static visualization
    print("Generating static visualization...")
    fig_path = model.visualize_bh_cyclic_mapping(cycle_period=cycle_period)
    
    # Generate short animation
    print("Generating animation...")
    anim_path = model.create_cycle_animation(
        cycle_period=cycle_period, 
        num_cycles=2, 
        duration=10
    )
    
    print("\n✓ Simulation completed. Output saved to:")
    print(f"- Static visualization: {fig_path}")
    print(f"- Animation: {anim_path}")
    
    return {
        'fig_path': fig_path,
        'anim_path': anim_path
    }

def main():
    """Main function to run the simulation"""
    parser = argparse.ArgumentParser(description="Genesis-Sphere Cyclic Cosmology Simulation")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=0.8, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=1.0, help="Angular frequency parameter")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Zero-prevention constant")
    parser.add_argument("--cycle-period", type=float, help="Explicit cycle period (overrides omega)")
    parser.add_argument("--param-exploration", action="store_true", help="Run parameter exploration")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(parent_dir, 'output', 'cyclic_bh')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.param_exploration:
        run_parameter_exploration(output_dir)
    else:
        run_cycle_bh_mapping_simulation(
            alpha=args.alpha,
            beta=args.beta,
            omega=args.omega,
            epsilon=args.epsilon,
            cycle_period=args.cycle_period,
            output_dir=output_dir
        )
    
    print("\nTo explore further:")
    print("1. Try different omega values to change cycle frequency")
    print("2. Adjust beta to modify time dilation effects")
    print("3. Run with --param-exploration to see parameter effect visualizations")
    
    # Check if FFmpeg is available
    if not check_ffmpeg():
        print("\nNote: FFmpeg is not installed. Animations are saved as individual frames.")
        print("To install FFmpeg for creating videos:")
        print("1. Download from https://ffmpeg.org/download.html")
        print("2. Add the bin directory to your PATH environment variable")
        print("3. Restart your terminal/command prompt after installation")

if __name__ == "__main__":
    main()
