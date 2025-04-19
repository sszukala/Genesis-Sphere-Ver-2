"""
Generates datasets to analyze how the omega parameter affects cycle periods in the Genesis-Sphere model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'models'))

from models.genesis_model import GenesisSphereModel

def generate_cycle_period_dataset(omega_range=(0.5, 3.0), num_omega=10, 
                                 alpha=0.02, beta=0.8, epsilon=0.1,
                                 time_range=(-50, 50), n_points=1000):
    """
    Generate dataset analyzing cycle periods for different omega values.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    omega_values = np.linspace(omega_range[0], omega_range[1], num_omega)
    time = np.linspace(time_range[0], time_range[1], n_points)
    
    results = []
    
    # Plot to compare all densities
    plt.figure(figsize=(15, 8))
    
    # For storing measured periods
    theoretical_periods = []  # 2π/ω
    measured_periods = []     # actual periods in density oscillation
    
    # Generate density oscillations for each omega value
    for omega in omega_values:
        # Calculate theoretical period (2π/ω)
        theoretical_period = 2 * np.pi / omega
        
        # Create Genesis-Sphere model
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        
        # Calculate Genesis-Sphere functions
        gs_results = gs_model.evaluate_all(time)
        density = gs_results['density']
        
        # Find peaks to measure actual period
        peaks, _ = find_peaks(density, height=np.mean(density))
        
        # Calculate average period from peak distances if we have enough peaks
        if len(peaks) >= 2:
            peak_times = time[peaks]
            periods = np.diff(peak_times)
            measured_period = np.mean(periods)
        else:
            measured_period = float('nan')
        
        theoretical_periods.append(theoretical_period)
        measured_periods.append(measured_period)
        
        # Add to results
        for t, d in zip(time, density):
            results.append({
                'omega': omega,
                'time': t,
                'density': d,
                'theoretical_period': theoretical_period,
                'measured_period': measured_period
            })
        
        # Add to plot
        plt.plot(time, density, label=f'ω = {omega:.2f}, Period ≈ {measured_period:.2f}')
    
    # Finalize plot
    plt.title('Density Oscillations for Different ω Values in Genesis-Sphere Model')
    plt.xlabel('Time')
    plt.ylabel('Density ρ(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'cycle_period_comparison.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Plot theoretical vs measured periods
    plt.figure(figsize=(10, 6))
    plt.plot(omega_values, theoretical_periods, 'b-', label='Theoretical Period (2π/ω)')
    plt.plot(omega_values, measured_periods, 'r--', label='Measured Period')
    plt.title('Theoretical vs. Measured Cycle Periods in Genesis-Sphere Model')
    plt.xlabel('Angular Frequency (ω)')
    plt.ylabel('Cycle Period')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'theoretical_vs_measured_periods.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Save period comparison data
    period_df = pd.DataFrame({
        'omega': omega_values,
        'theoretical_period': theoretical_periods,
        'measured_period': measured_periods
    })
    period_csv_path = os.path.join(output_dir, 'cycle_period_comparison.csv')
    period_df.to_csv(period_csv_path, index=False)
    
    # Save full dataset to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'cycle_period_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved cycle period dataset to {csv_path}")
    return df

def main():
    """Main function to generate the cycle period dataset"""
    print("Generating cycle period analysis datasets...")
    generate_cycle_period_dataset()
    print("Done.")

if __name__ == "__main__":
    main()
