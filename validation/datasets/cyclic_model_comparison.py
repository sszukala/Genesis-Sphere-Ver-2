"""
Generates datasets comparing Genesis-Sphere with established cyclic cosmological models.
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

def generate_ekpyrotic_model(time, scale=1.0):
    """Generate data for Ekpyrotic model (Steinhardt & Turok)"""
    # Simplified model of density in ekpyrotic scenario
    # Characterized by slow contraction, rapid expansion, and brane collisions
    
    # Scale the time to match the ekpyrotic cycle period
    t = time / scale
    
    # Base oscillating component
    base = 1 + 0.5 * np.cos(t)
    
    # Add brane collision spikes at each cycle boundary
    brane_collision = 3.0 * np.exp(-10 * (t % (2*np.pi) - np.pi)**2)
    
    # Combine components
    density = base + brane_collision
    
    # Calculate scale factor (a ∝ ρ^(-1/3) in simple model)
    scale_factor = density**(-1/3)
    
    # Normalize scale factor
    scale_factor = scale_factor / np.max(scale_factor)
    
    return {
        'density': density,
        'scale_factor': scale_factor,
        'model_name': 'Ekpyrotic',
        'description': 'Slow contraction, rapid expansion with brane collisions'
    }

def generate_oscillating_model(time, scale=1.0):
    """Generate data for Tolman's oscillating universe model"""
    # Simplified model of density in oscillating universe
    # Each cycle has slightly different characteristics due to entropy
    
    # Scale the time to match the cycle period
    t = time / scale
    
    # Calculate cycle number for each time point
    cycle_num = np.floor(t / (2*np.pi))
    
    # Base oscillating component with decreasing amplitude due to entropy
    amplitude = 1.0 / (1.0 + 0.1 * cycle_num)
    phase = t % (2*np.pi)
    
    # Generate density profile
    density = 0.5 + amplitude * (1 + np.cos(phase))
    
    # Calculate scale factor
    scale_factor = density**(-1/3)
    
    # Normalize scale factor
    scale_factor = scale_factor / np.max(scale_factor)
    
    return {
        'density': density,
        'scale_factor': scale_factor,
        'model_name': 'Oscillating (Tolman)',
        'description': 'Cosmos oscillates with entropy effects between cycles'
    }

def generate_quantum_bounce_model(time, scale=1.0):
    """Generate data for Loop Quantum Cosmology bounce model"""
    # Simplified model of density in LQC
    # Characterized by smooth bounce without singularity
    
    # Scale the time to match the cycle period
    t = time / scale
    
    # Base oscillating component
    base = 1 + np.cos(t)
    
    # Quantum effects prevent singularity - density is bounded
    quantum_correction = 1.0 / (1.0 + 5.0 * np.exp(-5.0 * (t % (2*np.pi) - np.pi)**2))
    
    # Combine components
    density = base * quantum_correction
    
    # Calculate scale factor
    scale_factor = density**(-1/3)
    
    # Normalize scale factor
    scale_factor = scale_factor / np.max(scale_factor)
    
    return {
        'density': density,
        'scale_factor': scale_factor,
        'model_name': 'Quantum Bounce',
        'description': 'Quantum gravity prevents singularity at bounce'
    }

def generate_cyclic_comparison_dataset(alpha=0.02, beta=0.8, omega=1.0, epsilon=0.1,
                                     time_range=(-20, 20), n_points=1000):
    """
    Generate dataset comparing Genesis-Sphere with established cyclic cosmology models.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate time array
    time = np.linspace(time_range[0], time_range[1], n_points)
    
    # Create Genesis-Sphere model
    gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
    
    # Calculate Genesis-Sphere functions
    gs_results = gs_model.evaluate_all(time)
    gs_density = gs_results['density']
    gs_temporal_flow = gs_results['temporal_flow']
    
    # Generate cyclic cosmology models
    ekpyrotic_data = generate_ekpyrotic_model(time, scale=2.0)
    oscillating_data = generate_oscillating_model(time, scale=2.5)
    quantum_data = generate_quantum_bounce_model(time, scale=3.0)
    
    # Normalize all densities to similar scale for comparison
    gs_density_norm = gs_density / np.max(gs_density)
    ekpyrotic_density_norm = ekpyrotic_data['density'] / np.max(ekpyrotic_data['density'])
    oscillating_density_norm = oscillating_data['density'] / np.max(oscillating_data['density'])
    quantum_density_norm = quantum_data['density'] / np.max(quantum_data['density'])
    
    # Find peaks for all models
    gs_peaks, _ = find_peaks(gs_density_norm, height=0.5)
    ekpyrotic_peaks, _ = find_peaks(ekpyrotic_density_norm, height=0.5)
    oscillating_peaks, _ = find_peaks(oscillating_density_norm, height=0.5)
    quantum_peaks, _ = find_peaks(quantum_density_norm, height=0.5)
    
    # Calculate periods
    def calculate_period(time, peaks):
        if len(peaks) >= 2:
            peak_times = time[peaks]
            periods = np.diff(peak_times)
            return np.mean(periods)
        return np.nan
    
    gs_period = calculate_period(time, gs_peaks)
    ekpyrotic_period = calculate_period(time, ekpyrotic_peaks)
    oscillating_period = calculate_period(time, oscillating_peaks)
    quantum_period = calculate_period(time, quantum_peaks)
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Density comparison
    plt.subplot(2, 1, 1)
    plt.plot(time, gs_density_norm, 'b-', label=f'Genesis-Sphere (ω={omega}, period={gs_period:.2f})')
    plt.plot(time, ekpyrotic_density_norm, 'r--', label=f'Ekpyrotic (period={ekpyrotic_period:.2f})')
    plt.plot(time, oscillating_density_norm, 'g-.', label=f'Oscillating (period={oscillating_period:.2f})')
    plt.plot(time, quantum_density_norm, 'm:', label=f'Quantum Bounce (period={quantum_period:.2f})')
    plt.title('Density Comparison: Genesis-Sphere vs. Established Cyclic Models')
    plt.xlabel('Time')
    plt.ylabel('Normalized Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Scale factor comparison
    plt.subplot(2, 1, 2)
    # Calculate simple scale factor for Genesis-Sphere (a ∝ ρ^(-1/3) in simple model)
    gs_scale_factor = gs_density**(-1/3)
    gs_scale_factor = gs_scale_factor / np.max(gs_scale_factor)
    
    plt.plot(time, gs_scale_factor, 'b-', label='Genesis-Sphere')
    plt.plot(time, ekpyrotic_data['scale_factor'], 'r--', label='Ekpyrotic')
    plt.plot(time, oscillating_data['scale_factor'], 'g-.', label='Oscillating')
    plt.plot(time, quantum_data['scale_factor'], 'm:', label='Quantum Bounce')
    plt.title('Scale Factor Comparison: Genesis-Sphere vs. Established Cyclic Models')
    plt.xlabel('Time')
    plt.ylabel('Normalized Scale Factor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'cyclic_model_comparison.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Create data for CSV
    results = []
    
    for t, gs_d, gs_tf, gs_sf, ek_d, ek_sf, os_d, os_sf, q_d, q_sf in zip(
            time, gs_density_norm, gs_temporal_flow, gs_scale_factor,
            ekpyrotic_density_norm, ekpyrotic_data['scale_factor'],
            oscillating_density_norm, oscillating_data['scale_factor'],
            quantum_density_norm, quantum_data['scale_factor']):
        
        results.append({
            'time': t,
            'gs_density': gs_d,
            'gs_temporal_flow': gs_tf,
            'gs_scale_factor': gs_sf,
            'ekpyrotic_density': ek_d,
            'ekpyrotic_scale_factor': ek_sf,
            'oscillating_density': os_d,
            'oscillating_scale_factor': os_sf,
            'quantum_density': q_d,
            'quantum_scale_factor': q_sf
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'cyclic_model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved cyclic model comparison dataset to {csv_path}")
    
    # Create summary dataset
    summary = [
        {'model': 'Genesis-Sphere', 'period': gs_period, 'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon},
        {'model': 'Ekpyrotic', 'period': ekpyrotic_period},
        {'model': 'Oscillating', 'period': oscillating_period},
        {'model': 'Quantum Bounce', 'period': quantum_period}
    ]
    
    summary_df = pd.DataFrame(summary)
    summary_csv_path = os.path.join(output_dir, 'cyclic_model_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Create table with model characteristics for documentation
    model_characteristics = [
        {'model': 'Genesis-Sphere', 'key_features': 'Temporal flow slows near t=0; Sinusoidal density modulation', 
         'parameters': f'α={alpha}, β={beta}, ω={omega}, ε={epsilon}', 'cycle_period': f'{gs_period:.2f}'},
        {'model': 'Ekpyrotic', 'key_features': 'Slow contraction, rapid expansion; Brane collisions', 
         'parameters': 'Theoretical', 'cycle_period': f'{ekpyrotic_period:.2f}'},
        {'model': 'Oscillating (Tolman)', 'key_features': 'Entropy increases with each cycle; Decreasing amplitude', 
         'parameters': 'Theoretical', 'cycle_period': f'{oscillating_period:.2f}'},
        {'model': 'Quantum Bounce', 'key_features': 'Quantum gravity prevents singularity; Smooth bounce', 
         'parameters': 'Theoretical', 'cycle_period': f'{quantum_period:.2f}'}
    ]
    
    char_df = pd.DataFrame(model_characteristics)
    char_csv_path = os.path.join(output_dir, 'model_characteristics.csv')
    char_df.to_csv(char_csv_path, index=False)
    
    return df

def main():
    """Main function to generate the cyclic model comparison dataset"""
    print("Generating cyclic model comparison datasets...")
    generate_cyclic_comparison_dataset()
    print("Done.")

if __name__ == "__main__":
    main()
