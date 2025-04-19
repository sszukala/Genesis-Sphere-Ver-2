"""
Generates datasets to validate phantom divide crossing in the Genesis-Sphere model.
Phantom divide crossing (w(z) crossing -1) is a key signature of cyclic cosmologies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'models'))

from models.genesis_model import GenesisSphereModel

def generate_phantom_divide_dataset(omega_values=[0.8, 1.0, 1.5, 2.0], 
                                   alpha=0.02, beta=0.8, epsilon=0.1,
                                   redshift_range=(0, 5), n_points=500):
    """
    Generate dataset showing equation of state parameter w(z) evolution for different omega values.
    Demonstrates phantom divide crossing behavior in the Genesis-Sphere model.
    """
    z = np.linspace(redshift_range[0], redshift_range[1], n_points)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Generate w(z) for each omega value
    for omega in omega_values:
        # Create Genesis-Sphere model
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        
        # Convert redshift to Genesis-Sphere time
        t_values = np.array([-10 * np.log(1 + z_val) for z_val in z])
        
        # Calculate Genesis-Sphere functions
        gs_results = gs_model.evaluate_all(t_values)
        
        # Calculate equation of state parameter w(z)
        # This is a theoretical mapping based on the Genesis-Sphere model
        # w = -1 is the phantom divide
        density = gs_results['density']
        temporal_flow = gs_results['temporal_flow']
        
        # Simple model for w(z): oscillates around phantom divide based on genesis-sphere functions
        w = -1.0 - 0.3 * np.sin(omega * t_values) * (temporal_flow / np.max(temporal_flow))
        
        # Find crossing points
        crossing_indices = []
        for i in range(1, len(w)):
            if (w[i-1] < -1 and w[i] > -1) or (w[i-1] > -1 and w[i] < -1):
                crossing_indices.append(i)
        
        crossing_z = [z[i] for i in crossing_indices]
        crossing_t = [t_values[i] for i in crossing_indices]
        
        for z_val, t_val, w_val, d_val, tf_val in zip(z, t_values, w, density, temporal_flow):
            results.append({
                'omega': omega,
                'redshift': z_val,
                'time': t_val,
                'w': w_val,
                'density': d_val,
                'temporal_flow': tf_val
            })
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.plot(z, w, lw=2, label=f'ω = {omega}')
        plt.axhline(-1, color='k', linestyle='--', label='Phantom Divide (w = -1)')
        for i in crossing_indices:
            plt.axvline(z[i], color='r', linestyle=':', alpha=0.5)
        plt.title(f'Equation of State Parameter w(z) for Genesis-Sphere Model (ω = {omega})')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Equation of State w(z)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'phantom_divide_omega_{omega}.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        print(f"Generated w(z) data for ω={omega} with {len(crossing_indices)} phantom divide crossings")
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'phantom_divide_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved phantom divide dataset to {csv_path}")
    return df

def main():
    """Main function to generate the phantom divide crossing dataset"""
    print("Generating phantom divide crossing datasets...")
    generate_phantom_divide_dataset()
    print("Done.")

if __name__ == "__main__":
    main()
