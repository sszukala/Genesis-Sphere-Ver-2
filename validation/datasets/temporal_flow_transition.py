"""
Generates datasets to validate temporal flow behavior near cycle transitions (t=0).
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

def generate_temporal_flow_dataset(beta_values=[0.4, 0.8, 1.2, 1.6], 
                                 alpha=0.02, omega=1.0, epsilon=0.1,
                                 time_range=(-5, 5), focus_range=(-1, 1),
                                 n_points=1000, n_focus_points=200):
    """
    Generate dataset showing temporal flow behavior near t=0 for different beta values.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate full range time array
    time = np.linspace(time_range[0], time_range[1], n_points)
    
    # Generate focused time array (near t=0)
    focus_time = np.linspace(focus_range[0], focus_range[1], n_focus_points)
    
    results = []
    
    # Generate temporal flow for each beta value
    for beta in beta_values:
        # Create Genesis-Sphere model
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        
        # Calculate Genesis-Sphere functions for full range
        gs_results = gs_model.evaluate_all(time)
        density = gs_results['density']
        temporal_flow = gs_results['temporal_flow']
        
        # Calculate Genesis-Sphere functions for focus range
        gs_focus_results = gs_model.evaluate_all(focus_time)
        focus_density = gs_focus_results['density']
        focus_temporal_flow = gs_focus_results['temporal_flow']
        
        # Calculate minimum temporal flow (maximum time dilation)
        min_tf = np.min(temporal_flow)
        min_tf_time = time[np.argmin(temporal_flow)]
        
        # Calculate recovery rate (how quickly temporal flow returns to normal)
        # We'll measure this as the time taken to reach 50% of max temporal flow
        half_max_tf = (np.max(temporal_flow) + min_tf) / 2
        
        # Find index where temporal flow first exceeds half_max_tf after reaching minimum
        min_idx = np.argmin(temporal_flow)
        recovery_indices = np.where(temporal_flow[min_idx:] > half_max_tf)[0]
        
        if len(recovery_indices) > 0:
            recovery_time = time[min_idx + recovery_indices[0]] - min_tf_time
        else:
            recovery_time = np.nan
        
        # Add to results
        for t, d, tf in zip(time, density, temporal_flow):
            results.append({
                'beta': beta,
                'time': t,
                'density': d,
                'temporal_flow': tf,
                'min_temporal_flow': min_tf,
                'recovery_time': recovery_time
            })
        
        # Create visualization of full range
        plt.figure(figsize=(12, 6))
        plt.plot(time, temporal_flow, lw=2, label=f'Temporal Flow, β = {beta}')
        plt.axvline(0, color='k', linestyle='--', label='Cycle Transition (t=0)')
        plt.axhline(min_tf, color='r', linestyle=':', label=f'Min Tf = {min_tf:.4f}')
        plt.title(f'Temporal Flow near Cycle Transition (β = {beta})')
        plt.xlabel('Time')
        plt.ylabel('Temporal Flow Tf(t)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'temporal_flow_beta_{beta}.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        # Create visualization of focused range
        plt.figure(figsize=(12, 6))
        plt.plot(focus_time, focus_temporal_flow, lw=2, label=f'Temporal Flow, β = {beta}')
        plt.axvline(0, color='k', linestyle='--', label='Cycle Transition (t=0)')
        plt.title(f'Temporal Flow near Cycle Transition (β = {beta}, Zoomed In)')
        plt.xlabel('Time')
        plt.ylabel('Temporal Flow Tf(t)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        fig_path = os.path.join(output_dir, f'temporal_flow_beta_{beta}_zoom.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        
        print(f"Generated temporal flow data for β={beta} with min Tf={min_tf:.4f}, recovery time={recovery_time:.4f}")
    
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'temporal_flow_transition_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved temporal flow transition dataset to {csv_path}")
    
    # Generate summary dataset showing relationship between beta and min temporal flow
    summary_data = []
    for beta in beta_values:
        # Filter data for this beta
        beta_data = df[df['beta'] == beta]
        min_tf = beta_data['min_temporal_flow'].iloc[0]
        recovery_time = beta_data['recovery_time'].iloc[0]
        
        summary_data.append({
            'beta': beta,
            'min_temporal_flow': min_tf,
            'recovery_time': recovery_time
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'temporal_flow_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    # Create summary visualization
    plt.figure(figsize=(12, 6))
    plt.plot(summary_df['beta'], summary_df['min_temporal_flow'], 'bo-', label='Min Temporal Flow')
    plt.plot(summary_df['beta'], summary_df['recovery_time'], 'ro-', label='Recovery Time')
    plt.title('Effect of β Parameter on Temporal Flow Behavior at Cycle Transition')
    plt.xlabel('Beta (β)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'temporal_flow_beta_summary.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    return df

def main():
    """Main function to generate the temporal flow transition dataset"""
    print("Generating temporal flow transition datasets...")
    generate_temporal_flow_dataset()
    print("Done.")

if __name__ == "__main__":
    main()
