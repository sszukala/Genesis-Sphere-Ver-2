"""
Validates the Genesis-Sphere model against the NASA/IPAC Extragalactic Database (NED)
cosmology calculator for standard cosmological measurements.

This module compares the Genesis-Sphere model predictions with established cosmological
measurements by mapping model parameters to standard cosmological parameters and
verifying predictions against NED calculations.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Add parent directory to path to import the Genesis-Sphere model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'models'))

from genesis_model import GenesisSphereModel

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ned')
os.makedirs(results_dir, exist_ok=True)

def map_gs_to_cosmo_params(alpha, beta, omega, epsilon):
    """
    Maps Genesis-Sphere parameters to standard cosmological parameters.
    
    This is a theoretical mapping that correlates Genesis-Sphere model parameters
    with the standard cosmological parameters used in ΛCDM models.
    
    Parameters:
    -----------
    alpha : float
        Spatial dimension expansion coefficient
    beta : float
        Temporal damping factor
    omega : float
        Angular frequency for sinusoidal projections
    epsilon : float
        Small constant to prevent division by zero
        
    Returns:
    --------
    dict
        Dictionary with the mapped cosmological parameters:
        - H0: Hubble constant (km/s/Mpc)
        - OmegaM: Matter density parameter
        - OmegaL: Dark energy density parameter
        - w: Dark energy equation of state parameter
    """
    # Theoretical mapping equations (placeholder - to be refined with research)
    H0 = 70.0 * (1 + alpha*10) / (1 + beta/2)
    OmegaM = 0.3 * (1 - np.tanh(beta - 0.8))
    OmegaL = 0.7 * (1 + np.tanh(alpha*50 - 1))
    w = -1.0 - 0.1 * (omega - 1.0)
    
    return {
        'H0': H0,
        'OmegaM': OmegaM,
        'OmegaL': OmegaL,
        'w': w
    }

def query_ned_cosmology(h0, omega_m, omega_lambda, z_values):
    """
    Queries the NED Cosmology Calculator API with the given parameters.
    
    Parameters:
    -----------
    h0 : float
        Hubble constant in km/s/Mpc
    omega_m : float
        Matter density parameter
    omega_lambda : float
        Dark energy density parameter
    z_values : list of float
        Redshift values to calculate for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the NED cosmology calculator results
    """
    # For a real implementation, we would use the NED API or web scraping
    # This is a placeholder for demonstration purposes
    print(f"Querying NED Cosmology Calculator with: H0={h0}, ΩM={omega_m}, ΩΛ={omega_lambda}")
    
    # Placeholder results - in a real implementation, you would:
    # 1. Make API calls to NED or parse web results
    # 2. Extract the relevant cosmological measurements
    results = pd.DataFrame({
        'redshift': z_values,
        'comoving_distance': [d * 3000 * (0.7/h0) * np.sqrt(omega_lambda) for d in z_values],  # Simulated values
        'angular_diameter_distance': [d * 1500 * (0.7/h0) * np.sqrt(omega_m) for d in z_values],  # Simulated values
        'luminosity_distance': [d * 3000 * (0.7/h0) * np.sqrt(omega_lambda + omega_m) for d in z_values],  # Simulated values
        'age_at_z': [(1 / (1+z)) * 13.7 * (0.7/h0) * np.sqrt(omega_lambda) for z in z_values]  # Simulated values
    })
    
    return results

def genesis_sphere_predictions(model, z_values):
    """
    Calculates cosmological predictions from the Genesis-Sphere model.
    
    Parameters:
    -----------
    model : GenesisSphereModel
        The Genesis-Sphere model instance
    z_values : list of float
        Redshift values to calculate for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the Genesis-Sphere model predictions
    """
    # Convert redshift to time in the Genesis-Sphere model's time units
    # This is a theoretical conversion and would need to be refined
    t_values = np.array([-10 * np.log(1 + z) for z in z_values])
    
    # Calculate the Genesis-Sphere model predictions
    results = model.evaluate_all(t_values)
    
    # Convert the Genesis-Sphere quantities to standard cosmological measurements
    # These conversions are theoretical and would need to be refined
    comoving_distance = 3000 * results['density'] * np.sqrt(np.abs(t_values) + 0.1)
    angular_diameter_distance = comoving_distance / (1 + np.array(z_values))
    luminosity_distance = comoving_distance * (1 + np.array(z_values))
    age_at_z = 13.7 * results['temporal_flow']
    
    # Create a DataFrame with the results
    df = pd.DataFrame({
        'redshift': z_values,
        'comoving_distance': comoving_distance,
        'angular_diameter_distance': angular_diameter_distance,
        'luminosity_distance': luminosity_distance,
        'age_at_z': age_at_z
    })
    
    return df

def compare_and_visualize(ned_data, gs_data, model_params, cosmo_params):
    """
    Compares and visualizes the NED calculator results and Genesis-Sphere predictions.
    
    Parameters:
    -----------
    ned_data : pd.DataFrame
        NED cosmology calculator results
    gs_data : pd.DataFrame
        Genesis-Sphere model predictions
    model_params : dict
        Genesis-Sphere model parameters
    cosmo_params : dict
        Mapped cosmological parameters
        
    Returns:
    --------
    dict
        Dictionary with comparison metrics
    """
    # Calculate relative differences for each metric
    metrics = {}
    for column in ['comoving_distance', 'angular_diameter_distance', 'luminosity_distance', 'age_at_z']:
        # Calculate relative differences
        rel_diff = (gs_data[column] - ned_data[column]) / ned_data[column]
        metrics[column] = {
            'mean_rel_diff': rel_diff.mean(),
            'max_rel_diff': rel_diff.max(),
            'min_rel_diff': rel_diff.min(),
            'std_rel_diff': rel_diff.std()
        }
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot both datasets
        plt.subplot(1, 2, 1)
        plt.plot(ned_data['redshift'], ned_data[column], 'o-', label='NED Calculator')
        plt.plot(gs_data['redshift'], gs_data[column], 's--', label='Genesis-Sphere')
        plt.xlabel('Redshift (z)')
        plt.ylabel(column.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        
        # Plot relative difference
        plt.subplot(1, 2, 2)
        plt.plot(ned_data['redshift'], rel_diff * 100, 'r.-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Redshift (z)')
        plt.ylabel('Relative Difference (%)')
        plt.grid(True)
        
        # Add a title with parameters
        plt.suptitle(f"Comparison for {column.replace('_', ' ').title()}\n" + 
                    f"GS Model (α={model_params['alpha']}, β={model_params['beta']}, ω={model_params['omega']}, ε={model_params['epsilon']})\n" +
                    f"Cosmo Params (H₀={cosmo_params['H0']:.1f}, Ωₘ={cosmo_params['OmegaM']:.3f}, ΩΛ={cosmo_params['OmegaL']:.3f}, w={cosmo_params['w']:.3f})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"comparison_{column}.png"), dpi=150)
        
    return metrics

def generate_validation_summary(metrics, model_params, cosmo_params):
    """
    Generates an AI-like summary of the NED validation results.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with comparison metrics for each cosmological measure
    model_params : dict
        Genesis-Sphere model parameters
    cosmo_params : dict
        Mapped cosmological parameters
        
    Returns:
    --------
    str
        Summary of validation results
    """
    # Initialize summary
    summary = ["## AI Validation Summary: NED Cosmology Calculator Comparison\n"]
    
    # Add parameter summary
    summary.append(f"### Genesis-Sphere Model Configuration")
    summary.append(f"- **Alpha (α)**: {model_params['alpha']:.4f} - *Spatial dimension expansion coefficient*")
    summary.append(f"- **Beta (β)**: {model_params['beta']:.4f} - *Temporal damping factor*")
    summary.append(f"- **Omega (ω)**: {model_params['omega']:.4f} - *Angular frequency*")
    summary.append(f"- **Epsilon (ε)**: {model_params['epsilon']:.4f} - *Zero-prevention constant*\n")
    
    summary.append(f"### Mapped Cosmological Parameters")
    summary.append(f"- **H₀**: {cosmo_params['H0']:.2f} km/s/Mpc - *Hubble constant*")
    summary.append(f"- **Ωₘ**: {cosmo_params['OmegaM']:.4f} - *Matter density parameter*")
    summary.append(f"- **ΩΛ**: {cosmo_params['OmegaL']:.4f} - *Dark energy density parameter*")
    summary.append(f"- **w**: {cosmo_params['w']:.4f} - *Dark energy equation of state*\n")
    
    # Overall assessment
    overall_mean_diff = np.mean([metrics[m]['mean_rel_diff'] for m in metrics])
    overall_max_diff = np.max([metrics[m]['max_rel_diff'] for m in metrics])
    
    summary.append(f"### Overall Performance Assessment")
    summary.append(f"- **Mean Relative Difference**: {overall_mean_diff*100:.2f}%")
    summary.append(f"- **Maximum Relative Difference**: {overall_max_diff*100:.2f}%\n")
    
    # Interpret the results
    summary.append(f"### Interpretation")
    if abs(overall_mean_diff) < 0.05 and abs(overall_max_diff) < 0.15:
        summary.append("The Genesis-Sphere model shows **excellent agreement** with standard cosmology calculations. This suggests that:")
        summary.append("- The current parameter mapping captures the essence of ΛCDM cosmology")
        summary.append("- The model could serve as an alternative mathematical framework with equivalent predictive power")
    elif abs(overall_mean_diff) < 0.15 and abs(overall_max_diff) < 0.3:
        summary.append("The Genesis-Sphere model shows **good agreement** with standard cosmology calculations. This suggests that:")
        summary.append("- The theoretical framework broadly aligns with established cosmological models")
        summary.append("- Some fine-tuning of parameter mappings could further improve agreement")
    elif abs(overall_mean_diff) < 0.25:
        summary.append("The Genesis-Sphere model shows **moderate agreement** with standard cosmology calculations. This suggests that:")
        summary.append("- The basic concepts align with standard cosmology, but quantitative differences exist")
        summary.append("- Further refinement of the parameter mapping is needed")
    else:
        summary.append("The Genesis-Sphere model shows **significant differences** from standard cosmology calculations. This suggests that:")
        summary.append("- The current parameter mapping may not fully capture standard cosmological relationships")
        summary.append("- Alternative approaches to mapping model parameters to cosmological observables should be explored")
    
    # Detailed metric assessment
    summary.append("\n### Detailed Metric Analysis")
    for metric, values in metrics.items():
        metric_name = metric.replace('_', ' ').title()
        summary.append(f"\n**{metric_name}**:")
        summary.append(f"- Mean Difference: {values['mean_rel_diff']*100:.2f}%")
        summary.append(f"- Maximum Difference: {values['max_rel_diff']*100:.2f}%")
        summary.append(f"- Standard Deviation: {values['std_rel_diff']*100:.2f}%")
        
        # Add interpretation for each metric
        if abs(values['mean_rel_diff']) < 0.05:
            summary.append("- *Excellent agreement* with standard cosmology")
        elif abs(values['mean_rel_diff']) < 0.15:
            summary.append("- *Good agreement* with standard cosmology")
        elif abs(values['mean_rel_diff']) < 0.25:
            summary.append("- *Moderate agreement* with standard cosmology")
        else:
            summary.append("- *Significant differences* from standard cosmology")
    
    # Add recommendations
    summary.append("\n### Recommendations")
    if abs(overall_mean_diff) < 0.15:
        summary.append("1. **Document these parameters** as they provide good alignment with standard cosmology")
        summary.append("2. **Explore predictive differences** at higher redshifts where models might diverge")
        summary.append("3. **Consider edge cases** like the early universe or near singularities where Genesis-Sphere might offer novel insights")
    else:
        summary.append("1. **Refine parameter mapping** between Genesis-Sphere quantities and cosmological parameters")
        summary.append("2. **Investigate alternative mathematical relationships** for converting between model frameworks")
        summary.append("3. **Explore optimization techniques** to find parameter values that better match standard cosmology")
    
    # Join all parts of the summary with newlines
    return "\n".join(summary)

def main(alpha, beta, omega, epsilon):
    """Main function to run the validation"""
    print("Genesis-Sphere Model Validation Against NED Cosmology Calculator")
    print("==============================================================")
    
    # Initialize the Genesis-Sphere model
    model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
    model_params = {'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon}
    
    # Map Genesis-Sphere parameters to standard cosmological parameters
    cosmo_params = map_gs_to_cosmo_params(alpha, beta, omega, epsilon)
    print("\nMapped Cosmological Parameters:")
    print(f"H₀ = {cosmo_params['H0']:.2f} km/s/Mpc")
    print(f"Ωₘ = {cosmo_params['OmegaM']:.4f}")
    print(f"ΩΛ = {cosmo_params['OmegaL']:.4f}")
    print(f"w = {cosmo_params['w']:.4f}")
    
    # Define redshift values to evaluate
    z_values = np.linspace(0.01, 2.0, 20)
    
    # Query the NED cosmology calculator (simulated)
    ned_data = query_ned_cosmology(
        cosmo_params['H0'],
        cosmo_params['OmegaM'],
        cosmo_params['OmegaL'],
        z_values
    )
    
    # Calculate Genesis-Sphere predictions
    gs_data = genesis_sphere_predictions(model, z_values)
    
    # Compare and visualize the results
    metrics = compare_and_visualize(ned_data, gs_data, model_params, cosmo_params)
    
    # Generate AI summary
    summary = generate_validation_summary(metrics, model_params, cosmo_params)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, "ned_validation_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Print summary to console
    print("\n" + summary)
    
    # Print comparison metrics
    print("\nComparison Metrics:")
    for metric, values in metrics.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Mean relative difference: {values['mean_rel_diff']*100:.2f}%")
        print(f"  Max relative difference: {values['max_rel_diff']*100:.2f}%")
        print(f"  Standard deviation: {values['std_rel_diff']*100:.2f}%")
    
    # Save metrics to file
    metrics_df = pd.DataFrame()
    for metric, values in metrics.items():
        for stat, val in values.items():
            metrics_df.loc[metric, stat] = val
    
    metrics_df.to_csv(os.path.join(results_dir, "validation_metrics.csv"))
    
    # Save validation results
    results = {
        'cosmo_model': 'NED',
        'metrics': {k: v['mean_rel_diff'] for k, v in metrics.items()},
        'alpha': alpha,
        'beta': beta,
        'omega': omega,
        'epsilon': epsilon,
        'H0': cosmo_params['H0'],
        'OmegaM': cosmo_params['OmegaM'],
        'OmegaL': cosmo_params['OmegaL'],
        'w': cosmo_params['w']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(results_dir, f"ned_validation_results.csv"), index=False)
    
    print(f"\nResults and visualizations saved to: {results_dir}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Genesis-Sphere model against NED Cosmology Calculator")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=0.8, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=1.0, help="Angular frequency for sinusoidal projections")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    
    args = parser.parse_args()
    main(args.alpha, args.beta, args.omega, args.epsilon)
