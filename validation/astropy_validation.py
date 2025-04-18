"""
Validates the Genesis-Sphere model against standard cosmological models from astropy.cosmology.

This module provides tools to compare the Genesis-Sphere model with established 
cosmological models like Planck18, WMAP9, etc., using the astropy.cosmology package.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, WMAP9, FlatLambdaCDM
import astropy.units as u
import pandas as pd

# Add parent directory to path to import the Genesis-Sphere model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'models'))

from genesis_model import GenesisSphereModel

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'astropy')
os.makedirs(results_dir, exist_ok=True)

def get_cosmology_model(model_name):
    """
    Returns the specified astropy cosmology model.
    
    Parameters:
    -----------
    model_name : str
        Name of the astropy cosmology model to use
        
    Returns:
    --------
    astropy.cosmology.FLRW
        The requested cosmology model
    """
    if model_name.lower() == 'planck18':
        return Planck18
    elif model_name.lower() == 'wmap9':
        return WMAP9
    elif model_name.lower().startswith('custom'):
        # Parse custom parameters from the name
        # Format: custom_H0_OmegaM_OmegaL
        params = model_name.split('_')
        if len(params) >= 4:
            H0 = float(params[1])
            OmegaM = float(params[2])
            OmegaL = float(params[3])
            return FlatLambdaCDM(H0=H0, Om0=OmegaM, name=f"Custom_{H0}_{OmegaM}")
    
    # Default to Planck18
    print(f"Model '{model_name}' not recognized, using Planck18 instead.")
    return Planck18

def genesis_sphere_to_astropy_mapping(gs_model, z_range):
    """
    Maps Genesis-Sphere model predictions to quantities comparable with astropy cosmology.
    
    Parameters:
    -----------
    gs_model : GenesisSphereModel
        The Genesis-Sphere model instance
    z_range : array-like
        Range of redshift values to map
        
    Returns:
    --------
    dict
        Dictionary containing mapped quantities
    """
    # Convert redshift to time in the Genesis-Sphere model's time units
    # This is a theoretical conversion and would need to be refined
    t_values = np.array([-10 * np.log(1 + z) for z in z_range])
    
    # Calculate all Genesis-Sphere functions
    results = gs_model.evaluate_all(t_values)
    
    # Map Genesis-Sphere quantities to cosmological quantities
    # These mappings are placeholder examples and would need to be properly derived
    
    # Calculate Hubble parameter H(z) - theoretical mapping
    density = results['density']
    temporal_flow = results['temporal_flow']
    velocity = results['velocity']
    pressure = results['pressure']
    
    # Example mappings (theoretical)
    hubble_parameter = 70 * np.sqrt(density) * temporal_flow
    matter_density = 0.3 * (pressure / density) * (1 + z_range)**3
    dark_energy_density = 0.7 * density * temporal_flow
    
    return {
        'redshift': z_range,
        'hubble_parameter': hubble_parameter,
        'matter_density': matter_density,
        'dark_energy_density': dark_energy_density,
        'density': density,
        'temporal_flow': temporal_flow,
        'velocity': velocity,
        'pressure': pressure
    }

def get_astropy_predictions(cosmo_model, z_range):
    """
    Gets predictions from an astropy cosmology model.
    
    Parameters:
    -----------
    cosmo_model : astropy.cosmology.FLRW
        The astropy cosmology model
    z_range : array-like
        Range of redshift values
        
    Returns:
    --------
    dict
        Dictionary containing astropy model predictions
    """
    # Calculate cosmological quantities
    hubble_parameter = cosmo_model.H(z_range).value  # in km/s/Mpc
    matter_density = cosmo_model.Om(z_range)
    dark_energy_density = cosmo_model.Ode(z_range)
    
    # Calculate other useful quantities
    critical_density = cosmo_model.critical_density(z_range).value  # in g/cm^3
    age = cosmo_model.age(z_range).value  # in Gyr
    
    return {
        'redshift': z_range,
        'hubble_parameter': hubble_parameter,
        'matter_density': matter_density,
        'dark_energy_density': dark_energy_density,
        'critical_density': critical_density,
        'age': age
    }

def compare_models(gs_data, astropy_data, comparison_type, gs_params, astropy_model_name):
    """
    Compares Genesis-Sphere and astropy predictions.
    
    Parameters:
    -----------
    gs_data : dict
        Genesis-Sphere model predictions
    astropy_data : dict
        Astropy cosmology model predictions
    comparison_type : str
        Type of comparison to perform
    gs_params : dict
        Genesis-Sphere model parameters
    astropy_model_name : str
        Name of the astropy cosmology model
        
    Returns:
    --------
    dict
        Dictionary with comparison metrics
    """
    z_range = gs_data['redshift']
    metrics = {}
    
    if comparison_type == 'hubble_evolution':
        # Compare Hubble parameter evolution
        plt.figure(figsize=(12, 6))
        plt.plot(z_range, gs_data['hubble_parameter'], 'b-', label='Genesis-Sphere')
        plt.plot(z_range, astropy_data['hubble_parameter'], 'r--', label=f'Astropy ({astropy_model_name})')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Hubble Parameter H(z) [km/s/Mpc]')
        plt.legend()
        plt.grid(True)
        plt.title(f"Hubble Parameter Evolution Comparison\nGenesis-Sphere (α={gs_params['alpha']}, β={gs_params['beta']}, ω={gs_params['omega']}, ε={gs_params['epsilon']})")
        plt.savefig(os.path.join(results_dir, f"hubble_evolution_{astropy_model_name}.png"), dpi=150)
        
        # Calculate metrics
        rel_diff = (gs_data['hubble_parameter'] - astropy_data['hubble_parameter']) / astropy_data['hubble_parameter']
        metrics['hubble_parameter'] = {
            'mean_rel_diff': np.mean(rel_diff),
            'max_rel_diff': np.max(rel_diff),
            'std_rel_diff': np.std(rel_diff)
        }
        
    elif comparison_type == 'density_evolution':
        # Compare matter density evolution
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(z_range, gs_data['matter_density'], 'b-', label='Genesis-Sphere (Matter)')
        plt.plot(z_range, astropy_data['matter_density'], 'r--', label=f'Astropy ({astropy_model_name}) (Matter)')
        plt.ylabel('Matter Density Parameter Ωₘ(z)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(z_range, gs_data['dark_energy_density'], 'g-', label='Genesis-Sphere (Dark Energy)')
        plt.plot(z_range, astropy_data['dark_energy_density'], 'm--', label=f'Astropy ({astropy_model_name}) (Dark Energy)')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Dark Energy Density Parameter ΩΛ(z)')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f"Density Evolution Comparison\nGenesis-Sphere (α={gs_params['alpha']}, β={gs_params['beta']}, ω={gs_params['omega']}, ε={gs_params['epsilon']})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"density_evolution_{astropy_model_name}.png"), dpi=150)
        
        # Calculate metrics
        metrics['matter_density'] = {
            'mean_rel_diff': np.mean((gs_data['matter_density'] - astropy_data['matter_density']) / astropy_data['matter_density']),
            'max_rel_diff': np.max(np.abs((gs_data['matter_density'] - astropy_data['matter_density']) / astropy_data['matter_density'])),
            'std_rel_diff': np.std((gs_data['matter_density'] - astropy_data['matter_density']) / astropy_data['matter_density'])
        }
        
        metrics['dark_energy_density'] = {
            'mean_rel_diff': np.mean((gs_data['dark_energy_density'] - astropy_data['dark_energy_density']) / astropy_data['dark_energy_density']),
            'max_rel_diff': np.max(np.abs((gs_data['dark_energy_density'] - astropy_data['dark_energy_density']) / astropy_data['dark_energy_density'])),
            'std_rel_diff': np.std((gs_data['dark_energy_density'] - astropy_data['dark_energy_density']) / astropy_data['dark_energy_density'])
        }
        
    elif comparison_type == 'gs_functions':
        # Plot Genesis-Sphere model functions
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(z_range, gs_data['density'], 'r-')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Space-Time Density ρ(t)')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(z_range, gs_data['temporal_flow'], 'b-')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Temporal Flow Tf(t)')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(z_range, gs_data['velocity'], 'g-')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Modulated Velocity v(t)')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(z_range, gs_data['pressure'], 'm-')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Modulated Pressure p(t)')
        plt.grid(True)
        
        plt.suptitle(f"Genesis-Sphere Functions vs. Redshift\n(α={gs_params['alpha']}, β={gs_params['beta']}, ω={gs_params['omega']}, ε={gs_params['epsilon']})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"gs_functions_{astropy_model_name}.png"), dpi=150)
        
        # No direct comparison metrics for this visualization
        metrics['gs_functions'] = {'plotted': True}
    
    return metrics

def generate_validation_summary(metrics, gs_params, astropy_model_name, comparison_type):
    """
    Generates an AI-like summary of the astropy validation results.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with comparison metrics
    gs_params : dict
        Genesis-Sphere model parameters
    astropy_model_name : str
        Name of the astropy cosmology model
    comparison_type : str
        Type of comparison performed
        
    Returns:
    --------
    str
        Summary of validation results
    """
    # Initialize summary
    summary = [f"## AI Validation Summary: Astropy {astropy_model_name} Comparison\n"]
    
    # Add parameter summary
    summary.append(f"### Genesis-Sphere Model Configuration")
    summary.append(f"- **Alpha (α)**: {gs_params['alpha']:.4f} - *Spatial dimension expansion coefficient*")
    summary.append(f"- **Beta (β)**: {gs_params['beta']:.4f} - *Temporal damping factor*")
    summary.append(f"- **Omega (ω)**: {gs_params['omega']:.4f} - *Angular frequency*")
    summary.append(f"- **Epsilon (ε)**: {gs_params['epsilon']:.4f} - *Zero-prevention constant*\n")
    
    summary.append(f"### Comparison Type: {comparison_type.replace('_', ' ').title()}\n")
    
    # Analyze each comparison metric
    if comparison_type != 'gs_functions':
        overall_assessment = []
        for metric, values in metrics.items():
            metric_name = metric.replace('_', ' ').title()
            summary.append(f"**{metric_name} Comparison**:")
            summary.append(f"- Mean Relative Difference: {values['mean_rel_diff']*100:.2f}%")
            summary.append(f"- Maximum Relative Difference: {abs(values['max_rel_diff'])*100:.2f}%")
            summary.append(f"- Standard Deviation: {values['std_rel_diff']*100:.2f}%\n")
            
            # Interpret this metric's comparison
            if abs(values['mean_rel_diff']) < 0.05:
                overall_assessment.append(f"{metric_name}: Excellent match")
                summary.append("- *Excellent agreement* with the {astropy_model_name} model")
            elif abs(values['mean_rel_diff']) < 0.15:
                overall_assessment.append(f"{metric_name}: Good match")
                summary.append("- *Good agreement* with the {astropy_model_name} model")
            elif abs(values['mean_rel_diff']) < 0.25:
                overall_assessment.append(f"{metric_name}: Moderate match")
                summary.append("- *Moderate agreement* with the {astropy_model_name} model")
            else:
                overall_assessment.append(f"{metric_name}: Significant differences")
                summary.append("- *Significant differences* from the {astropy_model_name} model\n")
        
        # Add overall interpretation
        summary.append(f"### Overall Assessment")
        for assessment in overall_assessment:
            summary.append(f"- {assessment}")
        
        # Aggregated interpretation
        if comparison_type == 'hubble_evolution':
            summary.append("\n### Interpretation of Hubble Evolution Comparison")
            if any("Significant differences" in a for a in overall_assessment):
                summary.append("The Genesis-Sphere model shows **notable differences** in predicting the expansion history of the universe compared to the standard {astropy_model_name} model. This suggests:")
                summary.append("- The current parameter mapping between Genesis-Sphere and standard cosmology may need refinement")
                summary.append("- The temporal evolution equations in Genesis-Sphere may capture different physics")
                summary.append("- Further investigation into how the temporal flow function Tf(t) relates to cosmic expansion is warranted")
            else:
                summary.append("The Genesis-Sphere model **effectively reproduces** the Hubble parameter evolution of the standard {astropy_model_name} model. This suggests:")
                summary.append("- The current parameter mapping successfully translates between frameworks")
                summary.append("- Genesis-Sphere could serve as an alternative mathematical formulation with similar predictive power")
                summary.append("- Focus should shift to areas where the models might diverge at extreme regimes")
        
        elif comparison_type == 'density_evolution':
            summary.append("\n### Interpretation of Density Evolution Comparison")
            if any("Significant differences" in a for a in overall_assessment):
                summary.append("The Genesis-Sphere model shows **different density evolution characteristics** compared to the standard {astropy_model_name} model. This suggests:")
                summary.append("- The Genesis-Sphere density function ρ(t) may capture different physics than the standard ΛCDM density parameters")
                summary.append("- The redshift-to-time mapping may need refinement")
                summary.append("- Further theoretical development on how Genesis-Sphere relates to matter and energy density is needed")
            else:
                summary.append("The Genesis-Sphere model **successfully reproduces** the density evolution patterns of the standard {astropy_model_name} model. This suggests:")
                summary.append("- The space-time density function ρ(t) correctly maps to cosmological density parameters")
                summary.append("- The sinusoidal and quadratic components effectively model cosmological density evolution")
    else:
        # For gs_functions comparison type
        summary.append("### Genesis-Sphere Functions Analysis")
        summary.append("This visualization shows the behavior of core Genesis-Sphere functions across redshift:")
        summary.append("- The space-time density function ρ(t)")
        summary.append("- The temporal flow function Tf(t)")
        summary.append("- Derived modulated velocity v(t)")
        summary.append("- Derived modulated pressure p(t)")
        summary.append("\nThese plots provide insight into how Genesis-Sphere quantities evolve with redshift, which can be compared conceptually with standard cosmological evolution.")
    
    # Add recommendations
    summary.append("\n### Recommendations")
    if comparison_type != 'gs_functions':
        if any("Significant differences" in a for a in overall_assessment):
            summary.append("1. **Revisit parameter mapping** between Genesis-Sphere and standard cosmology")
            summary.append("2. **Explore alternative mathematical relationships** for converting between frameworks")
            summary.append("3. **Conduct parameter optimization** to find values that better match {astropy_model_name}")
            summary.append("4. **Focus theoretical development** on the areas with largest discrepancies")
        else:
            summary.append("1. **Document this parameter set** as it provides good agreement with {astropy_model_name}")
            summary.append("2. **Explore predictive differences** at higher redshifts where models might diverge")
            summary.append("3. **Investigate edge cases** like the early universe where Genesis-Sphere might offer novel insights")
            summary.append("4. **Test against observational data** to validate both Genesis-Sphere and standard models")
    else:
        summary.append("1. **Map Genesis-Sphere functions** to standard cosmological observables")
        summary.append("2. **Identify regions of unique behavior** where the model might differ from standard cosmology")
        summary.append("3. **Explore physical interpretations** of the function behaviors in terms of cosmic evolution")
    
    # Join all parts of the summary with newlines
    return "\n".join(summary)

def main(alpha, beta, omega, epsilon, model, compare):
    """Main function to run the validation"""
    print("Genesis-Sphere Model Validation Against Astropy Cosmology Models")
    print("=============================================================")
    
    # Initialize the Genesis-Sphere model
    gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
    gs_params = {'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon}
    
    # Get the astropy cosmology model
    cosmo_model = get_cosmology_model(model)
    print(f"\nAstropy cosmology model: {cosmo_model.name}")
    print(f"H₀ = {cosmo_model.H0.value:.2f} km/s/Mpc")
    print(f"Ωₘ = {cosmo_model.Om0:.4f}")
    print(f"ΩΛ = {cosmo_model.Ode0:.4f}")
    
    # Define redshift range
    z_range = np.linspace(0.01, 2.0, 100)
    
    # Get Genesis-Sphere predictions mapped to cosmological quantities
    gs_data = genesis_sphere_to_astropy_mapping(gs_model, z_range)
    
    # Get astropy model predictions
    astropy_data = get_astropy_predictions(cosmo_model, z_range)
    
    # Compare models
    metrics = compare_models(gs_data, astropy_data, compare, gs_params, cosmo_model.name)
    
    # Generate AI summary
    summary = generate_validation_summary(metrics, gs_params, cosmo_model.name, compare)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, f"astropy_validation_summary_{model}_{compare}.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Print summary to console
    print("\n" + summary)
    
    # Print comparison metrics
    print("\nComparison Metrics:")
    for metric, values in metrics.items():
        if metric != 'gs_functions':  # Skip non-comparative metrics
            print(f"\n{metric.replace('_', ' ').title()}:")
            for stat, val in values.items():
                print(f"  {stat.replace('_', ' ').title()}: {val*100:.2f}%")
    
    # Save metrics to file
    metrics_df = pd.DataFrame()
    for metric, values in metrics.items():
        if metric != 'gs_functions':  # Skip non-comparative metrics
            for stat, val in values.items():
                metrics_df.loc[metric, stat] = val
    
    if not metrics_df.empty:
        metrics_df.to_csv(os.path.join(results_dir, f"validation_metrics_{model}_{compare}.csv"), index=True, encoding='utf-8')
    
    print(f"\nResults and visualizations saved to: {results_dir}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Genesis-Sphere model against astropy cosmology models")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=0.8, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=1.0, help="Angular frequency for sinusoidal projections")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    parser.add_argument("--model", type=str, default="Planck18", help="Astropy cosmology model to use (Planck18, WMAP9, custom_H0_OmegaM_OmegaL)")
    parser.add_argument("--compare", type=str, default="hubble_evolution", 
                        choices=["hubble_evolution", "density_evolution", "gs_functions"],
                        help="Type of comparison to perform")
    
    args = parser.parse_args()
    main(args.alpha, args.beta, args.omega, args.epsilon, args.model, args.compare)
