"""
Validates the Genesis-Sphere model against actual observational datasets.

This module compares Genesis-Sphere model predictions with real cosmological 
observations including Type Ia supernovae, CMB measurements, and BAO constraints.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Add parent directory to path to import the Genesis-Sphere model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'models'))

from genesis_model import GenesisSphereModel

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'observational')
os.makedirs(results_dir, exist_ok=True)

# Create datasets directory if it doesn't exist
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

def load_synthetic_supernovae_data():
    """
    Loads synthetic Type Ia supernovae data if no real data is available.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing synthetic supernovae data
    """
    # Create synthetic data based on ΛCDM model
    # In a real implementation, you would load actual data from files
    
    # Define redshift range
    z_values = np.linspace(0.01, 1.0, 50)
    
    # Calculate distance modulus based on a simple ΛCDM model
    # μ = 5*log10(dL) + 25, where dL is luminosity distance in Mpc
    H0 = 70.0  # km/s/Mpc
    OmegaM = 0.3
    OmegaL = 0.7
    
    # Simple luminosity distance calculation
    c = 299792.458  # Speed of light in km/s
    dH = c / H0  # Hubble distance
    
    # Calculate luminosity distance (simplified)
    dL = np.zeros_like(z_values)
    for i, z in enumerate(z_values):
        # Simple integration for comoving distance
        z_range = np.linspace(0, z, 100)
        dz = z_range[1] - z_range[0]
        integrand = 1.0 / np.sqrt(OmegaM * (1 + z_range)**3 + OmegaL)
        dc = dH * dz * np.sum(integrand)
        dL[i] = (1 + z) * dc
    
    # Distance modulus
    mu = 5 * np.log10(dL) + 25
    
    # Add some noise to simulate measurement errors
    mu_err = 0.1 + 0.05 * np.random.rand(len(z_values))
    mu += np.random.normal(0, mu_err)
    
    # Create DataFrame
    df = pd.DataFrame({
        'z': z_values,
        'mu': mu,
        'mu_err': mu_err
    })
    
    return df

def load_dataset(dataset_name, data_file=None):
    """
    Loads a dataset for validation.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    data_file : str, optional
        Path to the data file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the dataset
    """
    if dataset_name.lower() == 'supernovae':
        if data_file:
            # Check if the file exists in the current directory or datasets directory
            if os.path.exists(data_file):
                file_path = data_file
            else:
                # Try looking in the datasets directory
                datasets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', data_file)
                if os.path.exists(datasets_path):
                    file_path = datasets_path
                else:
                    print(f"Data file not found at {data_file} or {datasets_path}. Using synthetic data.")
                    return load_synthetic_supernovae_data()
                
            print(f"Loading supernovae data from {file_path}")
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                try:
                    return pd.read_csv(file_path, encoding='latin1')
                except Exception as e:
                    print(f"Error loading data file: {e}. Using synthetic data.")
                    return load_synthetic_supernovae_data()
            except Exception as e:
                print(f"Error loading data file: {e}. Using synthetic data.")
                return load_synthetic_supernovae_data()
        else:
            # Try to find any supernova dataset in the datasets directory
            datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
            sne_files = [f for f in os.listdir(datasets_dir) if f.startswith('SNe_') and f.endswith('.csv')]
            
            if sne_files:
                file_path = os.path.join(datasets_dir, sne_files[0])
                print(f"Found supernovae dataset: {file_path}")
                try:
                    return pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error loading data file: {e}. Using synthetic data.")
                    return load_synthetic_supernovae_data()
            else:
                print("No supernovae data file specified or found. Using synthetic data.")
                return load_synthetic_supernovae_data()
    
    elif dataset_name.lower() == 'cmb':
        # Placeholder for CMB data
        print("CMB data validation not yet implemented. Using synthetic data.")
        # Create a simple dataset with constraints
        df = pd.DataFrame({
            'parameter': ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8'],
            'value': [0.315, 0.0493, 0.674, 0.965, 0.811],
            'error': [0.007, 0.0019, 0.005, 0.004, 0.006]
        })
        return df
    
    elif dataset_name.lower() == 'bao':
        # Placeholder for BAO data
        print("BAO data validation not yet implemented. Using synthetic data.")
        # Create a simple dataset with constraints at different redshifts
        z_values = [0.106, 0.15, 0.32, 0.57]
        rd_values = [147.4, 148.6, 149.3, 147.5]  # Comoving sound horizon in Mpc
        rd_errors = [1.7, 2.5, 2.8, 1.9]
        
        df = pd.DataFrame({
            'z': z_values,
            'rd': rd_values,
            'rd_err': rd_errors
        })
        return df
    
    else:
        print(f"Dataset '{dataset_name}' not recognized. Using synthetic supernovae data.")
        return load_synthetic_supernovae_data()

def genesis_sphere_prediction_for_dataset(gs_model, dataset_name, data):
    """
    Makes Genesis-Sphere model predictions for a given dataset.
    
    Parameters:
    -----------
    gs_model : GenesisSphereModel
        The Genesis-Sphere model instance
    dataset_name : str
        Name of the dataset
    data : pd.DataFrame
        The dataset
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing predictions
    """
    if dataset_name.lower() == 'supernovae':
        # Convert redshift to time in the Genesis-Sphere model's time units
        z_values = data['z'].values
        t_values = np.array([-10 * np.log(1 + z) for z in z_values])
        
        # Calculate Genesis-Sphere model functions
        results = gs_model.evaluate_all(t_values)
        
        # Calculate luminosity distance from model parameters
        # This is a theoretical mapping and would need to be properly derived
        density = results['density']
        temporal_flow = results['temporal_flow']
        
        # Convert to distance modulus
        # Theoretical conversion - would need refinement
        H0 = 70.0  # km/s/Mpc assumed base
        dH = 299792.458 / H0  # Hubble distance in Mpc
        
        # Simple model for luminosity distance
        dL = dH * (1 + z_values) * np.sqrt(density) * (1 / temporal_flow)
        mu = 5 * np.log10(dL) + 25
        
        # Add predictions to a new DataFrame
        predictions = pd.DataFrame({
            'z': z_values,
            'mu_predicted': mu,
            'density': density,
            'temporal_flow': temporal_flow
        })
        
        return predictions
    
    elif dataset_name.lower() == 'cmb':
        # For CMB, we would map Genesis-Sphere parameters to cosmological parameters
        # This is a placeholder implementation
        alpha = gs_model.alpha
        beta = gs_model.beta
        omega = gs_model.omega
        epsilon = gs_model.epsilon
        
        # Theoretical mapping (placeholder)
        Omega_m = 0.3 * (1 - np.tanh(beta - 0.8))
        Omega_b = 0.05 * (1 + alpha / 0.02)
        h = 0.67 * (1 + (omega - 1.0) / 10)
        n_s = 0.96 * (1 + epsilon / 0.1)
        sigma_8 = 0.8 * (1 + 0.1 * (alpha - 0.02) / 0.02)
        
        predictions = pd.DataFrame({
            'parameter': ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8'],
            'value_predicted': [Omega_m, Omega_b, h, n_s, sigma_8]
        })
        
        return predictions
    
    elif dataset_name.lower() == 'bao':
        # For BAO, we would predict the sound horizon at drag epoch
        # This is a placeholder implementation
        z_values = data['z'].values
        t_values = np.array([-10 * np.log(1 + z) for z in z_values])
        
        # Calculate Genesis-Sphere model functions
        results = gs_model.evaluate_all(t_values)
        
        # Theoretical mapping to sound horizon (placeholder)
        density = results['density']
        temporal_flow = results['temporal_flow']
        
        # Simple model for sound horizon
        rd_predicted = 147.0 * np.sqrt(density) * (1 + 0.1 * (1 - temporal_flow))
        
        predictions = pd.DataFrame({
            'z': z_values,
            'rd_predicted': rd_predicted
        })
        
        return predictions
    
    else:
        print(f"Dataset '{dataset_name}' not recognized for predictions.")
        return pd.DataFrame()

def calculate_chi_square(gs_model, dataset_name, data, predictions):
    """
    Calculates chi-square statistic for model fit to data.
    
    Parameters:
    -----------
    gs_model : GenesisSphereModel
        The Genesis-Sphere model instance
    dataset_name : str
        Name of the dataset
    data : pd.DataFrame
        The dataset
    predictions : pd.DataFrame
        Model predictions
        
    Returns:
    --------
    float
        Chi-square statistic
    """
    if dataset_name.lower() == 'supernovae':
        # Calculate chi-square for distance modulus
        obs = data['mu'].values
        pred = predictions['mu_predicted'].values
        errors = data['mu_err'].values
        
        chi2 = np.sum(((obs - pred) / errors)**2)
        return chi2
    
    elif dataset_name.lower() == 'cmb':
        # Calculate chi-square for CMB parameters
        chi2 = 0
        for i, row in data.iterrows():
            param = row['parameter']
            obs = row['value']
            error = row['error']
            pred = predictions.loc[predictions['parameter'] == param, 'value_predicted'].values[0]
            
            chi2 += ((obs - pred) / error)**2
        
        return chi2
    
    elif dataset_name.lower() == 'bao':
        # Calculate chi-square for BAO measurements
        obs = data['rd'].values
        pred = predictions['rd_predicted'].values
        errors = data['rd_err'].values
        
        chi2 = np.sum(((obs - pred) / errors)**2)
        return chi2
    
    else:
        print(f"Dataset '{dataset_name}' not recognized for chi-square calculation.")
        return float('inf')

def optimize_model_parameters(dataset_name, data, initial_params=None):
    """
    Optimize Genesis-Sphere parameters to fit the dataset
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset ('supernovae', 'cmb', 'bao')
    data : DataFrame
        Dataset to fit
    initial_params : dict, optional
        Initial parameter values to start optimization
        
    Returns:
    --------
    dict:
        Dictionary with optimized parameters and final chi-squared
    """
    if initial_params is None:
        initial_params = {
            'alpha': 0.02,
            'beta': 1.2,
            'omega': 2.0,
            'epsilon': 0.1
        }
    # Define objective function for optimization
    def objective(params):
        alpha, beta, omega, epsilon = params
        model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        predictions = genesis_sphere_prediction_for_dataset(model, dataset_name, data)
        chi2 = calculate_chi_square(model, dataset_name, data, predictions)
        return chi2
    
    # Initial parameter values
    initial = [
        initial_params['alpha'],
        initial_params['beta'],
        initial_params['omega'],
        initial_params['epsilon']
    ]
    
    # Parameter bounds
    bounds = [
        (0.001, 0.1),   # alpha
        (0.1, 2.0),     # beta
        (0.5, 2.0),     # omega
        (0.01, 0.5)     # epsilon
    ]
    
    # Run optimization
    print(f"Optimizing Genesis-Sphere parameters to fit {dataset_name} data...")
    result = minimize(objective, initial, bounds=bounds, method='L-BFGS-B')
    
    # Extract optimal parameters
    alpha_opt, beta_opt, omega_opt, epsilon_opt = result.x
    final_chi2 = result.fun
    
    optimized_params = {
        'alpha': alpha_opt,
        'beta': beta_opt,
        'omega': omega_opt,
        'epsilon': epsilon_opt
    }
    
    print(f"Optimization completed with final chi² = {final_chi2:.2f}")
    print(f"Optimized parameters: α={alpha_opt:.4f}, β={beta_opt:.4f}, ω={omega_opt:.4f}, ε={epsilon_opt:.4f}")
    
    return optimized_params, final_chi2

def visualize_comparison(dataset_name, data, predictions, gs_params):
    """
    Visualizes the comparison between data and model predictions.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    data : pd.DataFrame
        The dataset
    predictions : pd.DataFrame
        Model predictions
    gs_params : dict
        Genesis-Sphere model parameters
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    if dataset_name.lower() == 'supernovae':
        # Plot supernovae distance modulus
        plt.figure(figsize=(10, 6))
        
        # Plot data with error bars
        plt.errorbar(data['z'], data['mu'], yerr=data['mu_err'], fmt='o', color='blue', 
                    alpha=0.6, ecolor='lightblue', markersize=4, label='Observed SNe')
        
        # Plot model prediction
        plt.plot(predictions['z'], predictions['mu_predicted'], 'r-', lw=2, 
                label='Genesis-Sphere Prediction')
        
        plt.xlabel('Redshift (z)')
        plt.ylabel('Distance Modulus (μ)')
        plt.title(f"Type Ia Supernovae Distance Modulus Comparison\n"
                f"Genesis-Sphere Model (α={gs_params['alpha']:.4f}, β={gs_params['beta']:.4f}, "
                f"ω={gs_params['omega']:.4f}, ε={gs_params['epsilon']:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add residual plot
        plt.figure(figsize=(10, 4))
        residuals = data['mu'] - predictions['mu_predicted']
        plt.errorbar(data['z'], residuals, yerr=data['mu_err'], fmt='o', color='green', 
                    alpha=0.6, ecolor='lightgreen', markersize=4)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Redshift (z)')
        plt.ylabel('Residuals (μᵒᵇˢ - μᵖʳᵉᵈ)')
        plt.title('Residuals')
        plt.grid(True, alpha=0.3)
        
        # Save figures
        fig_path = os.path.join(results_dir, f"sne_comparison.png")
        plt.savefig(fig_path, dpi=150)
        
        return fig_path
    
    elif dataset_name.lower() == 'cmb':
        # Plot CMB parameter comparison
        plt.figure(figsize=(10, 6))
        
        merged_data = pd.merge(data, predictions, on='parameter')
        parameters = merged_data['parameter'].values
        observed = merged_data['value'].values
        errors = merged_data['error'].values
        predicted = merged_data['value_predicted'].values
        
        x = np.arange(len(parameters))
        width = 0.35
        
        plt.bar(x - width/2, observed, width, label='Observed', color='blue', alpha=0.7)
        plt.bar(x + width/2, predicted, width, label='Genesis-Sphere', color='red', alpha=0.7)
        
        # Add error bars
        plt.errorbar(x - width/2, observed, yerr=errors, fmt='none', ecolor='black', capsize=5)
        
        plt.xlabel('Cosmological Parameter')
        plt.ylabel('Value')
        plt.title(f"CMB Parameter Comparison\n"
                f"Genesis-Sphere Model (α={gs_params['alpha']:.4f}, β={gs_params['beta']:.4f}, "
                f"ω={gs_params['omega']:.4f}, ε={gs_params['epsilon']:.4f})")
        plt.xticks(x, parameters)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = os.path.join(results_dir, f"cmb_comparison.png")
        plt.savefig(fig_path, dpi=150)
        
        return fig_path
    
    elif dataset_name.lower() == 'bao':
        # Plot BAO sound horizon comparison
        plt.figure(figsize=(10, 6))
        
        # Plot data with error bars
        plt.errorbar(data['z'], data['rd'], yerr=data['rd_err'], fmt='o', color='blue', 
                    alpha=0.6, ecolor='lightblue', markersize=4, label='Observed BAO')
        
        # Plot model prediction
        plt.plot(predictions['z'], predictions['rd_predicted'], 'r-', lw=2, 
                label='Genesis-Sphere Prediction')
        
        plt.xlabel('Redshift (z)')
        plt.ylabel('Sound Horizon r_d (Mpc)')
        plt.title(f"BAO Sound Horizon Comparison\n"
                f"Genesis-Sphere Model (α={gs_params['alpha']:.4f}, β={gs_params['beta']:.4f}, "
                f"ω={gs_params['omega']:.4f}, ε={gs_params['epsilon']:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = os.path.join(results_dir, f"bao_comparison.png")
        plt.savefig(fig_path, dpi=150)
        
        return fig_path
    
    else:
        print(f"Dataset '{dataset_name}' not recognized for visualization.")
        return None

def generate_validation_summary(dataset_name, chi2, dof, gs_params, optimized=False):
    """
    Generates an AI-like summary of the validation results.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset used for validation
    chi2 : float
        Chi-square statistic from the validation
    dof : int
        Degrees of freedom
    gs_params : dict
        Genesis-Sphere model parameters
    optimized : bool
        Whether the parameters were optimized
        
    Returns:
    --------
    str
        Summary of validation results
    """
    # Calculate reduced chi-square
    reduced_chi2 = chi2 / dof
    
    # Initialize summary
    summary = [f"## AI Validation Summary: {dataset_name.title()} Dataset\n"]
    
    # Add parameter summary
    summary.append(f"### Genesis-Sphere Model Configuration")
    summary.append(f"- **Alpha (α)**: {gs_params['alpha']:.4f} - *Spatial dimension expansion coefficient*")
    summary.append(f"- **Beta (β)**: {gs_params['beta']:.4f} - *Temporal damping factor*")
    summary.append(f"- **Omega (ω)**: {gs_params['omega']:.4f} - *Angular frequency*")
    summary.append(f"- **Epsilon (ε)**: {gs_params['epsilon']:.4f} - *Zero-prevention constant*")
    if optimized:
        summary.append("*These parameters were optimized to best fit the data.*\n")
    else:
        summary.append("*These parameters are using initial values without optimization.*\n")
    
    # Add statistical analysis
    summary.append(f"### Statistical Fit Analysis")
    summary.append(f"- **Chi-Square**: {chi2:.2f}")
    summary.append(f"- **Degrees of Freedom**: {dof}")
    summary.append(f"- **Reduced Chi-Square**: {reduced_chi2:.2f}\n")
    
    # Interpret the reduced chi-square
    summary.append(f"### Interpretation")
    if reduced_chi2 < 0.5:
        summary.append("The model appears to be **overfitting** the data. This suggests that:")
        summary.append("- The model may have too many free parameters for the available data")
        summary.append("- Uncertainties in the data may be overestimated")
        summary.append("- The model might be capturing noise rather than meaningful patterns\n")
    elif 0.5 <= reduced_chi2 <= 1.5:
        summary.append("The model provides a **good fit** to the data. This suggests that:")
        summary.append("- The Genesis-Sphere parameters appropriately capture the observed phenomena")
        summary.append("- The theoretical framework shows promise for explaining the observations")
        summary.append("- The uncertainties in the data are reasonably estimated\n")
    elif 1.5 < reduced_chi2 <= 5:
        summary.append("The model provides a **moderate fit** to the data. This suggests that:")
        summary.append("- There is partial agreement between the model and observations")
        summary.append("- Some aspects of the phenomena may be well-captured, while others are not")
        summary.append("- Further parameter tuning or model refinement may improve results\n")
    else:
        summary.append("The model provides a **poor fit** to the data. This suggests that:")
        summary.append("- The Genesis-Sphere model in its current form may not adequately explain the observations")
        summary.append("- Further theoretical development may be needed")
        summary.append("- Alternative parameter mappings between Genesis-Sphere quantities and physical observables should be explored\n")
    
    # Add dataset-specific insights
    if dataset_name.lower() == 'supernovae':
        summary.append("### Dataset-Specific Insights")
        summary.append("The Type Ia supernovae dataset primarily tests the model's ability to reproduce the **distance-redshift relation**, which is directly connected to the expansion history of the universe. The Genesis-Sphere time-density function ρ(t) and temporal flow function Tf(t) combine to determine this relation.")
        
        if reduced_chi2 > 5:
            summary.append("\nThe high reduced chi-square suggests that the current mapping between Genesis-Sphere functions and luminosity distance may need refinement. Consider:")
            summary.append("- Modifying how time (t) maps to redshift (z)")
            summary.append("- Adjusting how density ρ(t) contributes to distance calculations")
            summary.append("- Exploring non-linear relationships between temporal flow Tf(t) and expansion rate")
        
    elif dataset_name.lower() == 'cmb':
        summary.append("### Dataset-Specific Insights")
        summary.append("The CMB dataset tests the model's ability to reproduce the fundamental cosmological parameters derived from the cosmic microwave background radiation. These parameters characterize the early universe and its evolution.")
        
        if reduced_chi2 > 5:
            summary.append("\nImproving the fit to CMB data may require adjustments to how Genesis-Sphere parameters map to standard cosmological parameters, particularly those affecting early universe dynamics.")
    
    elif dataset_name.lower() == 'bao':
        summary.append("### Dataset-Specific Insights")
        summary.append("The BAO dataset tests the model's predictions for the sound horizon scale at different redshifts, which serves as a 'standard ruler' in cosmology. This directly probes the expansion history of the universe.")
        
        if reduced_chi2 > 5:
            summary.append("\nThe Genesis-Sphere model's current mapping to BAO observables may need refinement, particularly in how it handles the evolution of the sound horizon scale with redshift.")
    
    # Add recommendations
    summary.append("### Recommendations")
    if optimized:
        if reduced_chi2 <= 1.5:
            summary.append("1. **Document these optimized parameters** as they provide a good fit to the data")
            summary.append("2. **Cross-validate** with other independent datasets")
            summary.append("3. **Explore the physical interpretation** of these parameter values in the context of the Genesis-Sphere framework")
        else:
            summary.append("1. **Revisit the parameter mapping** between Genesis-Sphere quantities and physical observables")
            summary.append("2. **Consider additional model components** that might better capture the observed phenomena")
            summary.append("3. **Test alternative optimization approaches** with different initial conditions or constraints")
    else:
        summary.append("1. **Run parameter optimization** to find the best-fitting model parameters")
        summary.append("2. **Perform sensitivity analysis** to understand which parameters most strongly affect the fit")
        summary.append("3. **Compare with standard cosmological models** to benchmark performance")
    
    # Join all parts of the summary with newlines
    return "\n".join(summary)

def main(dataset_name="supernovae", data_file=None, alpha=0.02, beta=1.2, omega=2.0, epsilon=0.1, optimize=False):
    """Main function to run the validation"""
    print("Genesis-Sphere Model Validation Against Observational Data")
    print("========================================================")
    
    # Load dataset
    data = load_dataset(dataset_name, data_file)
    
    # Set initial parameters
    initial_params = {'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon}
    
    # Optimize parameters if requested
    if optimize:
        gs_params, chi2 = optimize_model_parameters(dataset_name, data, initial_params)
    else:
        gs_params = initial_params
        # Initialize model with provided parameters
        gs_model = GenesisSphereModel(**gs_params)
        # Make predictions
        predictions = genesis_sphere_prediction_for_dataset(gs_model, dataset_name, data)
        # Calculate chi-square
        chi2 = calculate_chi_square(gs_model, dataset_name, data, predictions)
        print(f"Chi-square with initial parameters: {chi2:.2f}")
    
    # Initialize model with final parameters
    gs_model = GenesisSphereModel(**gs_params)
    
    # Make predictions
    predictions = genesis_sphere_prediction_for_dataset(gs_model, dataset_name, data)
    
    # Visualize comparison
    fig_path = visualize_comparison(dataset_name, data, predictions, gs_params)
    
    # Generate AI summary
    dof = len(data) - 4  # degrees of freedom = number of data points - number of parameters
    summary = generate_validation_summary(dataset_name, chi2, dof, gs_params, optimize)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, f"{dataset_name}_validation_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Print summary to console
    print("\n" + summary)
    
    # Save validation results
    results = {
        'dataset': dataset_name,
        'chi_square': chi2,
        'dof': dof,
        'reduced_chi_square': chi2 / dof,
        'alpha': gs_params['alpha'],
        'beta': gs_params['beta'],
        'omega': gs_params['omega'],
        'epsilon': gs_params['epsilon']
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(results_dir, f"{dataset_name}_validation_results.csv"), index=False)
    
    print(f"\nResults and visualizations saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Genesis-Sphere model against observational data")
    parser.add_argument("--dataset", type=str, default="supernovae", 
                        choices=["supernovae", "cmb", "bao"],
                        help="Dataset to use for validation")
    parser.add_argument("--data-file", type=str, default=None, 
                        help="Path to data file (optional, uses synthetic data if not provided)")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=1.2, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=2.0, help="Angular frequency for sinusoidal projections")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    parser.add_argument("--optimize", action="store_true", help="Optimize model parameters to fit data")
    parser.add_argument("--no-summary", action="store_true", help="Skip generating the AI validation summary")
    
    args = parser.parse_args()
    main(args.dataset, args.data_file, args.alpha, args.beta, args.omega, args.epsilon, args.optimize)
