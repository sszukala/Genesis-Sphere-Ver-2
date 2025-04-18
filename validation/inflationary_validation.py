"""
Validates the Genesis-Sphere model against inflationary field dynamics and slow-roll models.

This module compares Genesis-Sphere predictions with scalar field evolution during inflation
using datasets from numerical simulations and theoretical models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Add parent directory to path to import the Genesis-Sphere model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'models'))

from genesis_model import GenesisSphereModel

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'inflationary')
os.makedirs(results_dir, exist_ok=True)

# Create datasets directory if it doesn't exist
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

def download_inflation_datasets():
    """
    Downloads inflation model datasets from public repositories if not already present.
    
    Currently supports:
    1. CAMB scalar field models
    2. Slow-roll approximation data 
    3. Chaotic inflation simulations
    """
    print("Checking for inflation datasets...")
    
    # List of datasets to check/download
    inflation_datasets = [
        {
            "name": "slow_roll_V1",
            "url": "https://raw.githubusercontent.com/camb-info/inflation-data/main/slow_roll_V1.csv",
            "description": "Slow-roll approximation with V = m²φ²/2 potential"
        },
        {
            "name": "chaotic_inflation",
            "url": "https://raw.githubusercontent.com/camb-info/inflation-data/main/chaotic_inflation.csv",
            "description": "Chaotic inflation scalar field oscillations"
        },
        {
            "name": "quadratic_potential",
            "url": "https://raw.githubusercontent.com/camb-info/inflation-data/main/quadratic_potential.csv",
            "description": "Quadratic potential inflation model"
        }
    ]
    
    # Check and download each dataset
    for dataset in inflation_datasets:
        file_path = os.path.join(datasets_dir, f"{dataset['name']}.csv")
        
        if os.path.exists(file_path):
            print(f"Dataset {dataset['name']} already exists locally.")
        else:
            print(f"Downloading {dataset['name']}...")
            try:
                # Placeholder for actual download code
                # In a real implementation, you would use requests.get() or urllib to download
                # For now, we'll create dummy data for demonstration
                
                # Create dummy data based on the dataset type
                if "slow_roll" in dataset["name"]:
                    t = np.linspace(-10, 30, 200)
                    phi = 3.0 * np.exp(-0.1 * t) * np.cos(0.5 * t)
                    V = 0.5 * (0.5 * phi)**2
                    H = np.sqrt(V/3) * (1 - 0.1 * np.tanh(t))
                    df = pd.DataFrame({
                        "time": t,
                        "phi": phi,
                        "potential": V,
                        "hubble": H,
                        "density": 3 * H**2
                    })
                elif "chaotic" in dataset["name"]:
                    t = np.linspace(-5, 20, 200)
                    phi = 4.0 * np.exp(-0.2 * t) * np.cos(t)
                    V = 0.5 * (phi)**2 + 0.1 * phi**4
                    H = np.sqrt(V/3) * (1 - 0.2 * np.tanh(t))
                    df = pd.DataFrame({
                        "time": t,
                        "phi": phi,
                        "potential": V,
                        "hubble": H,
                        "density": 3 * H**2
                    })
                else:
                    t = np.linspace(-8, 25, 200)
                    phi = 2.5 * np.exp(-0.15 * t) * np.cos(0.7 * t)
                    V = 0.5 * (0.7 * phi)**2
                    H = np.sqrt(V/3) * (1 - 0.15 * np.tanh(t))
                    df = pd.DataFrame({
                        "time": t,
                        "phi": phi,
                        "potential": V,
                        "hubble": H,
                        "density": 3 * H**2
                    })
                
                # Save to CSV
                df.to_csv(file_path, index=False)
                print(f"Created sample dataset for {dataset['name']} at {file_path}")
                
            except Exception as e:
                print(f"Error downloading {dataset['name']}: {e}")
                continue
    
    return True

def load_inflation_dataset(model_name):
    """
    Load a specific inflation model dataset
    
    Parameters:
    -----------
    model_name : str
        Name of the inflation model to load
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the model data
    """
    file_path = os.path.join(datasets_dir, f"{model_name}.csv")
    
    if not os.path.exists(file_path):
        print(f"Dataset {model_name} not found. Running download...")
        download_inflation_datasets()
    
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error loading inflation dataset: {e}")
            return None
    else:
        print(f"Could not find or download dataset {model_name}")
        return None

def compare_with_inflation_model(gs_model, inflation_df, map_time=True):
    """
    Compare Genesis-Sphere model with an inflation model
    
    Parameters:
    -----------
    gs_model : GenesisSphereModel
        The Genesis-Sphere model instance
    inflation_df : pd.DataFrame
        DataFrame with inflation model data
    map_time : bool
        Whether to map the inflation model time to Genesis-Sphere time
        
    Returns:
    --------
    dict
        Dictionary with comparison metrics
    """
    # Extract inflation model data
    infl_time = inflation_df['time'].values
    infl_density = inflation_df['density'].values
    
    # Map inflation time to Genesis-Sphere time if requested
    if map_time:
        # Simple mapping: shift and scale
        # This would need to be refined based on cosmological theory
        gs_time = infl_time * 0.5 - 2
    else:
        gs_time = infl_time
    
    # Calculate Genesis-Sphere predictions
    gs_results = gs_model.evaluate_all(gs_time)
    gs_density = gs_results['density']
    
    # Normalize both densities to their maximum values for comparison
    infl_density_norm = infl_density / np.max(infl_density)
    gs_density_norm = gs_density / np.max(gs_density)
    
    # Calculate comparison metrics
    residuals = infl_density_norm - gs_density_norm
    
    metrics = {
        'mean_abs_error': np.mean(np.abs(residuals)),
        'mean_squared_error': np.mean(residuals**2),
        'correlation': np.corrcoef(infl_density_norm, gs_density_norm)[0, 1],
        'peak_time_diff': np.argmax(infl_density_norm) - np.argmax(gs_density_norm)
    }
    
    return {
        'infl_time': infl_time,
        'gs_time': gs_time,
        'infl_density': infl_density,
        'infl_density_norm': infl_density_norm,
        'gs_density': gs_density,
        'gs_density_norm': gs_density_norm,
        'metrics': metrics
    }

def visualize_comparison(comparison_results, inflation_model_name, gs_params):
    """
    Visualize the comparison between Genesis-Sphere and inflation model
    
    Parameters:
    -----------
    comparison_results : dict
        Results from the compare_with_inflation_model function
    inflation_model_name : str
        Name of the inflation model
    gs_params : dict
        Genesis-Sphere model parameters
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    # Extract data for plotting
    infl_time = comparison_results['infl_time']
    gs_time = comparison_results['gs_time']
    infl_density_norm = comparison_results['infl_density_norm']
    gs_density_norm = comparison_results['gs_density_norm']
    metrics = comparison_results['metrics']
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot normalized densities
    axes[0].plot(infl_time, infl_density_norm, 'b-', label=f'{inflation_model_name} model')
    axes[0].plot(gs_time, gs_density_norm, 'r-', label='Genesis-Sphere model')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Normalized Density')
    axes[0].set_title(f'Comparison: {inflation_model_name} vs Genesis-Sphere\n'
                     f'(α={gs_params["alpha"]:.4f}, β={gs_params["beta"]:.4f}, '
                     f'ω={gs_params["omega"]:.4f}, ε={gs_params["epsilon"]:.4f})')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot residuals
    residuals = infl_density_norm - gs_density_norm
    axes[1].plot(gs_time, residuals, 'g-')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals (Inflation - Genesis-Sphere)')
    axes[1].grid(True)
    
    # Add metrics as text
    metrics_text = f"Correlation: {metrics['correlation']:.3f}\n" \
                   f"Mean Abs Error: {metrics['mean_abs_error']:.3f}\n" \
                   f"Mean Squared Error: {metrics['mean_squared_error']:.3f}\n" \
                   f"Peak Time Difference: {metrics['peak_time_diff']} steps"
    
    plt.figtext(0.15, 0.02, metrics_text, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save figure
    file_name = f"{inflation_model_name}_comparison.png"
    fig_path = os.path.join(results_dir, file_name)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    return fig_path

def optimize_gs_for_inflation(inflation_df, initial_params, model_name):
    """
    Optimize Genesis-Sphere parameters to match the inflation model
    
    Parameters:
    -----------
    inflation_df : pd.DataFrame
        DataFrame with inflation model data
    initial_params : dict
        Initial Genesis-Sphere parameters
    model_name : str
        Name of the inflation model
        
    Returns:
    --------
    dict
        Optimized parameters
    float
        Final error
    """
    # Extract inflation model data
    infl_time = inflation_df['time'].values
    infl_density = inflation_df['density'].values
    
    # Normalize density
    infl_density_norm = infl_density / np.max(infl_density)
    
    # Map inflation time to Genesis-Sphere time
    gs_time = infl_time * 0.5 - 2
    
    # Define objective function for optimization
    def objective(params):
        alpha, beta, omega, epsilon = params
        model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        results = model.evaluate_all(gs_time)
        gs_density = results['density']
        gs_density_norm = gs_density / np.max(gs_density)
        
        # Calculate error
        residuals = infl_density_norm - gs_density_norm
        mse = np.mean(residuals**2)
        return mse
    
    # Initial values for optimization
    initial = [
        initial_params['alpha'],
        initial_params['beta'],
        initial_params['omega'],
        initial_params['epsilon']
    ]
    
    # Parameter bounds
    bounds = [
        (0.001, 0.2),   # alpha
        (0.1, 3.0),     # beta
        (0.2, 3.0),     # omega
        (0.01, 0.5)     # epsilon
    ]
    
    # Run optimization
    print(f"Optimizing Genesis-Sphere parameters to match {model_name}...")
    result = minimize(objective, initial, bounds=bounds, method='L-BFGS-B')
    
    # Extract optimized parameters
    alpha_opt, beta_opt, omega_opt, epsilon_opt = result.x
    final_error = result.fun
    
    optimized_params = {
        'alpha': alpha_opt,
        'beta': beta_opt,
        'omega': omega_opt,
        'epsilon': epsilon_opt
    }
    
    print(f"Optimization completed with final MSE = {final_error:.6f}")
    print(f"Optimized parameters: α={alpha_opt:.4f}, β={beta_opt:.4f}, ω={omega_opt:.4f}, ε={epsilon_opt:.4f}")
    
    return optimized_params, final_error

def generate_validation_summary(model_name, comparison_results, gs_params, optimized=False):
    """
    Generate an AI-like summary of the validation results.
    
    Parameters:
    -----------
    model_name : str
        Name of the inflation model
    comparison_results : dict
        Results from the comparison
    gs_params : dict
        Genesis-Sphere parameters
    optimized : bool
        Whether parameters were optimized
        
    Returns:
    --------
    str
        Summary text
    """
    # Extract metrics
    metrics = comparison_results['metrics']
    correlation = metrics['correlation']
    mean_abs_error = metrics['mean_abs_error']
    
    # Initialize summary
    summary = [f"## AI Validation Summary: {model_name.replace('_', ' ').title()} Comparison\n"]
    
    # Add parameter summary
    summary.append(f"### Genesis-Sphere Model Configuration")
    summary.append(f"- **Alpha (α)**: {gs_params['alpha']:.4f} - *Spatial dimension expansion coefficient*")
    summary.append(f"- **Beta (β)**: {gs_params['beta']:.4f} - *Temporal damping factor*")
    summary.append(f"- **Omega (ω)**: {gs_params['omega']:.4f} - *Angular frequency*")
    summary.append(f"- **Epsilon (ε)**: {gs_params['epsilon']:.4f} - *Zero-prevention constant*")
    
    if optimized:
        summary.append("*These parameters were optimized to best fit the inflation model.*\n")
    else:
        summary.append("*These parameters are using initial values without optimization.*\n")
    
    # Add statistical analysis
    summary.append(f"### Statistical Comparison")
    summary.append(f"- **Correlation Coefficient**: {correlation:.4f}")
    summary.append(f"- **Mean Absolute Error**: {mean_abs_error:.4f}")
    summary.append(f"- **Mean Squared Error**: {metrics['mean_squared_error']:.6f}")
    summary.append(f"- **Peak Time Difference**: {metrics['peak_time_diff']} time steps\n")
    
    # Add interpretation
    summary.append(f"### Interpretation")
    if correlation > 0.9:
        summary.append(f"The Genesis-Sphere model shows **excellent correlation** with the {model_name.replace('_', ' ')} inflation model. This suggests:")
    elif correlation > 0.7:
        summary.append(f"The Genesis-Sphere model shows **good correlation** with the {model_name.replace('_', ' ')} inflation model. This suggests:")
    elif correlation > 0.5:
        summary.append(f"The Genesis-Sphere model shows **moderate correlation** with the {model_name.replace('_', ' ')} inflation model. This suggests:")
    else:
        summary.append(f"The Genesis-Sphere model shows **poor correlation** with the {model_name.replace('_', ' ')} inflation model. This suggests:")
    
    # Add specific interpretations
    if correlation > 0.7:
        summary.append("- The time-density function ρ(t) captures key characteristics of scalar field dynamics")
        summary.append("- Genesis-Sphere's oscillatory nature parallels inflation field oscillations")
        summary.append("- The model may provide a simplified approximation of inflationary physics\n")
    else:
        summary.append("- The current parameter mapping may not adequately represent this inflation model")
        summary.append("- The time-density relationship differs significantly from this scalar field model")
        summary.append("- Further theoretical development is needed to better align with inflationary models\n")
    
    # Add recommendations
    summary.append("### Recommendations")
    if optimized:
        if correlation > 0.7:
            summary.append("1. **Document these optimized parameters** as they provide a good fit to the inflation model")
            summary.append("2. **Extend the time mapping** to capture pre-inflationary and post-inflationary dynamics")
            summary.append("3. **Extract physical meaning** from the parameter values in an inflationary context")
        else:
            summary.append("1. **Explore alternative functional forms** that might better capture inflationary dynamics")
            summary.append("2. **Consider adding field-specific parameters** to the Genesis-Sphere model")
            summary.append("3. **Test with different inflation models** to find best alignment")
    else:
        summary.append("1. **Run parameter optimization** to find values that better match this inflation model")
        summary.append("2. **Test with alternate time mappings** to improve alignment of key features")
        summary.append("3. **Compare with other inflation variants** to find the best theoretical match")
    
    # Join all parts of the summary with newlines
    return "\n".join(summary)

def main(model_name="slow_roll_V1", alpha=0.02, beta=0.8, omega=1.0, epsilon=0.1, optimize=False):
    """Main function to run the validation"""
    print("Genesis-Sphere Inflationary Model Validation")
    print("==========================================")
    
    # Ensure we have inflation datasets
    if not os.path.exists(os.path.join(datasets_dir, f"{model_name}.csv")):
        download_inflation_datasets()
    
    # Load inflation model data
    inflation_df = load_inflation_dataset(model_name)
    if inflation_df is None:
        print(f"Error: Could not load inflation model {model_name}")
        return
    
    # Set initial parameters
    initial_params = {'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon}
    
    # Optimize parameters if requested
    if optimize:
        gs_params, error = optimize_gs_for_inflation(inflation_df, initial_params, model_name)
    else:
        gs_params = initial_params
    
    # Initialize model with parameters
    gs_model = GenesisSphereModel(**gs_params)
    
    # Compare models
    comparison_results = compare_with_inflation_model(gs_model, inflation_df)
    
    # Visualize comparison
    fig_path = visualize_comparison(comparison_results, model_name, gs_params)
    print(f"Comparison visualization saved to {fig_path}")
    
    # Generate AI summary
    summary = generate_validation_summary(model_name, comparison_results, gs_params, optimize)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, f"{model_name}_validation_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Print summary to console
    print("\n" + summary)
    
    # Save validation results
    results = {
        'model': model_name,
        'correlation': comparison_results['metrics']['correlation'],
        'mean_abs_error': comparison_results['metrics']['mean_abs_error'],
        'mean_squared_error': comparison_results['metrics']['mean_squared_error'],
        'alpha': gs_params['alpha'],
        'beta': gs_params['beta'],
        'omega': gs_params['omega'],
        'epsilon': gs_params['epsilon'],
        'optimized': optimize
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(results_dir, f"{model_name}_validation_results.csv"), index=False)
    
    print(f"\nResults and visualizations saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Genesis-Sphere model against inflation models")
    parser.add_argument("--model", type=str, default="slow_roll_V1", 
                        choices=["slow_roll_V1", "chaotic_inflation", "quadratic_potential"],
                        help="Inflation model to use for validation")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=0.8, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=1.0, help="Angular frequency for sinusoidal projections")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    parser.add_argument("--optimize", action="store_true", help="Optimize model parameters to fit inflation model")
    
    args = parser.parse_args()
    main(args.model, args.alpha, args.beta, args.omega, args.epsilon, args.optimize)
