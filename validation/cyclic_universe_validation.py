"""
Validates the Genesis-Sphere model against bouncing/cyclic universe models.

This module compares Genesis-Sphere predictions with cyclic universe models,
including Steinhardt & Turok's ekpyrotic scenario and Tolman's oscillating universe.
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
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'cyclic')
os.makedirs(results_dir, exist_ok=True)

# Create datasets directory if it doesn't exist
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

def generate_cyclic_model_data(model_type="ekpyrotic", n_cycles=3):
    """
    Generate sample data for cyclic universe models.
    
    Parameters:
    -----------
    model_type : str
        Type of cyclic model ('ekpyrotic', 'tolman', 'loop_quantum')
    n_cycles : int
        Number of cycles to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the generated data
    """
    # Base time array (one cycle)
    points_per_cycle = 200
    t_one_cycle = np.linspace(-np.pi, np.pi, points_per_cycle)
    
    # Extend to multiple cycles
    t = np.array([])
    for i in range(n_cycles):
        cycle_t = t_one_cycle + 2*np.pi*i
        t = np.append(t, cycle_t)
    
    # Different models have different density profiles
    if model_type == "ekpyrotic":
        # Ekpyrotic/cyclic universe (Steinhardt & Turok)
        # Characterized by slow contraction, rapid expansion
        density = np.array([])
        for i in range(n_cycles):
            base_density = 1 + 0.2*i  # Each cycle has slightly higher peak density (entropy increase)
            cycle_density = base_density * (1 + np.exp(-5*np.sin(t_one_cycle)))
            # Add the branes collision spike at each boundary
            brane_collision = 20 * np.exp(-100 * (t_one_cycle + np.pi)**2) + 20 * np.exp(-100 * (t_one_cycle - np.pi)**2)
            cycle_density += brane_collision
            density = np.append(density, cycle_density)
    
    elif model_type == "tolman":
        # Tolman's oscillating universe
        # Simple oscillation with entropy increase
        density = np.array([])
        for i in range(n_cycles):
            # Each cycle has lower peak density and longer period (entropy effect)
            amplitude = 2.0 / (1 + 0.3*i)
            cycle_density = amplitude * (1 + np.cos(t_one_cycle)) + 0.2
            density = np.append(density, cycle_density)
    
    else:  # loop_quantum
        # Loop Quantum Cosmology bounce
        # Characterized by quantum effects preventing singularity
        density = np.array([])
        for i in range(n_cycles):
            # Density is bounded by quantum effects
            cycle_density = 1.5 * (1 + np.tanh(3 * np.cos(t_one_cycle)))
            # Add quantum bounce effect
            bounce_effect = 0.5 / (0.1 + np.abs(np.sin(t_one_cycle/2)))
            mask = np.abs(np.sin(t_one_cycle/2)) < 0.2
            cycle_density[mask] = bounce_effect[mask]
            density = np.append(density, cycle_density)
    
    # Calculate scale factor (inverse relationship with density for simple model)
    scale_factor = 1.0 / np.sqrt(density)
    scale_factor = scale_factor / np.max(scale_factor)  # Normalize
    
    # Calculate Hubble parameter (H = ȧ/a)
    hubble = np.zeros_like(t)
    dt = t[1] - t[0]
    # Simple finite difference for derivative
    hubble[1:-1] = (scale_factor[2:] - scale_factor[:-2]) / (2*dt) / scale_factor[1:-1]
    # Handle endpoints
    hubble[0] = (scale_factor[1] - scale_factor[0]) / dt / scale_factor[0]
    hubble[-1] = (scale_factor[-1] - scale_factor[-2]) / dt / scale_factor[-1]
    
    # Create DataFrame
    return pd.DataFrame({
        'time': t,
        'density': density,
        'scale_factor': scale_factor,
        'hubble': hubble
    })

def load_or_create_cyclic_dataset(model_type="ekpyrotic"):
    """
    Load an existing cyclic model dataset or create a new one
    
    Parameters:
    -----------
    model_type : str
        Type of cyclic model to load
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the cyclic model data
    """
    file_path = os.path.join(datasets_dir, f"cyclic_{model_type}.csv")
    
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error loading cyclic dataset: {e}")
            print("Generating new data...")
    
    # Generate new data
    print(f"Generating new {model_type} cyclic universe data...")
    df = generate_cyclic_model_data(model_type)
    
    # Save to file
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"Saved {model_type} model data to {file_path}")
    
    return df

def compare_with_cyclic_model(gs_model, cyclic_df, model_type):
    """
    Compare Genesis-Sphere model with a cyclic universe model
    
    Parameters:
    -----------
    gs_model : GenesisSphereModel
        The Genesis-Sphere model instance
    cyclic_df : pd.DataFrame
        DataFrame with cyclic model data
    model_type : str
        Type of cyclic model
        
    Returns:
    --------
    dict
        Dictionary with comparison results
    """
    # Extract cyclic model data
    cyclic_time = cyclic_df['time'].values
    cyclic_density = cyclic_df['density'].values
    
    # Map cyclic time to Genesis-Sphere time
    # Genesis-Sphere is symmetric around t=0, so we may need to center or rescale
    gs_time = cyclic_time * 0.8  # Simple scaling
    
    # Get Genesis-Sphere predictions
    gs_results = gs_model.evaluate_all(gs_time)
    gs_density = gs_results['density']
    
    # Normalize both for comparison
    cyclic_density_norm = cyclic_density / np.max(cyclic_density)
    gs_density_norm = gs_density / np.max(gs_density)
    
    # Compute metrics
    residuals = cyclic_density_norm - gs_density_norm
    correlation = np.corrcoef(cyclic_density_norm, gs_density_norm)[0, 1]
    
    # Find peaks in both series to compare cycle characteristics
    from scipy.signal import find_peaks
    cyclic_peaks, _ = find_peaks(cyclic_density_norm, height=0.5)
    gs_peaks, _ = find_peaks(gs_density_norm, height=0.5)
    
    # Calculate cycle period comparison if we found multiple cycles
    if len(cyclic_peaks) > 1 and len(gs_peaks) > 1:
        cyclic_period = np.mean(np.diff(cyclic_time[cyclic_peaks]))
        gs_period = np.mean(np.diff(gs_time[gs_peaks]))
        period_ratio = gs_period / cyclic_period
    else:
        cyclic_period = 0
        gs_period = 0
        period_ratio = 0
    
    # Compute phase alignment - how well the cycles align
    if len(cyclic_peaks) > 0 and len(gs_peaks) > 0:
        # Compare first peak positions
        phase_diff = np.abs(cyclic_time[cyclic_peaks[0]] - gs_time[gs_peaks[0]])
    else:
        phase_diff = 0
    
    # Return all results and metrics
    return {
        'cyclic_time': cyclic_time,
        'gs_time': gs_time,
        'cyclic_density': cyclic_density,
        'cyclic_density_norm': cyclic_density_norm,
        'gs_density': gs_density,
        'gs_density_norm': gs_density_norm,
        'metrics': {
            'correlation': correlation,
            'mean_abs_error': np.mean(np.abs(residuals)),
            'mean_squared_error': np.mean(residuals**2),
            'cyclic_period': cyclic_period,
            'gs_period': gs_period,
            'period_ratio': period_ratio,
            'phase_diff': phase_diff,
            'cyclic_peaks': cyclic_peaks,
            'gs_peaks': gs_peaks
        }
    }

def visualize_cyclic_comparison(comparison_results, model_type, gs_params):
    """
    Visualize the comparison between Genesis-Sphere and cyclic model
    
    Parameters:
    -----------
    comparison_results : dict
        Results from the comparison function
    model_type : str
        Type of cyclic model
    gs_params : dict
        Genesis-Sphere parameters
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    # Extract data
    cyclic_time = comparison_results['cyclic_time']
    gs_time = comparison_results['gs_time']
    cyclic_density_norm = comparison_results['cyclic_density_norm']
    gs_density_norm = comparison_results['gs_density_norm']
    metrics = comparison_results['metrics']
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot normalized densities
    axes[0].plot(cyclic_time, cyclic_density_norm, 'b-', 
                label=f'{model_type.capitalize()} Cyclic Model')
    axes[0].plot(gs_time, gs_density_norm, 'r-', 
                label='Genesis-Sphere Model')
    
    # Mark peaks if available
    if len(metrics['cyclic_peaks']) > 0:
        axes[0].plot(cyclic_time[metrics['cyclic_peaks']], 
                    cyclic_density_norm[metrics['cyclic_peaks']], 
                    'bo', markersize=8)
    if len(metrics['gs_peaks']) > 0:
        axes[0].plot(gs_time[metrics['gs_peaks']], 
                    gs_density_norm[metrics['gs_peaks']], 
                    'ro', markersize=8)
    
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Normalized Density')
    axes[0].set_title(f'Comparison: {model_type.capitalize()} Cyclic Universe vs Genesis-Sphere\n'
                     f'(α={gs_params["alpha"]:.4f}, β={gs_params["beta"]:.4f}, '
                     f'ω={gs_params["omega"]:.4f}, ε={gs_params["epsilon"]:.4f})')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot residuals
    residuals = cyclic_density_norm - gs_density_norm
    axes[1].plot(gs_time, residuals, 'g-')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals (Cyclic Model - Genesis-Sphere)')
    axes[1].grid(True)
    
    # Add metrics as text
    metrics_text = f"Correlation: {metrics['correlation']:.3f}\n" \
                   f"Mean Abs Error: {metrics['mean_abs_error']:.3f}\n" \
                   f"Cyclic Period: {metrics['cyclic_period']:.3f}\n" \
                   f"Genesis-Sphere Period: {metrics['gs_period']:.3f}\n" \
                   f"Period Ratio: {metrics['period_ratio']:.3f}\n" \
                   f"Phase Difference: {metrics['phase_diff']:.3f}"
    
    plt.figtext(0.15, 0.02, metrics_text, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    
    # Save figure
    file_name = f"cyclic_{model_type}_comparison.png"
    fig_path = os.path.join(results_dir, file_name)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    return fig_path

def optimize_gs_for_cyclic(cyclic_df, initial_params, model_type):
    """
    Optimize Genesis-Sphere parameters to match a cyclic universe model
    
    Parameters:
    -----------
    cyclic_df : pd.DataFrame
        DataFrame with cyclic model data
    initial_params : dict
        Initial Genesis-Sphere parameters
    model_type : str
        Type of cyclic model
        
    Returns:
    --------
    dict
        Optimized parameters
    float
        Final error
    """
    # Extract cyclic model data
    cyclic_time = cyclic_df['time'].values
    cyclic_density = cyclic_df['density'].values
    
    # Normalize density
    cyclic_density_norm = cyclic_density / np.max(cyclic_density)
    
    # Map time - Genesis-Sphere time scale might need adjustment
    gs_time = cyclic_time * 0.8
    
    # Define objective function for optimization
    def objective(params):
        alpha, beta, omega, epsilon = params
        model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        results = model.evaluate_all(gs_time)
        gs_density = results['density']
        gs_density_norm = gs_density / np.max(gs_density)
        
        # Get correlation - we want to maximize correlation, so return negative
        correlation = np.corrcoef(cyclic_density_norm, gs_density_norm)[0, 1]
        
        # Calculate error as weighted combination of correlation and MSE
        residuals = cyclic_density_norm - gs_density_norm
        mse = np.mean(residuals**2)
        
        # Return a combined metric (higher correlation and lower MSE is better)
        return mse - 0.5 * correlation  # Negative correlation term because we minimize
    
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
        (0.2, 5.0),     # omega - wider range for cyclical models
        (0.01, 0.5)     # epsilon
    ]
    
    # Run optimization
    print(f"Optimizing Genesis-Sphere parameters to match {model_type} cyclic model...")
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
    
    print(f"Optimization completed with final error = {final_error:.6f}")
    print(f"Optimized parameters: α={alpha_opt:.4f}, β={beta_opt:.4f}, ω={omega_opt:.4f}, ε={epsilon_opt:.4f}")
    
    return optimized_params, final_error

def generate_cyclic_validation_summary(model_type, comparison_results, gs_params, optimized=False):
    """
    Generate an AI-like summary of the validation against a cyclic model.
    
    Parameters:
    -----------
    model_type : str
        Type of cyclic universe model
    comparison_results : dict
        Results from the comparison
    gs_params : dict
        Genesis-Sphere parameters
    optimized : bool
        Whether the parameters were optimized
        
    Returns:
    --------
    str
        Summary text
    """
    # Extract metrics
    metrics = comparison_results['metrics']
    correlation = metrics['correlation']
    
    # Initialize summary
    summary = [f"## AI Validation Summary: {model_type.capitalize()} Cyclic Universe Comparison\n"]
    
    # Add parameter summary
    summary.append(f"### Genesis-Sphere Model Configuration")
    summary.append(f"- **Alpha (α)**: {gs_params['alpha']:.4f} - *Spatial dimension expansion coefficient*")
    summary.append(f"- **Beta (β)**: {gs_params['beta']:.4f} - *Temporal damping factor*")
    summary.append(f"- **Omega (ω)**: {gs_params['omega']:.4f} - *Angular frequency*")
    summary.append(f"- **Epsilon (ε)**: {gs_params['epsilon']:.4f} - *Zero-prevention constant*")
    
    if optimized:
        summary.append("*These parameters were optimized to best fit the cyclic universe model.*\n")
    else:
        summary.append("*These parameters are using initial values without optimization.*\n")
    
    # Add statistical analysis
    summary.append(f"### Statistical Comparison")
    summary.append(f"- **Correlation Coefficient**: {correlation:.4f}")
    summary.append(f"- **Mean Absolute Error**: {metrics['mean_abs_error']:.4f}")
    summary.append(f"- **Mean Squared Error**: {metrics['mean_squared_error']:.6f}")
    
    # Add cycle characteristics if available
    if metrics['cyclic_period'] > 0:
        summary.append(f"- **{model_type.capitalize()} Cycle Period**: {metrics['cyclic_period']:.3f}")
        summary.append(f"- **Genesis-Sphere Cycle Period**: {metrics['gs_period']:.3f}")
        summary.append(f"- **Period Ratio (GS/Cyclic)**: {metrics['period_ratio']:.3f}")
        summary.append(f"- **Phase Difference**: {metrics['phase_diff']:.3f}\n")
    else:
        summary.append("\n")
    
    # Add interpretation based on correlation
    summary.append(f"### Interpretation")
    if correlation > 0.9:
        summary.append(f"The Genesis-Sphere model shows **excellent correlation** with the {model_type} cyclic universe model. This is particularly significant because:")
    elif correlation > 0.7:
        summary.append(f"The Genesis-Sphere model shows **good correlation** with the {model_type} cyclic universe model. This is notable because:")
    elif correlation > 0.5:
        summary.append(f"The Genesis-Sphere model shows **moderate correlation** with the {model_type} cyclic universe model. This suggests:")
    else:
        summary.append(f"The Genesis-Sphere model shows **weak correlation** with the {model_type} cyclic universe model. This indicates:")
    
    # Model-specific interpretations
    if model_type == "ekpyrotic":
        if correlation > 0.7:
            summary.append("- Genesis-Sphere's oscillatory function naturally captures brane collision dynamics")
            summary.append("- The time-density function ρ(t) mimics the ekpyrotic contraction-expansion cycle")
            summary.append("- The model successfully represents the asymmetry between contraction and expansion phases\n")
        else:
            summary.append("- The sharp density spikes at brane collisions are challenging to model with the current functions")
            summary.append("- The ekpyrotic model's specific contraction dynamics are not fully captured")
            summary.append("- Modifications may be needed to better represent cycling through brane collisions\n")
    
    elif model_type == "tolman":
        if correlation > 0.7:
            summary.append("- Genesis-Sphere effectively models Tolman's oscillating universe density cycles")
            summary.append("- The sinusoidal component in ρ(t) aligns with cosmic oscillation patterns")
            summary.append("- The parameter ω closely corresponds to the oscillation frequency in Tolman's model\n")
        else:
            summary.append("- The entropy-driven changes in cycle amplitude are not fully captured")
            summary.append("- The model may need additional parameters to represent growing cycle periods")
            summary.append("- Tolman's thermodynamic effects are not directly represented in the current formulation\n")
    
    else:  # loop_quantum
        if correlation > 0.7:
            summary.append("- Genesis-Sphere captures the quantum bounce dynamics near t=0")
            summary.append("- The temporal flow function Tf(t) simulates the quantum gravity effects at high density")
            summary.append("- The avoidance of singularity is well-represented by the combined ρ(t) and Tf(t) behaviors\n")
        else:
            summary.append("- Quantum effects at the bounce point need more specialized representation")
            summary.append("- The precise density limitation mechanisms differ from the Genesis-Sphere approach")
            summary.append("- Loop quantum cosmology employs quantum corrections not directly present in the model\n")
    
    # Add general observations about cyclic models
    summary.append("### Genesis-Sphere and Cyclic Cosmology")
    summary.append("The inherent time-symmetry in the Genesis-Sphere model makes it particularly suitable for modeling cyclic universes. Key observations:")
    summary.append("- The parameter ω directly controls oscillation frequency, mapping well to cosmic cycles")
    summary.append("- Genesis-Sphere naturally produces recurring density patterns without requiring custom functions")
    summary.append("- The model provides a simplified but effective representation of cycle dynamics\n")
    
    # Add recommendations
    summary.append("### Recommendations")
    if optimized:
        if correlation > 0.7:
            summary.append("1. **Document these optimized parameters** as an effective representation of the cyclic model")
            summary.append("2. **Investigate the physical meaning** of these parameter values in cyclic cosmology")
            summary.append("3. **Compare with other cyclic variants** to establish a broader theoretical framework")
        else:
            summary.append("1. **Extend the Genesis-Sphere model** with additional terms to better capture cyclic features")
            summary.append("2. **Explore alternative time mappings** between the models")
            summary.append("3. **Consider cycle-specific modifications** for this particular type of cyclic model")
    else:
        summary.append("1. **Run parameter optimization** to better align with this cyclic model")
        summary.append("2. **Adjust the time scaling factor** to match cycle periods more closely")
        summary.append("3. **Explore higher omega values** to better capture oscillatory behavior")
    
    # Join all parts of the summary with newlines
    return "\n".join(summary)

def main(model_type="ekpyrotic", alpha=0.02, beta=1.2, omega=2.0, epsilon=0.1, optimize=False):
    """Main function to run the validation"""
    print("Genesis-Sphere Cyclic Universe Model Validation")
    print("============================================")
    
    # Load or create cyclic model data
    cyclic_df = load_or_create_cyclic_dataset(model_type)
    
    # Set initial parameters
    initial_params = {'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon}
    
    # Optimize parameters if requested
    if optimize:
        gs_params, error = optimize_gs_for_cyclic(cyclic_df, initial_params, model_type)
    else:
        gs_params = initial_params
    
    # Initialize model with parameters
    gs_model = GenesisSphereModel(**gs_params)
    
    # Compare models
    comparison_results = compare_with_cyclic_model(gs_model, cyclic_df, model_type)
    
    # Visualize comparison
    fig_path = visualize_cyclic_comparison(comparison_results, model_type, gs_params)
    print(f"Comparison visualization saved to {fig_path}")
    
    # Generate AI summary
    summary = generate_cyclic_validation_summary(model_type, comparison_results, gs_params, optimize)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, f"cyclic_{model_type}_validation_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Print summary to console
    print("\n" + summary)
    
    # Save validation results
    results = {
        'model_type': model_type,
        'correlation': comparison_results['metrics']['correlation'],
        'mean_abs_error': comparison_results['metrics']['mean_abs_error'],
        'mean_squared_error': comparison_results['metrics']['mean_squared_error'],
        'alpha': gs_params['alpha'],
        'beta': gs_params['beta'],
        'omega': gs_params['omega'],
        'epsilon': gs_params['epsilon'],
        'optimized': optimize
    }
    
    if comparison_results['metrics']['cyclic_period'] > 0:
        results['cyclic_period'] = comparison_results['metrics']['cyclic_period']
        results['gs_period'] = comparison_results['metrics']['gs_period']
        results['period_ratio'] = comparison_results['metrics']['period_ratio']
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(results_dir, f"cyclic_{model_type}_validation_results.csv"), index=False)
    
    print(f"\nResults and visualizations saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Genesis-Sphere model against cyclic universe models")
    parser.add_argument("--model", type=str, default="ekpyrotic", 
                        choices=["ekpyrotic", "tolman", "loop_quantum"],
                        help="Cyclic universe model to use for validation")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=1.2, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=2.0, help="Angular frequency for sinusoidal projections")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    parser.add_argument("--optimize", action="store_true", help="Optimize model parameters to fit cyclic model")
    
    args = parser.parse_args()
    main(args.model, args.alpha, args.beta, args.omega, args.epsilon, args.optimize)
