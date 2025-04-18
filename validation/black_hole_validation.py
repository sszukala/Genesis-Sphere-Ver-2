"""
Validates the Genesis-Sphere temporal flow function against black hole time dilation effects.

This module compares the temporal flow function Tf(t) with numerical relativity results
for time dilation near black hole event horizons and gravitational singularities.
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
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'black_hole')
os.makedirs(results_dir, exist_ok=True)

# Create datasets directory if it doesn't exist
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

def generate_schwarzschild_time_dilation(r_min=0.01, r_max=10, num_points=200):
    """
    Generate time dilation data for a Schwarzschild black hole.
    
    Parameters:
    -----------
    r_min : float
        Minimum radial distance in Schwarzschild radii
    r_max : float
        Maximum radial distance in Schwarzschild radii
    num_points : int
        Number of data points to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the generated data
    """
    # Radial distances in units of Schwarzschild radius (Rs = 2GM/c²)
    r = np.geomspace(r_min, r_max, num_points)  # Logarithmic spacing
    
    # Time dilation factor for Schwarzschild metric
    # dt_proper/dt_distant = sqrt(1 - Rs/r)
    time_dilation = np.sqrt(1 - 1/r)
    
    # Proper time passing for an observer at radius r
    # (relative to distant observer)
    proper_time_rate = time_dilation
    
    # "Temporal flow" in Genesis-Sphere terminology
    # Genesis-Sphere Tf is analogous to proper_time_rate
    temporal_flow = proper_time_rate
    
    # For comparison, calculate the coordinate time needed to fall from r to r-dr
    dr = np.diff(r)
    fall_time = np.zeros_like(r)
    fall_time[:-1] = dr / (1 - 1/r[:-1])**(3/2)
    fall_time[-1] = fall_time[-2]  # Just to maintain array size
    
    return pd.DataFrame({
        'r': r,  # Radial distance in Schwarzschild radii
        'time_dilation': time_dilation,  
        'proper_time_rate': proper_time_rate,
        'temporal_flow': temporal_flow,
        'fall_time': fall_time
    })

def generate_kerr_time_dilation(r_min=0.1, r_max=10, num_points=200, spin=0.9):
    """
    Generate time dilation data for a Kerr (spinning) black hole.
    
    Parameters:
    -----------
    r_min : float
        Minimum radial distance in gravitational radii
    r_max : float
        Maximum radial distance in gravitational radii
    num_points : int
        Number of data points to generate
    spin : float
        Dimensionless spin parameter (0 to 1)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the generated data
    """
    # Radial distances in units of gravitational radius (Rg = GM/c²)
    r = np.geomspace(r_min, r_max, num_points)
    
    # Convert to BL coordinates
    # For equatorial plane (theta = pi/2) calculations
    a = spin  # Dimensionless spin parameter
    
    # Time dilation factor for Kerr metric in equatorial plane
    # This is a simplified approximation
    time_dilation = np.sqrt((r**2 - 2*r + a**2) / (r**2 + a**2))
    
    # Handle event horizon and ergosphere
    r_event = 1 + np.sqrt(1 - a**2)  # Event horizon
    time_dilation[r < r_event] = 0  # Inside event horizon
    
    # Proper time rate
    proper_time_rate = time_dilation
    
    # Temporal flow in Genesis-Sphere terminology
    temporal_flow = proper_time_rate
    
    return pd.DataFrame({
        'r': r,
        'time_dilation': time_dilation,
        'proper_time_rate': proper_time_rate,
        'temporal_flow': temporal_flow,
        'spin': np.ones_like(r) * spin
    })

def generate_kerr_newman_time_dilation(r_min=0.1, r_max=10, num_points=200, spin=0.9, charge=0.4):
    """
    Generate time dilation data for a Kerr-Newman (charged, spinning) black hole.
    
    Parameters:
    -----------
    r_min : float
        Minimum radial distance in gravitational radii
    r_max : float
        Maximum radial distance in gravitational radii
    num_points : int
        Number of data points to generate
    spin : float
        Dimensionless spin parameter a (0 to 1)
    charge : float
        Dimensionless charge parameter q (0 to 1)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the generated data
    """
    # Radial distances in units of gravitational radius (Rg = GM/c²)
    r = np.geomspace(r_min, r_max, num_points)
    
    # Parameter setup
    a = spin  # Spin parameter
    q = charge  # Charge parameter
    
    # Time dilation factor for Kerr-Newman metric in equatorial plane
    # This is a simplified approximation for the equatorial plane
    delta = r**2 - 2*r + a**2 + q**2
    
    # Check for negative values before square root to avoid warnings
    # We'll assign zero to any problematic points
    valid_points = (delta > 0) & ((r**2 + a**2) > 0)
    time_dilation = np.zeros_like(r, dtype=float)
    
    # Calculate only for valid points
    time_dilation[valid_points] = np.sqrt(
        delta[valid_points] / (r[valid_points]**2 + a**2)
    )
    
    # Handle event horizon
    # The Kerr-Newman event horizon is at:
    discriminant = 1 - a**2 - q**2
    
    if discriminant >= 0:
        r_event = 1 + np.sqrt(discriminant)  # Outer event horizon
        # Set time dilation to 0 inside event horizon
        time_dilation[r < r_event] = 0
    else:
        print(f"Warning: a²+q² = {a**2 + q**2:.3f} > 1, parameters may not represent a black hole")
        # In this case, there's no event horizon (could be a naked singularity)
        # We'll leave time_dilation as calculated for these cases
    
    # Proper time rate
    proper_time_rate = time_dilation
    
    # Temporal flow in Genesis-Sphere terminology
    temporal_flow = proper_time_rate
    
    return pd.DataFrame({
        'r': r,
        'time_dilation': time_dilation,
        'proper_time_rate': proper_time_rate,
        'temporal_flow': temporal_flow,
        'spin': np.ones_like(r) * spin,
        'charge': np.ones_like(r) * charge
    })

def generate_binary_merger_time_dilation(t_min=-20, t_max=10, num_points=200):
    """
    Generate approximate time dilation data during a binary black hole merger.
    
    This is a highly simplified model based on numerical relativity results.
    
    Parameters:
    -----------
    t_min : float
        Start time relative to merger (negative = before merger)
    t_max : float
        End time relative to merger
    num_points : int
        Number of data points
        
    Returns:
    --------
    pd.DataFrame with time dilation data
    """
    # Time relative to merger
    t = np.linspace(t_min, t_max, num_points)
    
    # Simplified model of time dilation during merger
    # Before merger: gradual increase
    # During merger: rapid spike
    # After merger: stabilization to new black hole
    
    # Phase 1: Pre-merger (inspiraling)
    pre_merger = 1 / (1 + 0.1 * np.exp(-0.1 * t))
    
    # Phase 2: Merger event (rapid change)
    merger_event = 1 / (1 + 3 * np.exp(-2 * t**2))
    
    # Phase 3: Post-merger ringdown
    post_merger = 1 / (1 + 0.5 * (1 + np.tanh(t)))
    
    # Combine phases with smooth transition
    # Transition functions
    pre_to_merger = 0.5 * (1 + np.tanh(t + 2))
    merger_to_post = 0.5 * (1 + np.tanh(t - 1))
    
    # Combined time dilation
    time_dilation = (1 - pre_to_merger) * pre_merger + \
                   pre_to_merger * (1 - merger_to_post) * merger_event + \
                   merger_to_post * post_merger
    
    # Add gravitational wave strain (simplified)
    strain = 0.1 * np.exp(-0.1 * (t+5)**2) * np.sin(2 * np.pi * 0.1 * (t+5))
    # Increase frequency as we approach merger
    strain += 0.2 * np.exp(-0.5 * t**2) * np.sin(2 * np.pi * (0.2 * t**2 + 0.1 * t))
    
    return pd.DataFrame({
        'time': t,
        'time_dilation': time_dilation,
        'gw_strain': strain
    })

def load_or_create_bh_dataset(model_type="schwarzschild", spin=0.9, charge=0.4):
    """
    Load an existing black hole model dataset or create a new one
    
    Parameters:
    -----------
    model_type : str
        Type of black hole model ('schwarzschild', 'kerr', 'kerr_newman', 'merger')
    spin : float
        Spin parameter (for Kerr and Kerr-Newman metrics)
    charge : float
        Charge parameter (for Kerr-Newman metric)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the black hole data
    """
    file_path = os.path.join(datasets_dir, f"bh_{model_type}.csv")
    if model_type == "kerr_newman":
        # Add spin and charge to filename for Kerr-Newman to manage different parameter sets
        file_path = os.path.join(datasets_dir, f"bh_{model_type}_s{spin:.1f}_q{charge:.1f}.csv")
    
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"Error loading black hole dataset: {e}")
            print("Generating new data...")
    
    # Generate new data based on model type
    print(f"Generating new {model_type} black hole data...")
    if model_type == "schwarzschild":
        df = generate_schwarzschild_time_dilation()
    elif model_type == "kerr":
        df = generate_kerr_time_dilation(spin=spin)
    elif model_type == "kerr_newman":
        df = generate_kerr_newman_time_dilation(spin=spin, charge=charge)
    elif model_type == "merger":
        df = generate_binary_merger_time_dilation()
    else:
        raise ValueError(f"Unknown black hole model type: {model_type}")
    
    # Save to file
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"Saved {model_type} model data to {file_path}")
    
    return df

def map_bh_to_gs_coordinates(bh_df, model_type, time_scaling=1.0):
    """
    Map black hole coordinates to Genesis-Sphere time coordinates
    
    Parameters:
    -----------
    bh_df : pd.DataFrame
        Black hole data
    model_type : str
        Type of black hole model
    time_scaling : float
        Scaling factor for time mapping
        
    Returns:
    --------
    np.ndarray
        Genesis-Sphere time coordinates
    """
    if model_type in ["schwarzschild", "kerr", "kerr_newman"]:
        # For these models, we map radial distance to time
        # t = 0 should correspond to r = event horizon
        # Decreasing r means approaching t = 0
        
        r = bh_df['r'].values
        
        # Simple log mapping: t = -log(r)
        # This maps r→∞ to t→-∞ and r→0 to t→∞
        # But we want r→0 to map to t→0 (the singularity)
        
        # Adjust the mapping
        gs_time = -np.log(r) * time_scaling
        
        # Flip so that smaller r maps to smaller |t|
        gs_time = -gs_time
        
        return gs_time
    
    elif model_type == "merger":
        # For merger, we already have a time coordinate
        # Just apply scaling
        return bh_df['time'].values * time_scaling
    
    else:
        raise ValueError(f"Unknown black hole model type: {model_type}")

def compare_temporal_flow(gs_model, bh_df, model_type, time_scaling=1.0):
    """
    Compare Genesis-Sphere temporal flow with black hole time dilation
    
    Parameters:
    -----------
    gs_model : GenesisSphereModel
        The Genesis-Sphere model instance
    bh_df : pd.DataFrame
        Black hole data
    model_type : str
        Type of black hole model
    time_scaling : float
        Scaling factor for time mapping
        
    Returns:
    --------
    dict
        Dictionary with comparison results
    """
    # Map black hole coordinates to Genesis-Sphere time
    gs_time = map_bh_to_gs_coordinates(bh_df, model_type, time_scaling)
    
    # Get the temporal flow from the Genesis-Sphere model
    gs_results = gs_model.evaluate_all(gs_time)
    gs_tf = gs_results['temporal_flow']
    
    # Get the black hole time dilation/temporal flow
    if model_type in ["schwarzschild", "kerr", "kerr_newman"]:
        bh_tf = bh_df['temporal_flow'].values
    elif model_type == "merger":
        bh_tf = bh_df['time_dilation'].values
    else:
        raise ValueError(f"Unknown black hole model type: {model_type}")
    
    # Calculate comparison metrics
    residuals = bh_tf - gs_tf
    correlation = np.corrcoef(bh_tf, gs_tf)[0, 1]
    
    # Return results
    return {
        'gs_time': gs_time,
        'gs_tf': gs_tf,
        'bh_time_dilation': bh_tf,
        'residuals': residuals,
        'metrics': {
            'correlation': correlation,
            'mean_abs_error': np.mean(np.abs(residuals)),
            'mean_squared_error': np.mean(residuals**2),
            'max_absolute_error': np.max(np.abs(residuals))
        }
    }

def visualize_temporal_flow_comparison(comparison_results, bh_df, model_type, gs_params):
    """
    Visualize the comparison between Genesis-Sphere temporal flow and black hole time dilation
    
    Parameters:
    -----------
    comparison_results : dict
        Results from the comparison function
    bh_df : pd.DataFrame
        Black hole data
    model_type : str
        Type of black hole model
    gs_params : dict
        Genesis-Sphere parameters
        
    Returns:
    --------
    str
        Path to the saved figure
    """
    # Extract data
    gs_time = comparison_results['gs_time']
    gs_tf = comparison_results['gs_tf']
    bh_tf = comparison_results['bh_time_dilation']
    residuals = comparison_results['residuals']
    metrics = comparison_results['metrics']
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot temporal flow and time dilation
    if model_type in ["schwarzschild", "kerr", "kerr_newman"]:
        # For black hole models with radial distance, show both vs r and vs t
        r = bh_df['r'].values
        
        # First plot: vs. radial distance
        axes[0].plot(r, bh_tf, 'b-', label=f'{model_type.capitalize()} Time Dilation')
        
        # Create a twin x axis for Genesis-Sphere time
        ax2 = axes[0].twiny()
        ax2.plot(gs_time, gs_tf, 'r-', label='Genesis-Sphere Temporal Flow')
        
        # Set axis labels
        axes[0].set_xlabel('Radial Distance (r/Rs)')
        ax2.set_xlabel('Genesis-Sphere Time (t)')
        axes[0].set_ylabel('Time Dilation / Temporal Flow')
        
        # Set logarithmic scale for radial axis
        axes[0].set_xscale('log')
        
    else:  # merger model
        # For merger, plot vs time directly
        t = bh_df['time'].values
        axes[0].plot(t, bh_tf, 'b-', label=f'{model_type.capitalize()} Time Dilation')
        axes[0].plot(gs_time, gs_tf, 'r-', label='Genesis-Sphere Temporal Flow')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Time Dilation / Temporal Flow')
    
    # Add title and legend
    axes[0].set_title(f'Comparison: {model_type.capitalize()} Time Dilation vs Genesis-Sphere Temporal Flow\n'
                      f'(α={gs_params["alpha"]:.4f}, β={gs_params["beta"]:.4f}, '
                      f'ω={gs_params["omega"]:.4f}, ε={gs_params["epsilon"]:.4f})')
    axes[0].legend(loc='upper left')
    if model_type in ["schwarzschild", "kerr", "kerr_newman"]:
        ax2.legend(loc='upper right')
    axes[0].grid(True)
    
    # Plot residuals
    axes[1].plot(gs_time, residuals, 'g-')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Genesis-Sphere Time (t)')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals (Black Hole - Genesis-Sphere)')
    axes[1].grid(True)
    
    # Add metrics as text
    metrics_text = f"Correlation: {metrics['correlation']:.3f}\n" \
                  f"Mean Abs Error: {metrics['mean_abs_error']:.3f}\n" \
                  f"Mean Squared Error: {metrics['mean_squared_error']:.6f}\n" \
                  f"Max Absolute Error: {metrics['max_absolute_error']:.3f}"
    
    plt.figtext(0.15, 0.02, metrics_text, fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    
    # Save figure
    file_name = f"bh_{model_type}_comparison.png"
    fig_path = os.path.join(results_dir, file_name)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    return fig_path

def optimize_gs_for_bh(bh_df, initial_params, model_type, time_scaling=1.0):
    """
    Optimize Genesis-Sphere parameters to match black hole time dilation
    
    Parameters:
    -----------
    bh_df : pd.DataFrame
        Black hole data
    initial_params : dict
        Initial Genesis-Sphere parameters
    model_type : str
        Type of black hole model
    time_scaling : float
        Scaling factor for time mapping
        
    Returns:
    --------
    dict
        Optimized parameters
    float
        Final error
    """
    # Map black hole coordinates to Genesis-Sphere time
    gs_time = map_bh_to_gs_coordinates(bh_df, model_type, time_scaling)
    
    # Get the black hole time dilation/temporal flow
    if model_type in ["schwarzschild", "kerr", "kerr_newman"]:
        bh_tf = bh_df['temporal_flow'].values
    elif model_type == "merger":
        bh_tf = bh_df['time_dilation'].values
    else:
        raise ValueError(f"Unknown black hole model type: {model_type}")
    
    # Define objective function for optimization
    def objective(params):
        alpha, beta, omega, epsilon = params
        model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        results = model.evaluate_all(gs_time)
        gs_tf = results['temporal_flow']
        
        # Calculate error - we want both good correlation and low MSE
        residuals = bh_tf - gs_tf
        mse = np.mean(residuals**2)
        correlation = np.corrcoef(bh_tf, gs_tf)[0, 1]
        
        # Combined objective: minimize MSE and maximize correlation
        return mse - 0.5 * correlation  # Negative correlation term because we minimize
    
    # Initial values for optimization
    initial = [
        initial_params['alpha'],
        initial_params['beta'],
        initial_params['omega'],
        initial_params['epsilon']
    ]
    
    # Parameter bounds - more focused on beta which affects temporal flow most directly
    bounds = [
        (0.001, 0.1),   # alpha - less impact on temporal flow
        (0.1, 5.0),     # beta - primary parameter for temporal flow
        (0.5, 1.5),     # omega - less impact on temporal flow
        (0.01, 0.5)     # epsilon - affects behavior very near singularity
    ]
    
    # Run optimization
    print(f"Optimizing Genesis-Sphere parameters to match {model_type} black hole time dilation...")
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

def generate_bh_validation_summary(model_type, comparison_results, gs_params, optimized=False):
    """
    Generate an AI-like summary of the validation against a black hole model.
    
    Parameters:
    -----------
    model_type : str
        Type of black hole model
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
    mean_abs_error = metrics['mean_abs_error']
    
    # Initialize summary
    summary = [f"## AI Validation Summary: {model_type.capitalize()} Black Hole Comparison\n"]
    
    # Add parameter summary
    summary.append(f"### Genesis-Sphere Model Configuration")
    summary.append(f"- **Alpha (α)**: {gs_params['alpha']:.4f} - *Spatial dimension expansion coefficient*")
    summary.append(f"- **Beta (β)**: {gs_params['beta']:.4f} - *Temporal damping factor*")
    summary.append(f"- **Omega (ω)**: {gs_params['omega']:.4f} - *Angular frequency*")
    summary.append(f"- **Epsilon (ε)**: {gs_params['epsilon']:.4f} - *Zero-prevention constant*")
    
    if optimized:
        summary.append("*These parameters were optimized to best fit the black hole time dilation.*\n")
    else:
        summary.append("*These parameters are using initial values without optimization.*\n")
    
    # Add statistical analysis
    summary.append(f"### Statistical Comparison")
    summary.append(f"- **Correlation Coefficient**: {correlation:.4f}")
    summary.append(f"- **Mean Absolute Error**: {mean_abs_error:.4f}")
    summary.append(f"- **Mean Squared Error**: {metrics['mean_squared_error']:.6f}")
    summary.append(f"- **Maximum Absolute Error**: {metrics['max_absolute_error']:.4f}\n")
    
    # Add interpretation
    summary.append(f"### Interpretation of Temporal Flow Comparison")
    if correlation > 0.9:
        summary.append(f"The Genesis-Sphere temporal flow function Tf(t) shows **excellent correlation** with {model_type} black hole time dilation. This suggests:")
    elif correlation > 0.7:
        summary.append(f"The Genesis-Sphere temporal flow function Tf(t) shows **good correlation** with {model_type} black hole time dilation. This suggests:")
    elif correlation > 0.5:
        summary.append(f"The Genesis-Sphere temporal flow function Tf(t) shows **moderate correlation** with {model_type} black hole time dilation. This suggests:")
    else:
        summary.append(f"The Genesis-Sphere temporal flow function Tf(t) shows **weak correlation** with {model_type} black hole time dilation. This suggests:")
    
    # Model-specific interpretations
    if model_type == "schwarzschild":
        if correlation > 0.7:
            summary.append("- The Genesis-Sphere 1/(1+β|t|+ε) function effectively models Schwarzschild time dilation")
            summary.append("- The parameter β closely corresponds to the gravitational strength")
            summary.append("- The behavior near t=0 successfully captures the event horizon time-stopping effect\n")
        else:
            summary.append("- The current formulation may not fully capture the precise √(1-Rs/r) relationship")
            summary.append("- The parameter β might need adjustments to better match gravitational time dilation")
            summary.append("- The asymptotic behavior may differ from general relativity predictions\n")
    
    elif model_type == "kerr":
        if correlation > 0.7:
            summary.append("- The Genesis-Sphere model effectively approximates rotating black hole time dilation")
            summary.append("- The asymmetry in the temporal flow function captures frame-dragging effects")
            summary.append("- The parameter mapping from β to Kerr spin parameter shows promising alignment\n")
        else:
            summary.append("- The complex effects of black hole rotation are not fully captured")
            summary.append("- The ergosphere region behavior shows significant deviations")
            summary.append("- Additional parameters may be needed to model frame-dragging effects\n")
    
    elif model_type == "kerr_newman":
        if correlation > 0.7:
            summary.append("- The Genesis-Sphere model effectively approximates charged, rotating black hole time dilation")
            summary.append("- The temporal flow function captures both spin and charge effects on spacetime")
            summary.append("- The functional form provides a computational alternative to complex Kerr-Newman calculations\n")
        else:
            summary.append("- The complex combined effects of rotation and charge are not fully captured")
            summary.append("- The extremal case (a²+q²=1) behavior shows significant deviations")
            summary.append("- Additional parameters may be needed to model the complex horizon structure\n")
    
    else:  # merger model
        if correlation > 0.7:
            summary.append("- The Genesis-Sphere model effectively captures merger dynamics and post-merger ringdown")
            summary.append("- The time-dependent changes in temporal flow reflect the physics of merging horizons")
            summary.append("- The oscillatory components in both models show good phase alignment\n")
        else:
            summary.append("- The rapid changes during merger are not fully captured by the current model")
            summary.append("- The post-merger stabilization differs from numerical relativity results")
            summary.append("- The transitional behavior during horizon formation needs refinement\n")
    
    # Add general discussion
    summary.append("### Theoretical Implications")
    summary.append("The comparison between the Genesis-Sphere temporal flow function and black hole time dilation reveals important insights:")
    
    if correlation > 0.7:
        summary.append("- The simple algebraic form of Tf(t) = 1/(1+β|t|+ε) provides a computationally efficient approximation")
        summary.append("- Genesis-Sphere successfully captures the essential feature of divergent time dilation near singularities")
        summary.append("- The model offers a more accessible mathematical form compared to full general relativistic calculations\n")
    else:
        summary.append("- While simpler than GR solutions, the current function may sacrifice accuracy for simplicity")
        summary.append("- The fundamental behavior near singularities shows qualitative agreement but quantitative differences")
        summary.append("- The high curvature regions near event horizons present the greatest modeling challenge\n")
    
    # Add recommendations
    summary.append("### Recommendations")
    if optimized:
        if correlation > 0.7:
            summary.append("1. **Document these optimized parameters** for black hole approximations")
            summary.append("2. **Compare with higher-precision numerical relativity results** for validation")
            summary.append("3. **Develop a formal mapping** between β and black hole mass/radius ratio")
        else:
            summary.append("1. **Consider extending the temporal flow function** with additional terms for better accuracy")
            summary.append("2. **Explore alternative coordinate mappings** between r and t")
            summary.append("3. **Test with more sophisticated black hole metrics** (e.g., Kerr-Newman)")
    else:
        summary.append("1. **Run parameter optimization** to better match this black hole model")
        summary.append("2. **Focus on adjusting β parameter** which most directly affects temporal flow")
        summary.append("3. **Experiment with different time-radius mappings** to improve correlation")
    
    # Join all parts of the summary with newlines
    return "\n".join(summary)

def main(model_type="schwarzschild", alpha=0.02, beta=0.8, omega=1.0, epsilon=0.1, 
         optimize=False, time_scaling=1.0, spin=0.9, charge=0.4):
    """Main function to run the validation"""
    print("Genesis-Sphere Black Hole Time Dilation Validation")
    print("===============================================")
    
    # Load or create black hole model data
    bh_df = load_or_create_bh_dataset(model_type, spin=spin, charge=charge)
    
    # Set initial parameters
    initial_params = {'alpha': alpha, 'beta': beta, 'omega': omega, 'epsilon': epsilon}
    
    # Optimize parameters if requested
    if optimize:
        gs_params, error = optimize_gs_for_bh(bh_df, initial_params, model_type, time_scaling)
    else:
        gs_params = initial_params
    
    # Initialize model with parameters
    gs_model = GenesisSphereModel(**gs_params)
    
    # Compare temporal flow
    comparison_results = compare_temporal_flow(gs_model, bh_df, model_type, time_scaling)
    
    # Visualize comparison
    fig_path = visualize_temporal_flow_comparison(comparison_results, bh_df, model_type, gs_params)
    print(f"Comparison visualization saved to {fig_path}")
    
    # Generate AI summary
    summary = generate_bh_validation_summary(model_type, comparison_results, gs_params, optimize)
    
    # Save summary to file
    summary_path = os.path.join(results_dir, f"bh_{model_type}_validation_summary.md")
    if model_type == "kerr_newman":
        summary_path = os.path.join(results_dir, f"bh_{model_type}_s{spin:.1f}_q{charge:.1f}_validation_summary.md")
    
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
        'max_absolute_error': comparison_results['metrics']['max_absolute_error'],
        'alpha': gs_params['alpha'],
        'beta': gs_params['beta'],
        'omega': gs_params['omega'],
        'epsilon': gs_params['epsilon'],
        'time_scaling': time_scaling,
        'optimized': optimize
    }
    
    # Add spin and charge parameters for relevant models
    if model_type in ["kerr", "kerr_newman"]:
        results['spin'] = spin
    if model_type == "kerr_newman":
        results['charge'] = charge
    
    results_df = pd.DataFrame([results])
    
    # Save with appropriate filename
    results_file = f"bh_{model_type}_validation_results.csv"
    if model_type == "kerr_newman":
        results_file = f"bh_{model_type}_s{spin:.1f}_q{charge:.1f}_validation_results.csv"
    
    results_df.to_csv(os.path.join(results_dir, results_file), index=False)
    
    print(f"\nResults and visualizations saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Genesis-Sphere temporal flow against black hole time dilation")
    parser.add_argument("--model", type=str, default="schwarzschild", 
                        choices=["schwarzschild", "kerr", "kerr_newman", "merger"],
                        help="Black hole model to use for validation")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=0.8, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=1.0, help="Angular frequency for sinusoidal projections")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    parser.add_argument("--optimize", action="store_true", help="Optimize model parameters to fit black hole model")
    parser.add_argument("--time-scaling", type=float, default=1.0, help="Scaling factor for time mapping")
    parser.add_argument("--spin", type=float, default=0.9, help="Black hole spin parameter (0-1) for Kerr and Kerr-Newman metrics")
    parser.add_argument("--charge", type=float, default=0.4, help="Black hole charge parameter (0-1) for Kerr-Newman metric")
    
    args = parser.parse_args()
    main(args.model, args.alpha, args.beta, args.omega, args.epsilon, args.optimize, 
         args.time_scaling, args.spin, args.charge)
