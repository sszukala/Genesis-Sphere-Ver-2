"""
Parameter sweep validation to confirm optimal parameters for the Genesis-Sphere model.
This script runs multiple validations around the theoretically optimal parameters
(ω=2.0, β=1.2) to verify they provide the best fit for astronomical datasets.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from datetime import datetime
import json
import argparse

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Create results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'parameter_sweep')
os.makedirs(results_dir, exist_ok=True)

def run_validation_with_parameters(omega, beta, alpha=0.02, epsilon=0.1):
    """
    Run celestial_correlation_validation.py with specified parameters
    and capture the metrics.
    
    Parameters:
    -----------
    omega : float
        Angular frequency
    beta : float
        Temporal damping factor
    alpha : float, optional
        Spatial dimension expansion coefficient
    epsilon : float, optional
        Zero-prevention constant
        
    Returns:
    --------
    dict
        Dictionary containing validation metrics
    """
    print(f"Running validation with ω={omega:.2f}, β={beta:.2f}...")
    
    try:
        from validation.celestial_correlation_validation import load_h0_measurements, load_supernovae_data, load_bao_data
        from validation.celestial_correlation_validation import analyze_h0_correlation, analyze_sne_fit, analyze_bao_detection
        from models.genesis_model import GenesisSphereModel
        
        h0_data = load_h0_measurements()
        sne_data = load_supernovae_data()
        bao_data = load_bao_data()
        
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
        
        h0_metrics = analyze_h0_correlation(gs_model, h0_data)
        sne_metrics = analyze_sne_fit(gs_model, sne_data)
        bao_metrics = analyze_bao_detection(gs_model, bao_data)
        
        return {
            'omega': omega,
            'beta': beta,
            'alpha': alpha,
            'epsilon': epsilon,
            'h0_correlation': h0_metrics['correlation'],
            'sne_r_squared': sne_metrics['r_squared'],
            'sne_reduced_chi2': sne_metrics['reduced_chi2'],
            'bao_high_z_effect_size': bao_metrics['high_z_effect_size'],
            'bao_r_squared': bao_metrics['r_squared'],
            'combined_score': (
                0.4 * h0_metrics['correlation'] + 
                0.4 * sne_metrics['r_squared'] + 
                0.2 * min(1.0, bao_metrics['high_z_effect_size']/10)  
            )
        }
    except Exception as e:
        print(f"Error in direct import method: {e}")
        print("Falling back to subprocess method...")
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'celestial_correlation_validation.py'),
            f"--omega={omega}",
            f"--beta={beta}",
            f"--alpha={alpha}",
            f"--epsilon={epsilon}",
            "--metrics-only"  
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metrics = json.loads(result.stdout)
            return metrics
        except subprocess.CalledProcessError as e:
            print(f"Validation failed with error: {e}")
            print(f"Error output: {e.stderr}")
            return {
                'omega': omega,
                'beta': beta,
                'alpha': alpha,
                'epsilon': epsilon,
                'h0_correlation': -1.0,
                'sne_r_squared': -1.0,
                'bao_high_z_effect_size': 0.0,
                'combined_score': -1.0
            }

def perform_parameter_sweep(omega_range, beta_range, alpha=0.02, epsilon=0.1):
    """
    Perform a parameter sweep across specified ranges of ω and β.
    
    Parameters:
    -----------
    omega_range : tuple
        Range of omega values as (start, end, num)
    beta_range : tuple
        Range of beta values as (start, end, num)
    alpha : float
        Fixed alpha value
    epsilon : float
        Fixed epsilon value
        
    Returns:
    --------
    DataFrame
        Results of all validation runs
    """
    omega_values = np.linspace(*omega_range)
    beta_values = np.linspace(*beta_range)
    
    results = []
    
    total_combinations = len(omega_values) * len(beta_values)
    current = 0
    
    for omega in omega_values:
        for beta in beta_values:
            current += 1
            print(f"Running combination {current}/{total_combinations}: ω={omega:.4f}, β={beta:.4f}")
            metrics = run_validation_with_parameters(omega, beta, alpha, epsilon)
            results.append(metrics)
    
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"parameter_sweep_{timestamp}.csv")
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    return df

def analyze_parameter_sweep(results_df, theoretical_optimal_params):
    """
    Analyze parameter sweep results and generate visualizations.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from parameter sweep
    theoretical_optimal_params : dict
        Dictionary containing the theoretical optimal parameters {'omega', 'beta', 'alpha', 'epsilon'}
        
    Returns:
    --------
    dict
        Containing 'best_params' dict and 'theoretical_optimal_metrics' dict (or None)
    """
    best_idx = results_df['combined_score'].idxmax()
    best_params = results_df.iloc[best_idx].to_dict() 
    
    print("\n=== Best Parameter Combination Found ===")
    print(f"ω = {best_params['omega']:.4f}")
    print(f"β = {best_params['beta']:.4f}")
    print(f"H₀ Correlation: {best_params['h0_correlation']*100:.2f}%")
    print(f"Supernovae R²: {best_params['sne_r_squared']*100:.2f}%")
    print(f"BAO Effect Size: {best_params['bao_high_z_effect_size']:.2f}")
    print(f"Combined Score: {best_params['combined_score']:.4f}")
    
    theoretical_omega = theoretical_optimal_params['omega']
    theoretical_beta = theoretical_optimal_params['beta']
    
    results_df['dist_to_theory'] = np.sqrt(
        (results_df['omega'] - theoretical_omega)**2 + 
        (results_df['beta'] - theoretical_beta)**2
    )
    
    closest_idx = results_df['dist_to_theory'].idxmin()
    theoretical_optimal_metrics = results_df.iloc[closest_idx].to_dict() 
    
    if theoretical_optimal_metrics['dist_to_theory'] < 0.5: 
        print(f"\n=== Performance at Parameters Closest to Theoretical Optimal (ω={theoretical_omega:.4f}, β={theoretical_beta:.4f}) ===")
        print(f"Actual ω = {theoretical_optimal_metrics['omega']:.4f}")
        print(f"Actual β = {theoretical_optimal_metrics['beta']:.4f}")
        print(f"H₀ Correlation: {theoretical_optimal_metrics['h0_correlation']*100:.2f}%")
        print(f"Supernovae R²: {theoretical_optimal_metrics['sne_r_squared']*100:.2f}%")
        print(f"BAO Effect Size: {theoretical_optimal_metrics['bao_high_z_effect_size']:.2f}")
        print(f"Combined Score: {theoretical_optimal_metrics['combined_score']:.4f}")
    else:
        print(f"\nTheoretical optimal parameters (ω={theoretical_omega:.1f}, β={theoretical_beta:.1f}) were not closely sampled in this sweep.")
        theoretical_optimal_metrics = None 
        
    print("\nGenerating combined score heatmap...", end="", flush=True)
    plt.figure(figsize=(12, 10))
    
    pivot_data = results_df.pivot(index="beta", columns="omega", values="combined_score")
    
    ax = sns.heatmap(pivot_data, cmap="viridis", annot=True, fmt=".3f")
    ax.invert_yaxis()  
    
    mark_params(ax, best_params['omega'], best_params['beta'], "Best", color='red')
    
    if theoretical_optimal_metrics is not None:
        mark_params(ax, theoretical_optimal_metrics['omega'], theoretical_optimal_metrics['beta'], "Theory", color='blue')
    
    plt.title("Combined Score by Parameter Combination")
    plt.tight_layout()
    
    heatmap_file = os.path.join(results_dir, "combined_score_heatmap.png")
    plt.savefig(heatmap_file, dpi=150)
    print(f" Done! Saved to {heatmap_file}")
    
    for metric in ['h0_correlation', 'sne_r_squared', 'bao_high_z_effect_size']:
        create_metric_heatmap(results_df, metric, best_params, theoretical_optimal_metrics) 
    
    return {
        'best_params': best_params,
        'theoretical_optimal_metrics': theoretical_optimal_metrics
    }

def mark_params(ax, omega, beta, label, color='red'):
    """Mark specific parameters on a heatmap plot"""
    try:
        omega_idx = ax.get_xticklabels()
        beta_idx = ax.get_yticklabels()
        
        omega_pos = min(range(len(omega_idx)), key=lambda i: abs(float(omega_idx[i].get_text()) - omega))
        beta_pos = min(range(len(beta_idx)), key=lambda i: abs(float(beta_idx[i].get_text()) - beta))
        
        ax.add_patch(plt.Circle((omega_pos + 0.5, beta_pos + 0.5), 0.4, fill=False, edgecolor=color, lw=2))
        ax.text(omega_pos + 0.5, beta_pos + 0.5, label, ha='center', va='center', color=color, fontweight='bold')
    except Exception as e:
        print(f"Could not mark parameters: {e}")

def create_metric_heatmap(results_df, metric, best_params, theoretical_optimal_metrics):
    """Create a heatmap for a specific metric"""
    print(f"\nGenerating {metric} heatmap...", end="", flush=True)
    
    plt.figure(figsize=(12, 10))
    
    pivot_data = results_df.pivot(index="beta", columns="omega", values=metric)
    
    ax = sns.heatmap(pivot_data, cmap="viridis", annot=True, fmt=".3f")
    ax.invert_yaxis() 
    
    mark_params(ax, best_params['omega'], best_params['beta'], "Best", color='red')
    
    if theoretical_optimal_metrics is not None:
        mark_params(ax, theoretical_optimal_metrics['omega'], theoretical_optimal_metrics['beta'], "Theory", color='blue')
        
    plt.title(f"{metric.replace('_', ' ').title()} by Parameter Combination")
    plt.tight_layout()
    
    heatmap_file = os.path.join(results_dir, f"{metric}_heatmap.png")
    plt.savefig(heatmap_file, dpi=150)
    print(f" Done! Saved to {heatmap_file}")
    plt.close()

def generate_validation_summary(results_df, best_params, theoretical_optimal_metrics, sweep_center_params):
    """Generate a markdown summary of the parameter sweep validation"""
    summary = [
        "# Genesis-Sphere Parameter Sweep Validation Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Validation Purpose",
        "\nThis validation performs a parameter sweep around potentially optimal parameters ",
        f"(centered near ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}) to verify performance against astronomical datasets.\n",
        "## Parameter Ranges Tested",
        f"\n- Omega (ω): {results_df['omega'].min():.2f} to {results_df['omega'].max():.2f} ({len(results_df['omega'].unique())} steps)",
        f"- Beta (β): {results_df['beta'].min():.2f} to {results_df['beta'].max():.2f} ({len(results_df['beta'].unique())} steps)",
        f"- Total Combinations Tested: {len(results_df)}\n",
        "## Best Parameter Combination Found in Sweep",
        "\nBased on combined performance across H₀ correlation, supernovae distance modulus fitting, and BAO signal detection:",
        f"\n| Parameter | Value | Description |",
        "|-----------|-------|-------------|",
        f"| Omega (ω) | {best_params['omega']:.4f} | Angular frequency |",
        f"| Beta (β) | {best_params['beta']:.4f} | Temporal damping factor |",
        f"| Alpha (α) | {best_params['alpha']:.4f} | Spatial dimension expansion coefficient |",
        f"| Epsilon (ε) | {best_params['epsilon']:.4f} | Zero-prevention constant |\n",
        "### Performance Metrics",
        f"\n- H₀ Correlation: {best_params['h0_correlation']*100:.2f}%",
        f"- Supernovae R²: {best_params['sne_r_squared']*100:.2f}%",
        f"- BAO Effect Size: {best_params['bao_high_z_effect_size']:.2f}",
        f"- Combined Score: {best_params['combined_score']:.4f}\n"
    ]
    
    if theoretical_optimal_metrics is not None:
        summary.extend([
            "## Performance at Parameters Closest to Sweep Center",
            f"\nPerformance at parameters closest to the sweep center (ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}):",
            f"\n| Parameter | Value | Description |",
            "|-----------|-------|-------------|",
            f"| Omega (ω) | {theoretical_optimal_metrics['omega']:.4f} | Angular frequency |",
            f"| Beta (β) | {theoretical_optimal_metrics['beta']:.4f} | Temporal damping factor |",
            f"| Alpha (α) | {theoretical_optimal_metrics['alpha']:.4f} | Spatial dimension expansion coefficient |",
            f"| Epsilon (ε) | {theoretical_optimal_metrics['epsilon']:.4f} | Zero-prevention constant |\n",
            "### Performance Metrics",
            f"\n- H₀ Correlation: {theoretical_optimal_metrics['h0_correlation']*100:.2f}%",
            f"- Supernovae R²: {theoretical_optimal_metrics['sne_r_squared']*100:.2f}%",
            f"- BAO Effect Size: {theoretical_optimal_metrics['bao_high_z_effect_size']:.2f}",
            f"- Combined Score: {theoretical_optimal_metrics['combined_score']:.4f}\n",
            "## Comparison: Best Found vs. Sweep Center",
            "\nThis comparison highlights the difference between the best parameters found in the sweep and the parameters closest to the sweep's starting center point.",
            f"\n- **Parameter Distance**: Euclidean distance between (ω, β) points = {theoretical_optimal_metrics['dist_to_theory']:.4f}",
            f"- **Combined Score Difference**: {best_params['combined_score'] - theoretical_optimal_metrics['combined_score']:.4f} (Higher is better for 'Best Found')",
            f"- **H₀ Correlation Difference**: {(best_params['h0_correlation'] - theoretical_optimal_metrics['h0_correlation'])*100:.2f}%",
            f"- **Supernovae R² Difference**: {(best_params['sne_r_squared'] - theoretical_optimal_metrics['sne_r_squared'])*100:.2f}%",
            f"- **BAO Effect Size Difference**: {best_params['bao_high_z_effect_size'] - theoretical_optimal_metrics['bao_high_z_effect_size']:.2f}\n"
        ])
        
        omega_diff = abs(best_params['omega'] - theoretical_optimal_metrics['omega'])
        beta_diff = abs(best_params['beta'] - theoretical_optimal_metrics['beta'])
        score_diff = best_params['combined_score'] - theoretical_optimal_metrics['combined_score']
        
        # Determine if the best parameters are significantly different from the center
        is_close_params = omega_diff < (results_df['omega'].max() - results_df['omega'].min()) / 4 and \
                          beta_diff < (results_df['beta'].max() - results_df['beta'].min()) / 4
        is_close_score = abs(score_diff) < 0.1 * abs(theoretical_optimal_metrics['combined_score']) if theoretical_optimal_metrics['combined_score'] != 0 else abs(score_diff) < 0.05

        summary.append("### Interpretation of Comparison")
        if is_close_params and is_close_score:
            conclusion = f"The best parameters found (ω={best_params['omega']:.2f}, β={best_params['beta']:.2f}) are very close to the sweep center (ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}) both in parameter space and performance score. This suggests the sweep center is near a local optimum within the tested range."
        elif is_close_params and not is_close_score:
            conclusion = f"The best parameters found (ω={best_params['omega']:.2f}, β={best_params['beta']:.2f}) are near the sweep center (ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}), but offer a significantly different performance score ({score_diff:+.4f}). This indicates a steep gradient in performance near the center."
        elif not is_close_params and is_close_score:
            conclusion = f"The best parameters found (ω={best_params['omega']:.2f}, β={best_params['beta']:.2f}) are relatively far from the sweep center (ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}), yet yield a similar performance score ({score_diff:+.4f}). This might indicate a plateau or multiple optima within the sweep range."
        else: # Not close params, not close score
             conclusion = f"The best parameters found (ω={best_params['omega']:.2f}, β={best_params['beta']:.2f}) are significantly different from the sweep center (ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}) and provide a notably different performance score ({score_diff:+.4f}). This suggests the optimal parameters may lie towards the edge or outside of the current sweep range, warranting further investigation in that direction."

        summary.extend(["\n" + conclusion + "\n"])
            
    else:
        summary.extend([
            "## Conclusion",
            f"\nThe sweep center parameters (ω={sweep_center_params['omega']:.2f}, β={sweep_center_params['beta']:.2f}) were not closely sampled in this sweep, so a direct comparison is not available.",
            "The best parameters found represent the optimum within the tested range.\n"
        ])

    summary.extend([
        "## Visualizations",
        "\n### Combined Score Heatmap",
        "![Combined Score Heatmap](combined_score_heatmap.png)",
        "\nThis heatmap shows the combined performance score across all metrics for different parameter combinations. Brighter colors indicate better performance.\n",
        "### Individual Metric Heatmaps",
        "![H₀ Correlation Heatmap](h0_correlation_heatmap.png)",
        "\nThis heatmap shows how well the model correlates with historical H₀ measurements across parameter combinations.\n",
        "![Supernovae R² Heatmap](sne_r_squared_heatmap.png)",
        "\nThis heatmap shows the R² values for fitting Type Ia supernovae distance modulus data across parameter combinations.\n",
        "![BAO Effect Size Heatmap](bao_high_z_effect_size_heatmap.png)",
        "\nThis heatmap shows the effect size for BAO signal detection at z~2.3 across parameter combinations.\n",
        "---",
        "\n*This report was automatically generated by the Genesis-Sphere parameter sweep validation framework.*"
    ])
    
    summary_path = os.path.join(results_dir, "parameter_sweep_summary.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"Validation summary saved to: {summary_path}")

def run_parameter_sweep(center_omega=2.3, center_beta=0.9, 
                        sweep_range_omega=0.3, sweep_range_beta=0.3, 
                        n_omega_steps=9, n_beta_steps=8, 
                        alpha=0.02, epsilon=0.1):
    """
    Performs a parameter sweep around a central point for omega and beta.
    
    Parameters:
    -----------
    center_omega : float
        Central value for the omega parameter sweep.
    center_beta : float
        Central value for the beta parameter sweep.
    sweep_range_omega : float
        Half-width of the sweep range for omega (e.g., 0.3 means center_omega +/- 0.3).
    sweep_range_beta : float
        Half-width of the sweep range for beta.
    n_omega_steps : int
        Number of steps to take for omega within the range.
    n_beta_steps : int
        Number of steps to take for beta within the range.
    alpha : float
        Fixed value for the alpha parameter.
    epsilon : float
        Fixed value for the epsilon parameter.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the results of the parameter sweep.
    """
    omega_values = np.linspace(center_omega - sweep_range_omega, center_omega + sweep_range_omega, n_omega_steps)
    beta_values = np.linspace(center_beta - sweep_range_beta, center_beta + sweep_range_beta, n_beta_steps)
    
    results = []
    
    total_combinations = len(omega_values) * len(beta_values)
    current = 0
    
    for omega in omega_values:
        for beta in beta_values:
            current += 1
            print(f"Running combination {current}/{total_combinations}: ω={omega:.4f}, β={beta:.4f}")
            metrics = run_validation_with_parameters(omega, beta, alpha, epsilon)
            results.append(metrics)
    
    df = pd.DataFrame(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"parameter_sweep_{timestamp}.csv")
    df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    return df

def main():
    """Main function to run the parameter sweep validation"""
    parser = argparse.ArgumentParser(description="Run parameter sweep validation for Genesis-Sphere")
    parser.add_argument("--mode", type=str, default="default", choices=["default", "focused", "broad"], 
                        help="Sweep mode preset ('default', 'focused', 'broad')")
    parser.add_argument("--center_omega", type=float, help="Center value for omega sweep (overrides mode)")
    parser.add_argument("--center_beta", type=float, help="Center value for beta sweep (overrides mode)")
    parser.add_argument("--range_omega", type=float, help="Half-width of omega sweep range (overrides mode)")
    parser.add_argument("--range_beta", type=float, help="Half-width of beta sweep range (overrides mode)")
    parser.add_argument("--steps_omega", type=int, help="Number of steps for omega (overrides mode)")
    parser.add_argument("--steps_beta", type=int, help="Number of steps for beta (overrides mode)")
    parser.add_argument("--alpha", type=float, default=0.02, help="Fixed alpha value")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Fixed epsilon value")
    
    args = parser.parse_args()

    # --- Mode Presets ---
    modes = {
        "default": { # Updated center to current theoretical optimal point
            "center_omega": 2.9, "center_beta": 0.3,  # New theoretical optimal point from previous sweep
            "range_omega": 0.3, "range_beta": 0.3,    # Keep similar range
            "steps_omega": 10, "steps_beta": 10       # 10 * 10 = 100 combinations
        },
        "focused": { # Tighter range around the new optimal point
            "center_omega": 2.9, "center_beta": 0.3,
            "range_omega": 0.1, "range_beta": 0.1,
            "steps_omega": 11, "steps_beta": 11      # ~121 combinations
        },
        "broad": { # Wider range around the new optimal point
            "center_omega": 2.9, "center_beta": 0.3,
            "range_omega": 1.0, "range_beta": 1.0,
            "steps_omega": 7, "steps_beta": 7        # ~49 combinations
        }
    }
    
    selected_mode = modes.get(args.mode, modes["default"])

    # Apply mode presets unless overridden by command-line arguments
    center_omega = args.center_omega if args.center_omega is not None else selected_mode["center_omega"]
    center_beta = args.center_beta if args.center_beta is not None else selected_mode["center_beta"]
    range_omega = args.range_omega if args.range_omega is not None else selected_mode["range_omega"]
    range_beta = args.range_beta if args.range_beta is not None else selected_mode["range_beta"]
    steps_omega = args.steps_omega if args.steps_omega is not None else selected_mode["steps_omega"]
    steps_beta = args.steps_beta if args.steps_beta is not None else selected_mode["steps_beta"]
    
    print(f"Starting Genesis-Sphere Parameter Sweep Validation (Mode: {args.mode})...")
    print(f"Sweep Center: ω={center_omega}, β={center_beta}")
    print(f"Sweep Range: ω=[{center_omega - range_omega:.2f}, {center_omega + range_omega:.2f}], "
          f"β=[{center_beta - range_beta:.2f}, {center_beta + range_beta:.2f}]")
    print(f"Sweep Steps: ω={steps_omega}, β={steps_beta} (Total: {steps_omega * steps_beta})")
    print(f"Fixed Parameters: α={args.alpha}, ε={args.epsilon}")
    
    # Run the sweep using the determined parameters
    results_df = run_parameter_sweep(
        center_omega=center_omega,
        center_beta=center_beta,
        sweep_range_omega=range_omega,
        sweep_range_beta=range_beta,
        n_omega_steps=steps_omega,
        n_beta_steps=steps_beta,
        alpha=args.alpha,
        epsilon=args.epsilon
    )
    
    # Use the actual sweep center points for analysis and summary generation
    sweep_center_params = {'omega': center_omega, 'beta': center_beta, 'alpha': args.alpha, 'epsilon': args.epsilon}
    best_params_info = analyze_parameter_sweep(results_df, sweep_center_params) # Pass sweep center instead of theoretical
    
    summary = generate_validation_summary(results_df, 
                                          best_params_info['best_params'], 
                                          best_params_info['theoretical_optimal_metrics'],
                                          sweep_center_params) 
    
    print("\nParameter sweep validation complete!")
    print(f"Results and visualizations saved to {results_dir}")
    
    return summary

if __name__ == "__main__":
    main()
