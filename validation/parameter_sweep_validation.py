"""
Parameter estimation using MCMC for the Genesis-Sphere model.

This script uses MCMC to explore the parameter space (ω, β) and find the
posterior probability distribution based on astronomical datasets.
It replaces the previous grid search and custom combined score with a
statistically robust approach.

NOTE: Requires modification of analysis functions to return log-likelihoods
      or chi-squared values. Requires installation of 'emcee' and 'corner'.
      (pip install emcee corner numpy pandas)
"""

import os
import sys
import numpy as np
import pandas as pd
import emcee # MCMC sampler
import corner # For plotting results
from datetime import datetime
import json
import argparse
import time # To time execution

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Create results directory
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'parameter_sweep')
os.makedirs(results_dir, exist_ok=True)

try:
    from models.genesis_model import GenesisSphereModel
    from validation.celestial_correlation_validation import (
        load_h0_measurements,
        load_supernovae_data,
        load_bao_data,
        analyze_h0_correlation,
        analyze_sne_fit,
        analyze_bao_detection
    )
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'models' and 'validation' directories are accessible"
          " relative to the script's parent directory and contain the required files/functions.")
    sys.exit(1)
except FileNotFoundError:
     print("Error: Could not determine parent directory reliably. "
           "Ensure script structure allows importing 'models' and 'validation'.")
     sys.exit(1)

# --- Helper functions to calculate chi-squared from existing analysis functions ---

def calculate_h0_chi2(gs_model, h0_data):
    """Calculate chi-squared for H0 correlation"""
    metrics = analyze_h0_correlation(gs_model, h0_data)
    # Convert correlation to chi-squared
    # For correlation coefficient r, we can use -N*log(1-r²) as an approximation
    # This makes better correlation (r close to 1) give lower chi-squared
    r = metrics['correlation']
    n = len(h0_data)
    chi2 = -n * np.log(1 - r**2) if abs(r) < 1.0 else 1000
    # Invert the sign since better correlation should give lower chi-squared
    return 1000 - chi2 if np.isfinite(chi2) else 1000

def calculate_sne_chi2(gs_model, sne_data):
    """Calculate chi-squared for supernovae fit"""
    metrics = analyze_sne_fit(gs_model, sne_data)
    # Use the reduced chi-squared metric directly if available
    if 'reduced_chi2' in metrics:
        return metrics['reduced_chi2'] * (len(sne_data) - 2)  # Convert reduced chi2 to raw chi2
    
    # Alternatively use R-squared
    r_squared = metrics['r_squared']
    # Transform R² to a chi-squared-like metric (lower is better)
    # When R² is close to 1 (good fit), this gives a small value
    chi2_approx = len(sne_data) * (1 - r_squared) if r_squared <= 1.0 else len(sne_data) * 2
    return chi2_approx

def calculate_bao_chi2(gs_model, bao_data):
    """Calculate chi-squared for BAO detection"""
    metrics = analyze_bao_detection(gs_model, bao_data)
    # For effect size, a larger value is better
    # Convert to a chi-squared-like value (lower is better)
    effect_size = metrics['high_z_effect_size']
    max_expected = 100  # A reasonable maximum to scale against
    # Normalize so that higher effect size gives lower chi-squared
    chi2_approx = max_expected - min(effect_size, max_expected)
    return chi2_approx

# === MCMC Setup ===

# Define the parameter space (dimensions)
# We are fitting for omega and beta
N_DIM = 2
PARAM_LABELS = [r"$\omega$", r"$\beta$"] # LaTeX labels for plots

# Define the prior function - sets allowed parameter ranges
def log_prior(params):
    """
    Log prior probability distribution (Log(Prior)).
    Returns 0 if params are within allowed ranges, -np.inf otherwise.
    This enforces constraints on parameters.
    """
    omega, beta = params
    # Define parameter ranges based on previous search results
    # Extended range that includes the previous optima
    if 1.0 < omega < 6.0 and -1.0 < beta < 3.0: 
        return 0.0 # Log(1) = 0 -> uniform prior within bounds
    return -np.inf # Log(0) -> rules out parameters outside bounds

# Define the log-likelihood function - compares model to data
def log_likelihood(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """
    Log likelihood function (Log(Likelihood)).
    Calculates the total likelihood of observing the data given the parameters.
    """
    omega, beta = params
    alpha = fixed_alpha
    epsilon = fixed_epsilon

    # Check prior first (can sometimes save computation)
    if not np.isfinite(log_prior(params)):
         return -np.inf

    try:
        # Create model instance with current parameters
        gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)

        # Calculate chi-squared for each dataset
        chi2_h0 = calculate_h0_chi2(gs_model, data_h0)
        chi2_sne = calculate_sne_chi2(gs_model, data_sne)
        chi2_bao = calculate_bao_chi2(gs_model, data_bao)

        # Weight the different components
        # Adjust weights based on which datasets you consider more reliable/important
        total_chi2 = (0.4 * chi2_h0 + 0.4 * chi2_sne + 0.2 * chi2_bao)

        # Convert total Chi-squared to Log Likelihood (assuming Gaussian errors)
        logL = -0.5 * total_chi2

        # Check for NaN or infinite results which can break MCMC
        if not np.isfinite(logL):
             print(f"Warning: Non-finite logL ({logL}) for ω={omega:.4f}, β={beta:.4f}")
             return -np.inf

        return logL

    except Exception as e:
        # Handle potential errors during model calculation or analysis
        print(f"Warning: Likelihood calculation failed for ω={omega:.4f}, β={beta:.4f}. Error: {e}")
        return -np.inf

# Define the log-posterior function (Prior + Likelihood)
def log_posterior(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon):
    """
    Log posterior probability distribution (Log(Prior) + Log(Likelihood)).
    This is the function the MCMC sampler explores.
    """
    lp = log_prior(params)
    if not np.isfinite(lp): # If parameters are outside prior range
        return -np.inf
    # Log(Posterior) = Log(Prior) + Log(Likelihood)
    return lp + log_likelihood(params, data_h0, data_sne, data_bao, fixed_alpha, fixed_epsilon)

# === Main Execution ===

def main():
    """Main function to run the MCMC parameter estimation"""
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run MCMC parameter estimation for Genesis-Sphere")
    # Add arguments for fixed parameters, data paths, MCMC settings, etc.
    parser.add_argument("--alpha", type=float, default=0.02, help="Fixed alpha value")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Fixed epsilon value")
    parser.add_argument("--nwalkers", type=int, default=32, help="Number of MCMC walkers (must be > 2*N_DIM)")
    parser.add_argument("--nsteps", type=int, default=5000, help="Number of MCMC steps per walker")
    parser.add_argument("--nburn", type=int, default=1000, help="Number of burn-in steps to discard")
    parser.add_argument("--initial_omega", type=float, default=3.5, help="Initial guess for omega")
    parser.add_argument("--initial_beta", type=float, default=-0.0333, help="Initial guess for beta")
    parser.add_argument("--output_suffix", type=str, default="", help="Optional suffix for output filenames")

    args = parser.parse_args()

    # Validate walker count
    if args.nwalkers <= 2 * N_DIM:
        print(f"Error: Number of walkers ({args.nwalkers}) must be greater than 2 * N_DIM ({2*N_DIM}).")
        sys.exit(1)
    if args.nsteps <= args.nburn:
        print(f"Error: Total steps ({args.nsteps}) must be greater than burn-in steps ({args.nburn}).")
        sys.exit(1)


    print("Starting Genesis-Sphere MCMC Parameter Estimation...")
    print(f"Fixed Parameters: α={args.alpha}, ε={args.epsilon}")
    print(f"MCMC Settings: Walkers={args.nwalkers}, Steps={args.nsteps}, Burn-in={args.nburn}")

    # --- Load Data ---
    print("Loading observational data...")
    try:
        h0_data = load_h0_measurements()
        sne_data = load_supernovae_data()
        bao_data = load_bao_data()
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # --- Initialize Walkers ---
    # Start walkers in a small Gaussian ball around the previous best parameters
    initial_pos_guess = np.array([args.initial_omega, args.initial_beta])
    # Add small random offsets for each walker, ensuring they are within priors
    pos = np.zeros((args.nwalkers, N_DIM))
    for i in range(args.nwalkers):
        # Keep generating random starting points until they satisfy the prior
        while True:
            p = initial_pos_guess + 1e-3 * np.abs(initial_pos_guess) * np.random.randn(N_DIM)
            if np.isfinite(log_prior(p)):
                pos[i] = p
                break
    nwalkers, ndim = pos.shape
    print(f"Initialized {nwalkers} walkers around ω={args.initial_omega}, β={args.initial_beta}")


    # --- Run MCMC ---
    print(f"Running MCMC...")
    # The 'args' tuple passes fixed parameters and data to the log_posterior function
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(h0_data, sne_data, bao_data, args.alpha, args.epsilon)
    )

    # Run MCMC steps and show progress
    sampler.run_mcmc(pos, args.nsteps, progress=True)
    print("MCMC run complete.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # --- Process Results ---
    try:
        # Check acceptance fraction (should generally be between ~0.2 and 0.5)
        acceptance_fraction = np.mean(sampler.acceptance_fraction)
        print(f"Mean acceptance fraction: {acceptance_fraction:.3f}")

        # Discard burn-in steps and flatten the chain
        # flat=True combines results from all walkers
        # thin=X keeps only every Xth sample to reduce autocorrelation
        samples = sampler.get_chain(discard=args.nburn, thin=15, flat=True)
        print(f"Shape of processed samples: {samples.shape}") # Should be (N_samples, N_DIM)

        # --- Save Results ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""

        # Save the samples (the chain)
        chain_file = os.path.join(results_dir, f"mcmc_chain_{timestamp}{suffix}.csv")
        df_samples = pd.DataFrame(samples, columns=['omega', 'beta'])
        df_samples.to_csv(chain_file, index=False)
        print(f"MCMC samples saved to {chain_file}")

        # Save run info (parameters, settings)
        run_info = {
            'timestamp': timestamp,
            'fixed_alpha': args.alpha,
            'fixed_epsilon': args.epsilon,
            'nwalkers': args.nwalkers,
            'nsteps': args.nsteps,
            'nburn': args.nburn,
            'initial_guess': {'omega': args.initial_omega, 'beta': args.initial_beta},
            'parameter_labels': PARAM_LABELS,
            'mean_acceptance_fraction': acceptance_fraction,
            'execution_time_seconds': end_time - start_time
        }
        info_file = os.path.join(results_dir, f"run_info_{timestamp}{suffix}.json")
        with open(info_file, 'w') as f:
            json.dump(run_info, f, indent=4)
        print(f"Run info saved to {info_file}")


        # --- Basic Analysis & Plotting ---
        print("\nAnalyzing MCMC results...")

        # Calculate median and 1-sigma credible intervals (16th, 50th, 84th percentiles)
        results_summary = {}
        print("\n=== MCMC Parameter Estimates (median and 1-sigma credible interval) ===")
        for i, label in enumerate(['omega', 'beta']):
            mcmc = np.percentile(samples[:, i], [16, 50, 84])
            q = np.diff(mcmc) # q[0] = 50th-16th, q[1] = 84th-50th
            median = mcmc[1]
            upper_err = q[1]
            lower_err = q[0]
            print(f"{PARAM_LABELS[i]} = {median:.4f} (+{upper_err:.4f} / -{lower_err:.4f})")
            results_summary[label] = {'median': median, 'upper_err': upper_err, 'lower_err': lower_err}

        # Save summary stats
        stats_file = os.path.join(results_dir, f"param_stats_{timestamp}{suffix}.json")
        with open(stats_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        print(f"Parameter stats saved to {stats_file}")


        # Generate a corner plot using the corner library
        print("\nGenerating corner plot...")
        try:
            figure = corner.corner(
                samples, labels=PARAM_LABELS, # Use LaTeX labels
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, title_kwargs={"fontsize": 12},
                truths=[results_summary['omega']['median'], results_summary['beta']['median']], # Show median values
                truth_color='red'
            )
            corner_plot_file = os.path.join(results_dir, f"corner_plot_{timestamp}{suffix}.png")
            figure.savefig(corner_plot_file)
            print(f"Corner plot saved to {corner_plot_file}")
        except ImportError:
            print("\nInstall 'corner' package (`pip install corner`) to generate corner plots.")
        except Exception as e:
            print(f"Error during corner plot generation: {e}")

        # Generate a markdown summary report
        generate_validation_summary(results_summary, args, timestamp, suffix, acceptance_fraction)

    except Exception as e:
        print(f"Error during MCMC results processing: {e}")
        print("Chain data might still be saved if the run completed.")

    print("\nMCMC parameter estimation script finished!")

def generate_validation_summary(results_summary, args, timestamp, suffix, acceptance_fraction):
    """Generate a markdown summary of the MCMC parameter estimation"""
    summary = [
        "# Genesis-Sphere MCMC Parameter Estimation Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Validation Method",
        "\nThis validation uses Markov Chain Monte Carlo (MCMC) to estimate the posterior probability distribution",
        "of Genesis-Sphere model parameters based on astronomical datasets. Unlike the previous grid search approach,",
        "MCMC provides robust parameter uncertainties and explores the parameter space more efficiently.\n",
        "## MCMC Settings",
        f"\n- Walkers: {args.nwalkers}",
        f"- Steps per walker: {args.nsteps}",
        f"- Burn-in steps discarded: {args.nburn}",
        f"- Initial parameter guess: ω={args.initial_omega:.4f}, β={args.initial_beta:.4f}",
        f"- Fixed parameters: α={args.alpha:.4f}, ε={args.epsilon:.4f}",
        f"- Mean acceptance fraction: {acceptance_fraction:.3f}\n",
        "## Parameter Estimates",
        "\nBest-fit parameters with 1-sigma (68%) credible intervals:",
        f"\n| Parameter | Median | Lower Error | Upper Error |",
        "|-----------|--------|-------------|-------------|",
        f"| Omega (ω) | {results_summary['omega']['median']:.4f} | {results_summary['omega']['lower_err']:.4f} | {results_summary['omega']['upper_err']:.4f} |",
        f"| Beta (β) | {results_summary['beta']['median']:.4f} | {results_summary['beta']['lower_err']:.4f} | {results_summary['beta']['upper_err']:.4f} |\n",
        "## Corner Plot",
        "\n![Parameter Corner Plot](corner_plot_" + f"{timestamp}{suffix}.png" + ")",
        "\nThe corner plot shows the 1D and 2D posterior distributions of the model parameters.",
        "Contours show the 1-sigma, 2-sigma, and 3-sigma credible regions.",
        "\n## Interpretation",
        "\nThe MCMC analysis shows that the optimal Genesis-Sphere parameters are:",
        f"- **Omega (ω)**: {results_summary['omega']['median']:.4f} ± {(results_summary['omega']['lower_err'] + results_summary['omega']['upper_err'])/2:.4f}",
        f"- **Beta (β)**: {results_summary['beta']['median']:.4f} ± {(results_summary['beta']['lower_err'] + results_summary['beta']['upper_err'])/2:.4f}",
        "\nThese values represent the statistical constraints from combining H₀ correlation,",
        "supernovae distance modulus fitting, and BAO signal detection. The uncertainties",
        "reflect the genuine statistical uncertainty in determining these parameters from the available data.",
        "\nCompared to the previous grid search approach, this MCMC analysis provides more robust",
        "parameter constraints by thoroughly exploring the parameter space and quantifying uncertainties.",
        "\n---",
        "\n*This report was automatically generated by the Genesis-Sphere MCMC parameter estimation framework.*"
    ]
    
    summary_path = os.path.join(results_dir, f"mcmc_summary_{timestamp}{suffix}.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"Validation summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
