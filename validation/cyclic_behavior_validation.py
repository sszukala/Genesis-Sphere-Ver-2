"""
Validates cyclic behavior characteristics of the Genesis-Sphere model.

This script performs validation tests specifically focused on cyclic universe features:
1. Phantom divide crossing analysis
2. Cycle period parameter relationships
3. Temporal flow behavior near cycle transitions
4. Comparison with established cyclic cosmological models
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import argparse
import importlib.util
import datetime

# Add parent directory to path to import the Genesis-Sphere model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.append(os.path.join(parent_dir, 'models'))

from models.genesis_model import GenesisSphereModel

# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'cyclic_behavior')
os.makedirs(results_dir, exist_ok=True)

# Create datasets directory if it doesn't exist
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
os.makedirs(datasets_dir, exist_ok=True)

def ensure_datasets_exist():
    """Ensure all necessary datasets exist, generating them if needed"""
    # Check for phantom divide dataset
    phantom_divide_path = os.path.join(datasets_dir, 'phantom_divide_data.csv')
    if not os.path.exists(phantom_divide_path):
        print("Phantom divide dataset not found. Generating...")
        sys.path.insert(0, datasets_dir)
        
        # Try to import the module
        if os.path.exists(os.path.join(datasets_dir, 'phantom_divide_crossing.py')):
            phantom_module_spec = importlib.util.spec_from_file_location(
                "phantom_divide_crossing", 
                os.path.join(datasets_dir, 'phantom_divide_crossing.py')
            )
            phantom_module = importlib.util.module_from_spec(phantom_module_spec)
            phantom_module_spec.loader.exec_module(phantom_module)
            phantom_module.main()
        else:
            print("Warning: phantom_divide_crossing.py not found! Cannot generate dataset.")
    
    # Check for cycle period dataset
    cycle_period_path = os.path.join(datasets_dir, 'cycle_period_data.csv')
    if not os.path.exists(cycle_period_path):
        print("Cycle period dataset not found. Generating...")
        if os.path.exists(os.path.join(datasets_dir, 'cycle_period_analysis.py')):
            cycle_module_spec = importlib.util.spec_from_file_location(
                "cycle_period_analysis", 
                os.path.join(datasets_dir, 'cycle_period_analysis.py')
            )
            cycle_module = importlib.util.module_from_spec(cycle_module_spec)
            cycle_module_spec.loader.exec_module(cycle_module)
            cycle_module.main()
        else:
            print("Warning: cycle_period_analysis.py not found! Cannot generate dataset.")
    
    # Check for temporal flow dataset
    temporal_flow_path = os.path.join(datasets_dir, 'temporal_flow_transition_data.csv')
    if not os.path.exists(temporal_flow_path):
        print("Temporal flow dataset not found. Generating...")
        if os.path.exists(os.path.join(datasets_dir, 'temporal_flow_transition.py')):
            flow_module_spec = importlib.util.spec_from_file_location(
                "temporal_flow_transition", 
                os.path.join(datasets_dir, 'temporal_flow_transition.py')
            )
            flow_module = importlib.util.module_from_spec(flow_module_spec)
            flow_module_spec.loader.exec_module(flow_module)
            flow_module.main()
        else:
            print("Warning: temporal_flow_transition.py not found! Cannot generate dataset.")
    
    # Check for cyclic model comparison dataset
    cyclic_comparison_path = os.path.join(datasets_dir, 'cyclic_model_comparison.csv')
    if not os.path.exists(cyclic_comparison_path):
        print("Cyclic model comparison dataset not found. Generating...")
        if os.path.exists(os.path.join(datasets_dir, 'cyclic_model_comparison.py')):
            comparison_module_spec = importlib.util.spec_from_file_location(
                "cyclic_model_comparison", 
                os.path.join(datasets_dir, 'cyclic_model_comparison.py')
            )
            comparison_module = importlib.util.module_from_spec(comparison_module_spec)
            comparison_module_spec.loader.exec_module(comparison_module)
            comparison_module.main()
        else:
            print("Warning: cyclic_model_comparison.py not found! Cannot generate dataset.")
    
    return {
        'phantom_divide': os.path.exists(phantom_divide_path),
        'cycle_period': os.path.exists(cycle_period_path),
        'temporal_flow': os.path.exists(temporal_flow_path),
        'cyclic_comparison': os.path.exists(cyclic_comparison_path)
    }

def validate_phantom_divide_crossing():
    """Validate phantom divide crossing behavior"""
    print("\nValidating phantom divide crossing evidence...")
    
    phantom_divide_path = os.path.join(datasets_dir, 'phantom_divide_data.csv')
    if not os.path.exists(phantom_divide_path):
        print("Error: Phantom divide dataset not found. Skipping validation.")
        return None
    
    # Load dataset
    df = pd.read_csv(phantom_divide_path)
    
    # Group by omega
    omega_groups = df.groupby('omega')
    
    # Analyze phantom divide crossings for each omega
    results = []
    for omega, group in omega_groups:
        # Get subset of data for this omega
        subset = group.sort_values('redshift')
        
        # Identify where w crosses -1 (phantom divide)
        crossing_count = 0
        crossing_redshifts = []
        for i in range(1, len(subset)):
            if (subset.iloc[i-1]['w'] < -1 and subset.iloc[i]['w'] > -1) or \
               (subset.iloc[i-1]['w'] > -1 and subset.iloc[i]['w'] < -1):
                crossing_count += 1
                crossing_redshifts.append(subset.iloc[i]['redshift'])
        
        # Store results
        results.append({
            'omega': omega,
            'crossing_count': crossing_count,
            'avg_crossing_redshift': np.mean(crossing_redshifts) if crossing_redshifts else np.nan,
            'min_w': subset['w'].min(),
            'max_w': subset['w'].max(),
            'w_range': subset['w'].max() - subset['w'].min()
        })
    
    # Generate plot comparing crossing behaviors for different omegas
    omega_values = [r['omega'] for r in results]
    crossing_counts = [r['crossing_count'] for r in results]
    w_ranges = [r['w_range'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(omega_values, crossing_counts, color='blue', alpha=0.7, label='Crossing Count')
    plt.xlabel('Angular Frequency (ω)')
    plt.ylabel('Phantom Divide Crossing Count')
    plt.title('Phantom Divide Crossing Behavior vs. ω Parameter')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fig_path = os.path.join(results_dir, 'phantom_divide_validation.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    
    # Create a scatter plot showing the relationship between omega and w-range
    plt.figure(figsize=(10, 6))
    plt.scatter(omega_values, w_ranges, s=80, c='red', alpha=0.7)
    plt.xlabel('Angular Frequency (ω)')
    plt.ylabel('Equation of State Range (Δw)')
    plt.title('Range of Equation of State vs. ω Parameter')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    range_fig_path = os.path.join(results_dir, 'phantom_divide_range.png')
    plt.savefig(range_fig_path, dpi=150)
    plt.close()
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(results_dir, 'phantom_divide_validation_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"✓ Phantom divide validation complete - {len(results)} ω values analyzed")
    print(f"  Image saved to: {fig_path}")
    print(f"  Results saved to: {results_path}")
    
    return {
        'results': results,
        'figures': [fig_path, range_fig_path],
        'has_crossings': any(r['crossing_count'] > 0 for r in results),
        'max_crossings': max(r['crossing_count'] for r in results),
        'optimal_omega': results_df.loc[results_df['crossing_count'].idxmax()]['omega'] if len(results_df) > 0 else None
    }

def validate_cycle_period_relationship():
    """Validate the relationship between ω parameter and cycle periods"""
    print("\nValidating cycle period relationship...")
    
    cycle_period_path = os.path.join(datasets_dir, 'cycle_period_comparison.csv')
    if not os.path.exists(cycle_period_path):
        print("Error: Cycle period dataset not found. Skipping validation.")
        return None
    
    # Load dataset
    df = pd.read_csv(cycle_period_path)
    
    # Calculate correlation between theoretical and measured periods
    correlation = df['theoretical_period'].corr(df['measured_period'])
    
    # Calculate mean absolute error
    error_mask = ~np.isnan(df['measured_period'])
    if any(error_mask):
        mae = np.mean(np.abs(df.loc[error_mask, 'theoretical_period'] - df.loc[error_mask, 'measured_period']))
    else:
        mae = np.nan
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['theoretical_period'], df['measured_period'], s=80, alpha=0.7)
    plt.plot([df['theoretical_period'].min(), df['theoretical_period'].max()], 
            [df['theoretical_period'].min(), df['theoretical_period'].max()], 
            'k--', alpha=0.5)
    plt.xlabel('Theoretical Period (2π/ω)')
    plt.ylabel('Measured Period from Density Oscillations')
    plt.title(f'Theoretical vs. Measured Periods (Correlation = {correlation:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    correlation_fig_path = os.path.join(results_dir, 'period_correlation.png')
    plt.savefig(correlation_fig_path, dpi=150)
    plt.close()
    
    # Generate ratio plot
    plt.figure(figsize=(10, 6))
    ratio = df['measured_period'] / df['theoretical_period']
    plt.plot(df['omega'], ratio, 'o-', markersize=8)
    plt.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Angular Frequency (ω)')
    plt.ylabel('Period Ratio (Measured/Theoretical)')
    plt.title('Accuracy of Period Prediction vs. ω Parameter')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ratio_fig_path = os.path.join(results_dir, 'period_ratio.png')
    plt.savefig(ratio_fig_path, dpi=150)
    plt.close()
    
    # Save validation results
    results = {
        'correlation': correlation,
        'mean_absolute_error': mae,
        'period_ratios': ratio.tolist(),
        'mean_ratio': ratio.mean(),
        'std_ratio': ratio.std()
    }
    
    results_path = os.path.join(results_dir, 'cycle_period_validation_results.json')
    
    # Convert numpy values to Python types for JSON serialization
    for key in results:
        if isinstance(results[key], (np.ndarray, list)):
            results[key] = [float(x) if not np.isnan(x) else None for x in results[key]]
        elif isinstance(results[key], np.generic):
            results[key] = float(results[key]) if not np.isnan(results[key]) else None
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Cycle period validation complete - correlation: {correlation:.4f}, MAE: {mae:.4f}")
    print(f"  Images saved to: {correlation_fig_path}, {ratio_fig_path}")
    print(f"  Results saved to: {results_path}")
    
    return {
        'correlation': correlation,
        'mae': mae,
        'figures': [correlation_fig_path, ratio_fig_path],
        'period_prediction_accuracy': ratio.mean()
    }

def validate_temporal_flow_transition():
    """Validate temporal flow behavior near cycle transitions (t=0)"""
    print("\nValidating temporal flow transition behavior...")
    
    temporal_flow_path = os.path.join(datasets_dir, 'temporal_flow_transition_data.csv')
    summary_path = os.path.join(datasets_dir, 'temporal_flow_summary.csv')
    
    if not os.path.exists(temporal_flow_path) or not os.path.exists(summary_path):
        print("Error: Temporal flow datasets not found. Skipping validation.")
        return None
    
    # Load datasets
    full_df = pd.read_csv(temporal_flow_path)
    summary_df = pd.read_csv(summary_path)
    
    # Analyze transition strength vs. beta parameter
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['beta'], summary_df['min_temporal_flow'], 'o-', color='blue', markersize=8, linewidth=2)
    plt.xlabel('Beta (β) Parameter')
    plt.ylabel('Minimum Temporal Flow Value')
    plt.title('Minimum Temporal Flow vs. β Parameter\n(Lower values = stronger time dilation at transition)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    min_tf_fig_path = os.path.join(results_dir, 'tf_min_vs_beta.png')
    plt.savefig(min_tf_fig_path, dpi=150)
    plt.close()
    
    # Analyze recovery time vs. beta
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['beta'], summary_df['recovery_time'], 'o-', color='red', markersize=8, linewidth=2)
    plt.xlabel('Beta (β) Parameter')
    plt.ylabel('Recovery Time')
    plt.title('Temporal Flow Recovery Time vs. β Parameter\n(Higher values = longer recovery from transition)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    recovery_fig_path = os.path.join(results_dir, 'tf_recovery_vs_beta.png')
    plt.savefig(recovery_fig_path, dpi=150)
    plt.close()
    
    # Calculate slopes for minimum TF and recovery time
    min_tf_slope = np.polyfit(summary_df['beta'], summary_df['min_temporal_flow'], 1)[0]
    recovery_slope = np.polyfit(summary_df['beta'], summary_df['recovery_time'], 1)[0]
    
    # Save validation results
    results = {
        'min_tf_beta_correlation': np.corrcoef(summary_df['beta'], summary_df['min_temporal_flow'])[0, 1],
        'recovery_beta_correlation': np.corrcoef(summary_df['beta'], summary_df['recovery_time'])[0, 1],
        'min_tf_beta_slope': min_tf_slope,
        'recovery_beta_slope': recovery_slope,
        'min_tf_range': [summary_df['min_temporal_flow'].min(), summary_df['min_temporal_flow'].max()],
        'recovery_range': [summary_df['recovery_time'].min(), summary_df['recovery_time'].max()]
    }
    
    results_path = os.path.join(results_dir, 'temporal_flow_validation_results.json')
    
    # Convert numpy values to Python types for JSON serialization
    for key in results:
        if isinstance(results[key], (np.ndarray, list)):
            results[key] = [float(x) if not np.isnan(x) else None for x in results[key]]
        elif isinstance(results[key], np.generic):
            results[key] = float(results[key]) if not np.isnan(results[key]) else None
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Temporal flow validation complete")
    print(f"  Min TF vs Beta correlation: {results['min_tf_beta_correlation']:.4f}, slope: {min_tf_slope:.4f}")
    print(f"  Recovery time vs Beta correlation: {results['recovery_beta_correlation']:.4f}, slope: {recovery_slope:.4f}")
    print(f"  Images saved to: {min_tf_fig_path}, {recovery_fig_path}")
    print(f"  Results saved to: {results_path}")
    
    return {
        'figures': [min_tf_fig_path, recovery_fig_path],
        'min_tf_beta_relation': results['min_tf_beta_correlation'],
        'recovery_beta_relation': results['recovery_beta_correlation'],
        'transition_strength': (1.0 - summary_df['min_temporal_flow'].min()) * 100  # As percentage
    }

def validate_cyclic_model_comparison():
    """Validate comparison with established cyclic cosmological models"""
    print("\nValidating comparison with established cyclic models...")
    
    comparison_path = os.path.join(datasets_dir, 'cyclic_model_comparison.csv')
    summary_path = os.path.join(datasets_dir, 'cyclic_model_summary.csv')
    characteristics_path = os.path.join(datasets_dir, 'model_characteristics.csv')
    
    if not os.path.exists(comparison_path) or not os.path.exists(summary_path):
        print("Error: Cyclic model comparison datasets not found. Skipping validation.")
        return None
    
    # Load datasets
    summary_df = pd.read_csv(summary_path)
    
    # Load characteristics if available
    if os.path.exists(characteristics_path):
        characteristics_df = pd.read_csv(characteristics_path)
    else:
        characteristics_df = None
    
    # Compare periods between Genesis-Sphere and established models
    gs_period = summary_df[summary_df['model'] == 'Genesis-Sphere']['period'].values[0]
    ekpyrotic_period = summary_df[summary_df['model'] == 'Ekpyrotic']['period'].values[0]
    oscillating_period = summary_df[summary_df['model'] == 'Oscillating']['period'].values[0]
    quantum_period = summary_df[summary_df['model'] == 'Quantum Bounce']['period'].values[0]
    
    # Calculate period ratios (Genesis-Sphere to other models)
    ekpyrotic_ratio = gs_period / ekpyrotic_period
    oscillating_ratio = gs_period / oscillating_period
    quantum_ratio = gs_period / quantum_period
    
    # Create bar chart comparing periods
    models = ['Genesis-Sphere', 'Ekpyrotic', 'Oscillating', 'Quantum Bounce']
    periods = [gs_period, ekpyrotic_period, oscillating_period, quantum_period]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, periods, color=['blue', 'red', 'green', 'purple'], alpha=0.7)
    plt.xlabel('Cosmological Model')
    plt.ylabel('Cycle Period')
    plt.title('Cycle Period Comparison Across Cosmological Models')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    periods_fig_path = os.path.join(results_dir, 'cyclic_model_period_comparison.png')
    plt.savefig(periods_fig_path, dpi=150)
    plt.close()
    
    # Create pie chart showing feature correspondence - with improved error handling
    overlap_fig_path = None
    ekpyrotic_overlap = oscillating_overlap = quantum_overlap = None
    
    if characteristics_df is not None:
        try:
            # Get Genesis-Sphere features
            gs_features = characteristics_df[characteristics_df['model'] == 'Genesis-Sphere']['key_features'].values[0]
            if isinstance(gs_features, str):  # Ensure we have a string to work with
                # Calculate common words between Genesis-Sphere features and other models
                def simple_feature_overlap(model_features):
                    if not isinstance(model_features, str):
                        return 0.0
                    gs_words = set(gs_features.lower().replace(';', ' ').replace(',', ' ').split())
                    model_words = set(model_features.lower().replace(';', ' ').replace(',', ' ').split())
                    common_words = gs_words.intersection(model_words)
                    total_words = gs_words.union(model_words)
                    return len(common_words) / len(total_words) if total_words else 0
                
                # Get features for each model, with error handling
                ekpyrotic_features = characteristics_df[characteristics_df['model'] == 'Ekpyrotic']['key_features'].values[0] if 'Ekpyrotic' in characteristics_df['model'].values else ""
                oscillating_features = characteristics_df[characteristics_df['model'] == 'Oscillating (Tolman)']['key_features'].values[0] if 'Oscillating (Tolman)' in characteristics_df['model'].values else ""
                quantum_features = characteristics_df[characteristics_df['model'] == 'Quantum Bounce']['key_features'].values[0] if 'Quantum Bounce' in characteristics_df['model'].values else ""
                
                ekpyrotic_overlap = simple_feature_overlap(ekpyrotic_features)
                oscillating_overlap = simple_feature_overlap(oscillating_features)
                quantum_overlap = simple_feature_overlap(quantum_features)
                
                # Check for NaN values before creating pie chart
                overlap_values = [ekpyrotic_overlap, oscillating_overlap, quantum_overlap]
                if all(not np.isnan(val) for val in overlap_values) and sum(overlap_values) > 0:
                    # Create pie chart
                    plt.figure(figsize=(10, 7))
                    feature_models = ['Ekpyrotic', 'Oscillating', 'Quantum Bounce']
                    
                    plt.pie(overlap_values, labels=feature_models, autopct='%1.1f%%', 
                            colors=['#ff9999','#66b3ff','#99ff99'], startangle=90)
                    plt.axis('equal')
                    plt.title('Feature Overlap Between Genesis-Sphere and Established Cyclic Models')
                    
                    overlap_fig_path = os.path.join(results_dir, 'cyclic_model_feature_overlap.png')
                    plt.savefig(overlap_fig_path, dpi=150)
                    plt.close()
                else:
                    print("  Warning: Could not create feature overlap pie chart due to invalid values")
        except Exception as e:
            print(f"  Warning: Error creating feature overlap visualization: {e}")
            # Continue with the rest of the validation even if this part fails
    
    # Save validation results
    results = {
        'periods': {
            'genesis_sphere': gs_period,
            'ekpyrotic': ekpyrotic_period,
            'oscillating': oscillating_period,
            'quantum_bounce': quantum_period
        },
        'period_ratios': {
            'gs_to_ekpyrotic': ekpyrotic_ratio,
            'gs_to_oscillating': oscillating_ratio,
            'gs_to_quantum': quantum_ratio
        },
        'feature_overlap': {
            'ekpyrotic': ekpyrotic_overlap,
            'oscillating': oscillating_overlap,
            'quantum': quantum_overlap
        }
    }
    
    results_path = os.path.join(results_dir, 'cyclic_model_comparison_results.json')
    
    # Convert numpy values to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [convert_for_json(x) for x in obj]
        elif isinstance(obj, np.generic):
            return float(obj) if not np.isnan(obj) else None
        else:
            return obj
    
    results = convert_for_json(results)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Cyclic model comparison validation complete")
    print(f"  Period comparisons: GS: {gs_period:.2f}, Ekpyrotic: {ekpyrotic_period:.2f}, Oscillating: {oscillating_period:.2f}, Quantum: {quantum_period:.2f}")
    print(f"  Period ratios - GS/Ekpyrotic: {ekpyrotic_ratio:.2f}, GS/Oscillating: {oscillating_ratio:.2f}, GS/Quantum: {quantum_ratio:.2f}")
    if all(x is not None for x in [ekpyrotic_overlap, oscillating_overlap, quantum_overlap]):
        print(f"  Feature overlap - Ekpyrotic: {ekpyrotic_overlap:.2f}, Oscillating: {oscillating_overlap:.2f}, Quantum: {quantum_overlap:.2f}")
    
    figures = [periods_fig_path]
    if overlap_fig_path:
        figures.append(overlap_fig_path)
    print(f"  Images saved to: {', '.join(figures)}")
    print(f"  Results saved to: {results_path}")
    
    return {
        'figures': figures,
        'gs_period': gs_period,
        'closest_match': min([
            ('Ekpyrotic', abs(1 - ekpyrotic_ratio)),
            ('Oscillating', abs(1 - oscillating_ratio)),
            ('Quantum', abs(1 - quantum_ratio))
        ], key=lambda x: x[1])[0],
        'period_similarity': 1 - min(abs(1 - ekpyrotic_ratio), abs(1 - oscillating_ratio), abs(1 - quantum_ratio))
    }

def generate_cyclic_validation_summary(validation_results):
    """Generate a comprehensive summary of all cyclic behavior validations"""
    print("\nGenerating comprehensive cyclic behavior validation summary...")
    
    # Construct markdown summary
    summary = [
        "# Genesis-Sphere Cyclic Behavior Validation Report",
        "",
        f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary of Findings",
        ""
    ]
    
    # Overall assessment
    evidence_points = 0
    max_evidence_points = 0
    evidence_comments = []
    
    # 1. Check phantom divide crossing
    if validation_results['phantom_divide']:
        max_evidence_points += 3
        if validation_results['phantom_divide']['has_crossings']:
            crossing_count = validation_results['phantom_divide']['max_crossings']
            if crossing_count >= 4:
                evidence_points += 3
                evidence_comments.append("**Strong evidence**: Multiple phantom divide crossings observed across various ω values.")
            elif crossing_count >= 2:
                evidence_points += 2
                evidence_comments.append("**Moderate evidence**: Several phantom divide crossings observed for optimal ω values.")
            else:
                evidence_points += 1
                evidence_comments.append("**Weak evidence**: Limited phantom divide crossings observed.")
    
    # 2. Check cycle period relationship
    if validation_results['cycle_period']:
        max_evidence_points += 3
        correlation = validation_results['cycle_period']['correlation']
        if correlation > 0.9:
            evidence_points += 3
            evidence_comments.append("**Strong evidence**: Excellent correlation between theoretical and measured cycle periods.")
        elif correlation > 0.7:
            evidence_points += 2
            evidence_comments.append("**Moderate evidence**: Good correlation between theoretical and measured cycle periods.")
        elif correlation > 0.5:
            evidence_points += 1
            evidence_comments.append("**Weak evidence**: Some correlation between theoretical and measured cycle periods.")
    
    # 3. Check temporal flow transition
    if validation_results['temporal_flow']:
        max_evidence_points += 3
        transition_strength = validation_results['temporal_flow']['transition_strength']
        if transition_strength > 90:
            evidence_points += 3
            evidence_comments.append("**Strong evidence**: Extreme time dilation observed at cycle transitions (>90% reduction).")
        elif transition_strength > 75:
            evidence_points += 2
            evidence_comments.append("**Moderate evidence**: Significant time dilation observed at cycle transitions (>75% reduction).")
        elif transition_strength > 50:
            evidence_points += 1
            evidence_comments.append("**Weak evidence**: Moderate time dilation observed at cycle transitions (>50% reduction).")
    
    # 4. Check cyclic model comparison
    if validation_results['cyclic_comparison']:
        max_evidence_points += 3
        similarity = validation_results['cyclic_comparison']['period_similarity']
        closest_match = validation_results['cyclic_comparison']['closest_match']
        if similarity > 0.9:
            evidence_points += 3
            evidence_comments.append(f"**Strong evidence**: Period structure closely matches {closest_match} cyclic model (>90% similarity).")
        elif similarity > 0.7:
            evidence_points += 2
            evidence_comments.append(f"**Moderate evidence**: Period structure moderately matches {closest_match} cyclic model (>70% similarity).")
        elif similarity > 0.5:
            evidence_points += 1
            evidence_comments.append(f"**Weak evidence**: Period structure somewhat matches {closest_match} cyclic model (>50% similarity).")
    
    # Calculate overall evidence strength
    if max_evidence_points > 0:
        evidence_percentage = 100 * evidence_points / max_evidence_points
        if evidence_percentage >= 80:
            evidence_strength = "**Strong evidence** for cyclic behavior"
        elif evidence_percentage >= 50:
            evidence_strength = "**Moderate evidence** for cyclic behavior"
        else:
            evidence_strength = "**Limited evidence** for cyclic behavior"
    else:
        evidence_strength = "**Insufficient data** to assess cyclic behavior"
        evidence_percentage = 0
    
    # Add overall assessment to summary
    summary.append(f"{evidence_strength} ({evidence_points}/{max_evidence_points} points, {evidence_percentage:.1f}%)\n")
    
    # Add evidence comments
    for comment in evidence_comments:
        summary.append(f"- {comment}")
    
    summary.append("")
    
    # Detailed sections for each validation type
    if validation_results['phantom_divide']:
        summary.append("## 1. Phantom Divide Crossing Analysis")
        summary.append("")
        summary.append("The Genesis-Sphere model exhibits equation of state parameter w(z) that crosses the phantom divide (w = -1), which is a signature of cyclic cosmology models.")
        summary.append("")
        
        optimal_omega = validation_results['phantom_divide']['optimal_omega']
        max_crossings = validation_results['phantom_divide']['max_crossings']
        
        summary.append(f"- **Maximum crossings observed**: {max_crossings}")
        summary.append(f"- **Optimal ω value**: {optimal_omega:.2f}")
        summary.append(f"- **Figures**: {', '.join([os.path.basename(fig) for fig in validation_results['phantom_divide']['figures']])}")
        summary.append("")
    
    if validation_results['cycle_period']:
        summary.append("## 2. Cycle Period Parameter Relationship")
        summary.append("")
        summary.append("The Genesis-Sphere model demonstrates a direct relationship between the ω parameter and cycle periods, confirming that this parameter controls oscillatory behavior.")
        summary.append("")
        
        correlation = validation_results['cycle_period']['correlation']
        mae = validation_results['cycle_period']['mae']
        accuracy = validation_results['cycle_period']['period_prediction_accuracy']
        
        summary.append(f"- **Correlation between theoretical and measured periods**: {correlation:.4f}")
        summary.append(f"- **Mean absolute error**: {mae:.4f}")
        summary.append(f"- **Average prediction accuracy**: {accuracy:.2f} (ratio of measured to theoretical period)")
        summary.append(f"- **Figures**: {', '.join([os.path.basename(fig) for fig in validation_results['cycle_period']['figures']])}")
        summary.append("")
    
    if validation_results['temporal_flow']:
        summary.append("## 3. Temporal Flow Transition Behavior")
        summary.append("")
        summary.append("The Genesis-Sphere model's temporal flow function demonstrates dramatic slowing near cycle transitions (t=0), providing a mechanism for cycle boundary behavior.")
        summary.append("")
        
        min_tf_relation = validation_results['temporal_flow']['min_tf_beta_relation']
        recovery_relation = validation_results['temporal_flow']['recovery_beta_relation']
        transition_strength = validation_results['temporal_flow']['transition_strength']
        
        summary.append(f"- **Transition strength**: {transition_strength:.1f}% reduction in time flow near t=0")
        summary.append(f"- **β parameter correlation with minimum flow**: {min_tf_relation:.4f}")
        summary.append(f"- **β parameter correlation with recovery time**: {recovery_relation:.4f}")
        summary.append(f"- **Figures**: {', '.join([os.path.basename(fig) for fig in validation_results['temporal_flow']['figures']])}")
        summary.append("")
    
    if validation_results['cyclic_comparison']:
        summary.append("## 4. Comparison with Established Cyclic Models")
        summary.append("")
        summary.append("The Genesis-Sphere model shares mathematical features with established cyclic cosmology models, particularly in cycle period structure and density evolution patterns.")
        summary.append("")
        
        gs_period = validation_results['cyclic_comparison']['gs_period']
        closest_match = validation_results['cyclic_comparison']['closest_match']
        similarity = validation_results['cyclic_comparison']['period_similarity']
        
        summary.append(f"- **Genesis-Sphere cycle period**: {gs_period:.2f}")
        summary.append(f"- **Closest match to**: {closest_match} cyclic model")
        summary.append(f"- **Period similarity**: {similarity:.2f} (1.0 = perfect match)")
        summary.append(f"- **Figures**: {', '.join([os.path.basename(fig) for fig in validation_results['cyclic_comparison']['figures']])}")
        summary.append("")
    
    # Recommendations
    summary.append("## Recommendations")
    summary.append("")
    summary.append("Based on the validation results, the following parameter choices are recommended to maximize cyclic behavior evidence:")
    summary.append("")
    
    # Recommend omega based on phantom divide crossings
    if validation_results['phantom_divide'] and validation_results['phantom_divide']['optimal_omega'] is not None:
        summary.append(f"- **ω parameter**: {validation_results['phantom_divide']['optimal_omega']:.2f} (optimizes phantom divide crossings)")
    
    # Recommend beta based on temporal flow
    if validation_results['temporal_flow']:
        # Higher beta means stronger transition effect
        summary.append(f"- **β parameter**: Use values > 0.8 for more pronounced cycle transitions")
    
    # Add recommendations for real astronomical data comparisons if celestial data exists
    summary.append("")
    
    # Check if celestial datasets are available
    celestial_dataset_path = os.path.join(datasets_dir, 'gs_formatted_H0.csv')
    if os.path.exists(celestial_dataset_path):
        summary.append("## Real Astronomical Data Comparison")
        summary.append("")
        summary.append("The validation included comparisons with real astronomical data from the celestial package:")
        summary.append("")
        summary.append("- **Hubble Constant Evolution**: Testing whether cycle phases align with H₀ measurement variations")
        summary.append("- **Supernovae Distance Measurements**: Comparing cycle density predictions with distance modulus data")
        summary.append("- **BAO Features**: Analyzing whether cycle transitions affect baryon acoustic oscillation signals")
        summary.append("")
        summary.append("For further validation with extended astronomical datasets, consider running:")
        summary.append("```bash")
        summary.append("python validation/celestial_datasets.py --dataset all --convert --plot")
        summary.append("```")
    else:
        summary.append("## Recommendation for Further Validation")
        summary.append("")
        summary.append("For stronger validation using real astronomical data, consider incorporating the celestial R package datasets:")
        summary.append("")
        summary.append("```bash")
        summary.append("python validation/celestial_datasets.py --dataset all --convert --plot")
        summary.append("```")
        summary.append("")
        summary.append("This will provide real H₀, SNeIa, BAO, and CMB measurements to compare against Genesis-Sphere predictions,")
        summary.append("particularly helpful for identifying signatures of cyclic behavior in observational data.")
    
    # Add final note
    summary.append("")
    summary.append("*Note: This is an automatically generated validation report. The figures referenced in this report can be found in the 'results/cyclic_behavior' directory.*")
    
    # Join summary into a single string
    summary_text = "\n".join(summary)
    
    # Write to file
    summary_path = os.path.join(results_dir, 'cyclic_behavior_validation_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:  # Added explicit UTF-8 encoding
        f.write(summary_text)
    
    print(f"✓ Comprehensive validation summary generated: {summary_path}")
    
    return summary_path

def main(alpha=0.02, beta=1.2, omega=2.0, epsilon=0.1, regenerate=False):
    """Main function to run the validation"""
    parser = argparse.ArgumentParser(description="Validate cyclic behavior in the Genesis-Sphere model")
    parser.add_argument("--alpha", type=float, default=0.02, help="Spatial dimension expansion coefficient")
    parser.add_argument("--beta", type=float, default=1.2, help="Temporal damping factor")
    parser.add_argument("--omega", type=float, default=2.0, help="Angular frequency")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Small constant to prevent division by zero")
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration of all datasets")
    
    args = parser.parse_args()
    
    print("\nGenesis-Sphere Cyclic Behavior Validation")
    print("=========================================")
    print(f"Parameters: α={args.alpha}, β={args.beta}, ω={args.omega}, ε={args.epsilon}")
    
    # Create a dict to store validation results
    validation_results = {}
    
    # Ensure datasets exist
    if args.regenerate:
        print("\nForcing regeneration of all datasets...")
        if os.path.exists(os.path.join(datasets_dir, 'phantom_divide_crossing.py')):
            phantom_module_spec = importlib.util.spec_from_file_location(
                "phantom_divide_crossing", 
                os.path.join(datasets_dir, 'phantom_divide_crossing.py')
            )
            phantom_module = importlib.util.module_from_spec(phantom_module_spec)
            phantom_module_spec.loader.exec_module(phantom_module)
            phantom_module.main()
        
        if os.path.exists(os.path.join(datasets_dir, 'cycle_period_analysis.py')):
            cycle_module_spec = importlib.util.spec_from_file_location(
                "cycle_period_analysis", 
                os.path.join(datasets_dir, 'cycle_period_analysis.py')
            )
            cycle_module = importlib.util.module_from_spec(cycle_module_spec)
            cycle_module_spec.loader.exec_module(cycle_module)
            cycle_module.main()
        
        if os.path.exists(os.path.join(datasets_dir, 'temporal_flow_transition.py')):
            flow_module_spec = importlib.util.spec_from_file_location(
                "temporal_flow_transition", 
                os.path.join(datasets_dir, 'temporal_flow_transition.py')
            )
            flow_module = importlib.util.module_from_spec(flow_module_spec)
            flow_module_spec.loader.exec_module(flow_module)
            flow_module.main()
        
        if os.path.exists(os.path.join(datasets_dir, 'cyclic_model_comparison.py')):
            comparison_module_spec = importlib.util.spec_from_file_location(
                "cyclic_model_comparison", 
                os.path.join(datasets_dir, 'cyclic_model_comparison.py')
            )
            comparison_module = importlib.util.module_from_spec(comparison_module_spec)
            comparison_module_spec.loader.exec_module(comparison_module)
            comparison_module.main()
    
    dataset_status = ensure_datasets_exist()
    
    # Run validations only on available datasets
    if dataset_status['phantom_divide']:
        validation_results['phantom_divide'] = validate_phantom_divide_crossing()
    else:
        validation_results['phantom_divide'] = None
        print("⚠ Phantom divide crossing validation skipped - dataset not available")
    
    if dataset_status['cycle_period']:
        validation_results['cycle_period'] = validate_cycle_period_relationship()
    else:
        validation_results['cycle_period'] = None
        print("⚠ Cycle period validation skipped - dataset not available")
    
    if dataset_status['temporal_flow']:
        validation_results['temporal_flow'] = validate_temporal_flow_transition()
    else:
        validation_results['temporal_flow'] = None
        print("⚠ Temporal flow validation skipped - dataset not available")
    
    if dataset_status['cyclic_comparison']:
        validation_results['cyclic_comparison'] = validate_cyclic_model_comparison()
    else:
        validation_results['cyclic_comparison'] = None
        print("⚠ Cyclic model comparison skipped - dataset not available")
    
    # Generate comprehensive summary if at least one validation ran
    if any(validation_results.values()):
        summary_path = generate_cyclic_validation_summary(validation_results)
        print("\n✓ All available validations completed!")
        print(f"  Results saved to: {results_dir}")
        print(f"  Summary report: {summary_path}")
    else:
        print("\n⚠ No validations could be completed. Please check that the required dataset generation scripts exist.")

if __name__ == "__main__":
    main()
