"""
Static Simulation of Genesis-Sphere Model

This script demonstrates how to use the Genesis-Sphere computational model
for static simulations and parameter exploration.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from genesis_model import GenesisSphereModel
import json


def parameter_sensitivity_analysis(save_dir='../output/model_analysis'):
    """
    Perform parameter sensitivity analysis for the Genesis-Sphere model.
    
    This function examines how changes in each parameter affect the model's
    behavior by varying one parameter at a time and plotting the results.
    
    Parameters:
    -----------
    save_dir : str
        Directory where analysis results will be saved
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Common time domain
    t = np.linspace(-12, 12, 1000)
    
    # Create baseline model
    model = GenesisSphereModel()
    baseline = model.evaluate_all(t)
    
    # Parameters to vary
    params = {
        'alpha': [0.01, 0.02, 0.05, 0.1],  # Dimension expansion
        'beta': [0.4, 0.8, 1.2, 1.6],      # Temporal damping
        'omega': [0.5, 1.0, 1.5, 2.0],     # Angular frequency
        'epsilon': [0.05, 0.1, 0.2, 0.5]   # Zero-prevention constant
    }
    
    # Functions to analyze
    functions = ['density', 'temporal_flow', 'velocity', 'pressure']
    
    # Run sensitivity analysis for each parameter
    for param_name, param_values in params.items():
        print(f"Analyzing sensitivity to {param_name}...")
        
        # Initialize a figure with subplots for each function
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Run simulation for each parameter value - explicitly pass num_points to match t
        results = model.simulate_parameter_variation(param_name, param_values, 
                                                    t_range=(-12, 12), num_points=1000)
        
        # Plot results for each function
        for i, func_name in enumerate(functions):
            ax = axes[i]
            
            # Plot baseline
            ax.plot(t, baseline[func_name], 'k--', label=f"Baseline")
            
            # Plot variations
            for value, result in results.items():
                ax.plot(t, result[func_name], label=f"{param_name}={value}")
            
            ax.set_title(f"Effect of {param_name} on {func_name}")
            ax.set_xlabel("Time (t)")
            ax.set_ylabel(f"{func_name.capitalize()}")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sensitivity_{param_name}.png"), dpi=300)
        plt.close()
    
    print(f"Sensitivity analysis completed. Results saved to {save_dir}")


def scenario_simulation(scenario_name, params, save_dir='../output/model_scenarios'):
    """
    Run a simulation for a specific named scenario with custom parameters.
    
    Parameters:
    -----------
    scenario_name : str
        Name of the scenario
    params : dict
        Dictionary of parameters for the model
    save_dir : str
        Directory where scenario results will be saved
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create model with scenario parameters
    model = GenesisSphereModel(**params)
    
    # Run and save full plot
    fig = model.plot_all()
    plt.suptitle(f"Scenario: {scenario_name}", fontsize=16)
    
    # Save figure
    plt.savefig(os.path.join(save_dir, f"scenario_{scenario_name.lower().replace(' ', '_')}.png"), dpi=300)
    
    # Save parameters as JSON
    with open(os.path.join(save_dir, f"params_{scenario_name.lower().replace(' ', '_')}.json"), 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Scenario '{scenario_name}' simulated and saved.")
    plt.close()
    
    return model


def main():
    """Run standard static simulations"""
    os.makedirs("../output/model_scenarios", exist_ok=True)
    
    print("Genesis-Sphere Static Simulation")
    print("================================")
    
    # Run parameter sensitivity analysis
    parameter_sensitivity_analysis()
    
    # Define and run specific scenarios
    scenarios = {
        "Standard Universe": {
            "alpha": 0.02, 
            "beta": 0.8, 
            "omega": 1.0, 
            "epsilon": 0.1
        },
        "High Density Oscillation": {
            "alpha": 0.01, 
            "beta": 0.7, 
            "omega": 2.0, 
            "epsilon": 0.1
        },
        "Extreme Time Dilation": {
            "alpha": 0.02, 
            "beta": 2.0, 
            "omega": 1.0, 
            "epsilon": 0.05
        },
        "Rapid Expansion": {
            "alpha": 0.1, 
            "beta": 0.8, 
            "omega": 0.5, 
            "epsilon": 0.1
        }
    }
    
    # Run each scenario
    for name, params in scenarios.items():
        scenario_simulation(name, params)
    
    print("All static simulations completed!")


if __name__ == "__main__":
    main()
