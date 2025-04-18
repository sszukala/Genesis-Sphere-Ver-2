"""
Genesis-Sphere Computational Model

This module implements the core mathematical model of the Genesis-Sphere framework,
encapsulating all equations in a reusable, parameterizable class structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List


class GenesisSphereModel:
    """
    Computational model of the Genesis-Sphere framework, implementing the core
    mathematical functions for space-time density and temporal flow.
    """
    
    def __init__(self, alpha: float = 0.02, beta: float = 0.8, 
                 omega: float = 1.0, epsilon: float = 0.1):
        """
        Initialize the Genesis-Sphere model with customizable parameters.
        
        Parameters:
        -----------
        alpha : float
            Spatial dimension expansion coefficient (default: 0.02)
        beta : float
            Temporal damping factor (default: 0.8)
        omega : float
            Angular frequency for sinusoidal projections (default: 1.0)
        epsilon : float
            Small constant to prevent division by zero (default: 0.1)
        """
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.epsilon = epsilon
    
    def projection_factor(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate the sinusoidal projection factor S(t).
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
            
        Returns:
        --------
        numpy.ndarray
            Sinusoidal projection factor values
        """
        return 1 / (1 + np.sin(self.omega * t)**2)
    
    def dimension_expansion(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate the dimension expansion factor D(t).
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
            
        Returns:
        --------
        numpy.ndarray
            Dimension expansion factor values
        """
        return 1 + self.alpha * t**2
    
    def rho(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate the time-density geometry function ρ(t).
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
            
        Returns:
        --------
        numpy.ndarray
            Time-density values
        """
        S = self.projection_factor(t)
        D = self.dimension_expansion(t)
        return S * D
    
    def tf(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate the temporal flow ratio function Tf(t).
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
            
        Returns:
        --------
        numpy.ndarray
            Temporal flow values
        """
        return 1 / (1 + self.beta * (np.abs(t) + self.epsilon))
    
    def velocity(self, t: np.ndarray, v0: float = 1.0) -> np.ndarray:
        """
        Calculate the modulated velocity v(t).
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
        v0 : float
            Initial unmodulated velocity (default: 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Modulated velocity values
        """
        return v0 * self.tf(t)
    
    def pressure(self, t: np.ndarray, p0: float = 1.0) -> np.ndarray:
        """
        Calculate the modulated pressure p(t).
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
        p0 : float
            Initial unmodulated pressure (default: 1.0)
            
        Returns:
        --------
        numpy.ndarray
            Modulated pressure values
        """
        return p0 * self.rho(t)
    
    def evaluate_all(self, t: np.ndarray, v0: float = 1.0, 
                     p0: float = 1.0) -> dict:
        """
        Evaluate all core functions for the given time values.
        
        Parameters:
        -----------
        t : numpy.ndarray
            Time values
        v0 : float
            Initial unmodulated velocity (default: 1.0)
        p0 : float
            Initial unmodulated pressure (default: 1.0)
            
        Returns:
        --------
        dict
            Dictionary containing all calculated values
        """
        S = self.projection_factor(t)
        D = self.dimension_expansion(t)
        rho = S * D
        tf = self.tf(t)
        velocity = v0 * tf
        pressure = p0 * rho
        
        return {
            'time': t,
            'projection': S,
            'expansion': D,
            'density': rho,
            'temporal_flow': tf,
            'velocity': velocity,
            'pressure': pressure
        }
    
    def plot_all(self, t_range: Tuple[float, float] = (-12, 12), 
                 num_points: int = 1000, v0: float = 1.0, 
                 p0: float = 1.0, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Generate a comprehensive plot of all Genesis-Sphere functions.
        
        Parameters:
        -----------
        t_range : tuple of float
            Time range (min, max) for the plot (default: (-12, 12))
        num_points : int
            Number of points to calculate (default: 1000)
        v0 : float
            Initial unmodulated velocity (default: 1.0)
        p0 : float
            Initial unmodulated pressure (default: 1.0)
        figsize : tuple of int
            Figure size (width, height) in inches (default: (12, 8))
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        t = np.linspace(t_range[0], t_range[1], num_points)
        results = self.evaluate_all(t, v0, p0)
        
        fig = plt.figure(figsize=figsize)
        
        # Plot 1: Projection & Expansion
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(t, results['projection'], label="S(t) - Projection")
        ax1.plot(t, results['expansion'], label="D(t) - Expansion")
        ax1.set_title("Projection & Expansion")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Space-Time Density
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(t, results['density'], color='darkred', 
                label="ρ(t) - Time-Density")
        ax2.set_title("Space-Time Density")
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Temporal Flow
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(t, results['temporal_flow'], color='blue', 
                label="Tf(t) - Temporal Flow")
        ax3.set_title("Temporal Flow Modulation")
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Derived Quantities
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(t, results['velocity'], color='green', 
                label="v(t) - Modulated Velocity")
        ax4.plot(t, results['pressure'], color='purple', 
                label="p(t) - Modulated Pressure")
        ax4.set_title("Derived Quantities")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def save_state(self) -> dict:
        """
        Save the current model state (all parameters).
        
        Returns:
        --------
        dict
            Dictionary containing all model parameters
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'omega': self.omega,
            'epsilon': self.epsilon
        }
    
    def load_state(self, state: dict) -> None:
        """
        Load a previously saved model state.
        
        Parameters:
        -----------
        state : dict
            Dictionary containing model parameters
        """
        self.alpha = state.get('alpha', self.alpha)
        self.beta = state.get('beta', self.beta)
        self.omega = state.get('omega', self.omega)
        self.epsilon = state.get('epsilon', self.epsilon)
    
    def simulate_parameter_variation(self, param_name: str, 
                                    values: List[float], 
                                    t_range: Tuple[float, float] = (-12, 12),
                                    num_points: int = 500) -> dict:
        """
        Simulate variation of a specific parameter and calculate results.
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter to vary ('alpha', 'beta', 'omega', or 'epsilon')
        values : list of float
            List of parameter values to simulate
        t_range : tuple of float
            Time range (min, max) for simulation (default: (-12, 12))
        num_points : int
            Number of time points (default: 500)
            
        Returns:
        --------
        dict
            Dictionary with simulation results for each parameter value
        """
        if param_name not in ['alpha', 'beta', 'omega', 'epsilon']:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        t = np.linspace(t_range[0], t_range[1], num_points)
        results = {}
        
        # Save original parameter value
        original_value = getattr(self, param_name)
        
        # Run simulation for each parameter value
        for value in values:
            setattr(self, param_name, value)
            results[value] = self.evaluate_all(t)
        
        # Restore original parameter value
        setattr(self, param_name, original_value)
        
        return results


if __name__ == "__main__":
    # Example usage
    model = GenesisSphereModel()
    t = np.linspace(-12, 12, 1000)
    
    # Calculate values
    rho_vals = model.rho(t)
    tf_vals = model.tf(t)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, rho_vals, label="ρ(t) - Time-Density")
    plt.plot(t, tf_vals, label="Tf(t) - Temporal Flow")
    plt.title("Genesis-Sphere Core Functions")
    plt.xlabel("Time (t)")
    plt.ylabel("Function Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("../output/model_test.png", dpi=300)
    plt.show()
    
    print("Genesis-Sphere model successfully tested.")
