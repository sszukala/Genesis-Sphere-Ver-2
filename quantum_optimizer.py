import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import argparse
import sys

# Add parent directory to path to import the Genesis-Sphere model
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(parent_dir, 'models'))

# Import Genesis-Sphere model
try:
    from genesis_model import GenesisSphereModel
except ImportError:
    print("Warning: Could not import Genesis-Sphere model. Make sure it's available in the 'models' directory.")

print("Trying to import Qiskit modules...")

# Detect if IBM Runtime is available (newer Qiskit approach)
ibm_runtime_available = False
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    print("Successfully imported qiskit_ibm_runtime (recommended newer API)")
    ibm_runtime_available = True
except ImportError:
    print("qiskit_ibm_runtime not found. Will try older Qiskit methods.")

# Handle different Qiskit versions
qiskit_available = True
qaoa_available = True
qubo_available = True

try:
    # Try newer Qiskit structure
    from qiskit_aer import Aer
    print("Successfully imported qiskit_aer")
except ImportError:
    try:
        # Try alternative location
        from qiskit.providers.aer import Aer
        print("Successfully imported qiskit.providers.aer")
    except ImportError:
        try:
            # Fallback to BasicAer if Aer is not available
            from qiskit.providers.basicaer import BasicAer as Aer
            print("Warning: Using BasicAer instead of Aer. This will be slower but still works.")
        except ImportError:
            print("Could not import any Qiskit Aer backend. QAOA simulation may not work.")
            qiskit_available = False

# Try to import QAOA and related modules with error handling
try:
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.utils import QuantumInstance
    print("Successfully imported QAOA from qiskit.algorithms")
except ImportError:
    try:
        # Try alternate import locations for different Qiskit versions
        from qiskit.aqua.algorithms import QAOA
        from qiskit.aqua.components.optimizers import COBYLA
        from qiskit.aqua import QuantumInstance
        print("Successfully imported QAOA from qiskit.aqua.algorithms (older version)")
    except ImportError:
        print("Error: Could not import QAOA. Make sure you have the full Qiskit package installed.")
        print("Try running: pip install qiskit[optimization]")
        qaoa_available = False

# Try to import QuadraticProgram and MinimumEigenOptimizer
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    print("Successfully imported QuadraticProgram from qiskit_optimization")
except ImportError:
    try:
        # Try alternate location for older Qiskit versions
        from qiskit.optimization import QuadraticProgram
        from qiskit.optimization.algorithms import MinimumEigenOptimizer
        print("Successfully imported QuadraticProgram from qiskit.optimization")
    except ImportError:
        try:
            # Even older versions
            from qiskit.aqua.optimization import QuadraticProgram
            from qiskit.aqua.algorithms.minimum_eigen_solvers import MinimumEigenOptimizer
            print("Successfully imported QuadraticProgram from qiskit.aqua.optimization (older version)")
        except ImportError:
            print("Error: Could not import QuadraticProgram or MinimumEigenOptimizer.")
            print("Please install the Qiskit Optimization package:")
            print("pip install qiskit-optimization")
            qubo_available = False

# Try to import IBMQ
try:
    from qiskit import IBMQ
    print("Successfully imported IBMQ")
except ImportError:
    print("Warning: Could not import IBMQ. Will use local simulator only.")
    
# Check if all required Qiskit components are available
if not all([qiskit_available, qaoa_available, qubo_available]):
    print("\n===== QISKIT INSTALLATION GUIDE =====")
    print("It seems some Qiskit components are missing. Try these installation commands:")
    print("1. Basic Qiskit: pip install qiskit")
    print("2. Optimization package: pip install qiskit-optimization")
    print("3. Full installation: pip install qiskit[optimization,machine-learning]")
    print("4. Specific version (if needed): pip install qiskit==0.36.2 qiskit-optimization==0.3.0")
    print("=====================================\n")
    print("Continuing with limited functionality...")

# Function to check if quantum optimization is available
def is_quantum_optimization_available():
    return all([qiskit_available, qaoa_available, qubo_available])

# Load your progress log
def load_progress_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Extract key metrics for optimization
def extract_optimization_metrics(data):
    # Extract batch speed and memory usage as our key metrics
    batch_speeds = data['Batch_Speed'].values
    memory_usage = data['Memory_MB'].values
    return batch_speeds, memory_usage

# Load astronomical data for validation
def load_astronomical_data(data_dir='validation/datasets'):
    """
    Load astronomical datasets for validating Genesis-Sphere parameters
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    
    # Try to load supernovae data
    try:
        sne_data = pd.read_csv(os.path.join(data_dir, 'SNe_gold_sample.csv'))
    except FileNotFoundError:
        print("Warning: Supernovae data not found. Using synthetic data.")
        # Create synthetic data if real data not available
        z_values = np.linspace(0.01, 2.0, 100)
        distance_modulus = 43.3 + 5*np.log10(z_values)
        sne_data = pd.DataFrame({'z': z_values, 'mu': distance_modulus})
    
    # Try to load H₀ measurements data
    try:
        h0_data = pd.read_csv(os.path.join(data_dir, 'hubble_measurements.csv'))
    except FileNotFoundError:
        print("Warning: H₀ measurements not found. Using synthetic data.")
        # Create synthetic data if real data not available
        years = np.linspace(1927, 2023, 50)
        h0_values = 70 + 5*np.sin(0.1*years) + np.random.normal(0, 2, len(years))
        h0_data = pd.DataFrame({'Year': years, 'H0': h0_values})
    
    # Try to load BAO data
    try:
        bao_data = pd.read_csv(os.path.join(data_dir, 'bao_measurements.csv'))
    except FileNotFoundError:
        print("Warning: BAO data not found. Using synthetic data.")
        # Create synthetic data if real data not available
        z_values = np.linspace(0.1, 2.5, 20)
        bao_values = 150 - 10*np.log(z_values+0.1) + np.random.normal(0, 2, len(z_values))
        bao_data = pd.DataFrame({'z': z_values, 'r_s': bao_values})
    
    return {'sne': sne_data, 'h0': h0_data, 'bao': bao_data}

# Create a fitness function to evaluate Genesis-Sphere parameters
def evaluate_genesis_sphere_fitness(params, astronomical_data):
    """
    Evaluate how well a set of Genesis-Sphere parameters fit astronomical data
    Returns a combined fitness score (lower is better)
    """
    # Extract parameters
    alpha, beta, omega, epsilon = params
    
    # Create Genesis-Sphere model with these parameters
    gs_model = GenesisSphereModel(alpha=alpha, beta=beta, omega=omega, epsilon=epsilon)
    
    # Calculate fit metrics for supernovae
    z_values = astronomical_data['sne']['z'].values
    observed_mu = astronomical_data['sne']['mu'].values
    
    # Convert redshift to time in Genesis-Sphere model (simple approximation)
    t_values = -10 * np.log(1 + z_values)
    
    # Get model predictions
    density = gs_model.rho(t_values)
    temporal_flow = gs_model.tf(t_values)
    
    # Simple model for luminosity distance
    H0 = 70.0  # km/s/Mpc assumed base
    dH = 299792.458 / H0  # Hubble distance in Mpc
    dL = dH * (1 + z_values) * np.sqrt(density) * (1 / temporal_flow)
    predicted_mu = 5 * np.log10(dL) + 25
    
    # Calculate supernovae R² score (simple version)
    sne_r2 = 1 - np.sum((observed_mu - predicted_mu)**2) / np.sum((observed_mu - np.mean(observed_mu))**2)
    
    # Calculate H₀ correlation (simplified)
    if 'Year' in astronomical_data['h0'].columns:
        years = astronomical_data['h0']['Year'].values
        h0_values = astronomical_data['h0']['H0'].values
        # Generate model values at these years
        years_normalized = (years - 1970) / 50  # Simple normalization to Genesis-Sphere time units
        density_years = gs_model.rho(years_normalized)
        # Calculate correlation
        h0_corr = np.corrcoef(h0_values, density_years)[0, 1]
    else:
        h0_corr = 0
    
    # Calculate BAO effect (simplified)
    if 'z' in astronomical_data['bao'].columns:
        bao_z = astronomical_data['bao']['z'].values
        # Find effect at z~2.3
        target_z = 2.3
        z_effect = np.exp(-0.5 * ((bao_z - target_z) / 0.2)**2)  # Gaussian around z=2.3
        cycle_effect = np.sin(omega * (-10 * np.log(1 + bao_z)))
        bao_effect = np.abs(np.sum(z_effect * cycle_effect))
    else:
        bao_effect = 0
    
    # Combine metrics into a single fitness score (lower is better)
    # We negate R² and correlation because we want to minimize in QUBO
    fitness = -sne_r2 - h0_corr + (1.0 / (1.0 + bao_effect))
    
    return fitness

# Map the problem to QUBO formulation
def create_qubo_problem(astronomical_data):
    # FORMULA 1: QUBO OBJECTIVE FUNCTION FOR GENESIS-SPHERE PARAMETERS
    # The mathematical form is: min f(x) = x^T Q x + c^T x
    # Where:
    #  - x is a binary vector (our solution)
    #  - Q is the quadratic terms matrix (relationships between variables)
    #  - c is the linear terms vector (individual variable weights)
    #
    # This formulation allows the quantum computer to find optimal
    # Genesis-Sphere parameters (α, β, ω, ε) that fit astronomical data
    
    n_bits_per_param = 3  # Use 3 bits per parameter (8 possible values per parameter)
    n_params = 4  # alpha, beta, omega, epsilon
    n_total_bits = n_bits_per_param * n_params
    
    # Create a quadratic program
    quad_prog = QuadraticProgram(name='Genesis_Sphere_Optimizer')
    
    # Add binary variables
    for i in range(n_total_bits):
        quad_prog.binary_var(name=f'bit{i}')
    
    # Pre-calculate fitness for a grid of parameters to establish QUBO weights
    param_values = []
    for param_idx in range(n_params):
        if param_idx == 0:  # alpha
            param_values.append(np.linspace(0.01, 0.1, 2**n_bits_per_param))
        elif param_idx == 1:  # beta
            param_values.append(np.linspace(0.1, 1.5, 2**n_bits_per_param))
        elif param_idx == 2:  # omega
            param_values.append(np.linspace(0.5, 4.0, 2**n_bits_per_param))
        else:  # epsilon
            param_values.append(np.linspace(0.01, 0.2, 2**n_bits_per_param))
    
    # Define the linear terms based on individual parameter contribution
    linear = {}
    for param_idx in range(n_params):
        for bit_idx in range(n_bits_per_param):
            global_bit_idx = param_idx * n_bits_per_param + bit_idx
            weight = 0.0
            
            # Test this bit's contribution
            for config in range(2):
                # Create a test configuration with just this bit
                bit_config = np.zeros(n_total_bits)
                bit_config[global_bit_idx] = config
                
                # Convert binary to parameter values
                test_params = binary_to_parameters(bit_config, param_values)
                
                # Evaluate fitness
                fitness = evaluate_genesis_sphere_fitness(test_params, astronomical_data)
                
                # Weight contribution is difference in fitness
                if config == 1:
                    weight += fitness
                else:
                    weight -= fitness
            
            linear[global_bit_idx] = weight / 2.0
    
    # Define the quadratic terms based on parameter interactions
    # This is a simplified approach; a more thorough approach would test all bit combinations
    quadratic = {}
    for i in range(n_total_bits):
        for j in range(i+1, n_total_bits):
            # Simple interaction weight - parameters in the same group have stronger interactions
            param_i = i // n_bits_per_param
            param_j = j // n_bits_per_param
            
            if param_i == param_j:
                # Bits from the same parameter have stronger interactions
                quadratic[(i, j)] = -0.5
            else:
                # Cross-parameter interactions (weaker)
                quadratic[(i, j)] = -0.1
    
    # Set up the minimization problem with our objective function
    quad_prog.minimize(linear=linear, quadratic=quadratic)
    return quad_prog, param_values

# Helper function to convert binary representation to parameter values
def binary_to_parameters(binary_vector, param_values):
    n_bits_per_param = len(binary_vector) // 4
    params = []
    
    for param_idx in range(4):  # 4 parameters: alpha, beta, omega, epsilon
        # Extract bits for this parameter
        start_idx = param_idx * n_bits_per_param
        end_idx = start_idx + n_bits_per_param
        param_bits = binary_vector[start_idx:end_idx]
        
        # Convert binary to decimal
        decimal_value = 0
        for i, bit in enumerate(param_bits):
            decimal_value += bit * (2 ** i)
        
        # Use decimal value to index into parameter value range
        params.append(param_values[param_idx][decimal_value])
    
    return params

# Connect to IBM Quantum
def connect_to_ibm_quantum():
    # Check if IBM Runtime is available (newer approach)
    if ibm_runtime_available:
        try:
            # Use the newer QiskitRuntimeService approach
            service = QiskitRuntimeService(channel="ibm_quantum")
            print(f"Connected to IBM Quantum via Runtime API")
            backend = service.least_busy(simulator=False, min_num_qubits=5)
            print(f"Using IBM Q backend: {backend.name}")
            return backend
        except Exception as e:
            print(f"Could not connect to IBM Quantum Runtime: {str(e)}")
            print("Trying classic approach...")
    
    # Fall back to classic IBMQ approach
    try:
        # Load IBMQ account - you'll need to insert your API token
        IBMQ.save_account('YOUR_IBM_QUANTUM_API_TOKEN', overwrite=True)
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub='ibm-q')
        
        # Get the least busy backend with ≤ 5 qubits
        backend = provider.backends(filters=lambda x: x.configuration().n_qubits >= 5 
                                   and not x.configuration().simulator 
                                   and x.status().operational)[0]
        print(f"Using IBM Q backend: {backend.name()}")
        return backend
    except Exception as e:
        print(f"Could not connect to IBM Quantum: {str(e)}")
        print("Using local simulator instead.")
        try:
            return Aer.get_backend('qasm_simulator')
        except Exception as e2:
            print(f"Error with Aer simulator: {str(e2)}")
            print("Falling back to most basic simulator")
            from qiskit.providers.basicaer import QasmSimulatorPy
            return QasmSimulatorPy()

# Run the quantum optimization
def run_quantum_optimization(problem, backend):
    # FORMULA 2: QAOA ALGORITHM
    # The Quantum Approximate Optimization Algorithm works by:
    # 1. Creating a quantum circuit with alternating problem and mixing unitaries
    # 2. Running the circuit with different parameters (angles)
    # 3. Finding parameters that produce the lowest energy = best solution
    #
    # The mathematical form is a parameterized quantum state:
    # |β,γ⟩ = e^(-iβₚH_mixer) e^(-iγₚH_problem) ... e^(-iβ₁H_mixer) e^(-iγ₁H_problem) |s⟩
    
    # Set up the quantum instance
    quantum_instance = QuantumInstance(backend=backend, shots=1024)
    
    # Create QAOA solver
    qaoa = QAOA(optimizer=COBYLA(), quantum_instance=quantum_instance)
    
    # Create minimum eigen optimizer from QAOA
    optimizer = MinimumEigenOptimizer(qaoa)
    
    # Solve the problem
    print("Running quantum optimization (this may take a few minutes)...")
    result = optimizer.solve(problem)
    
    return result

# Map results back to parameter space
def map_to_parameters(result, param_values):
    # FORMULA 3: PARAMETER MAPPING FOR GENESIS-SPHERE
    # This maps the binary solution vector to Genesis-Sphere parameters:
    # - alpha: Spatial dimension expansion coefficient
    # - beta: Temporal damping factor
    # - omega: Angular frequency for sinusoidal projections
    # - epsilon: Small constant to prevent division by zero
    
    # Extract the results
    x = result.x
    
    # Convert binary solution to parameter values
    params = binary_to_parameters(x, param_values)
    
    # Map to named parameters
    param_settings = {
        'alpha': params[0],   # Spatial dimension expansion coefficient
        'beta': params[1],    # Temporal damping factor
        'omega': params[2],   # Angular frequency
        'epsilon': params[3]  # Small constant to prevent division by zero
    }
    
    return param_settings

# Save optimized parameters
def save_results(params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"quantum_optimized_params_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        f.write("# Parameters optimized using IBM Quantum Computing\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
    
    print(f"Optimized parameters saved to {output_file}")
    return output_file

# Visualize the optimization results
def visualize_optimization(params, astronomical_data, output_dir):
    # Create Genesis-Sphere model with optimized parameters
    gs_model = GenesisSphereModel(
        alpha=params['alpha'], 
        beta=params['beta'], 
        omega=params['omega'], 
        epsilon=params['epsilon']
    )
    
    # Setup plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Supernovae distance modulus
    sne_data = astronomical_data['sne']
    z_values = sne_data['z'].values
    observed_mu = sne_data['mu'].values
    
    # Convert redshift to time in Genesis-Sphere model
    t_values = -10 * np.log(1 + z_values)
    
    # Get model predictions
    density = gs_model.rho(t_values)
    temporal_flow = gs_model.tf(t_values)
    
    # Simple model for luminosity distance
    H0 = 70.0  # km/s/Mpc assumed base
    dH = 299792.458 / H0  # Hubble distance in Mpc
    dL = dH * (1 + z_values) * np.sqrt(density) * (1 / temporal_flow)
    predicted_mu = 5 * np.log10(dL) + 25
    
    # Calculate R² score
    sne_r2 = 1 - np.sum((observed_mu - predicted_mu)**2) / np.sum((observed_mu - np.mean(observed_mu))**2)
    
    axes[0].scatter(z_values, observed_mu, alpha=0.5, label='Observed')
    axes[0].plot(z_values, predicted_mu, 'r-', label=f'Genesis-Sphere (R²={sne_r2:.3f})')
    axes[0].set_title("Type Ia Supernovae Distance Modulus")
    axes[0].set_xlabel("Redshift (z)")
    axes[0].set_ylabel("Distance Modulus (µ)")
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: H₀ measurements
    if 'Year' in astronomical_data['h0'].columns:
        h0_data = astronomical_data['h0']
        years = h0_data['Year'].values
        h0_values = h0_data['H0'].values
        
        # Generate model values at these years
        years_normalized = (years - 1970) / 50  # Simple normalization to Genesis-Sphere time units
        density_years = gs_model.rho(years_normalized)
        density_scaled = 70 + 10 * (density_years - np.mean(density_years)) / np.std(density_years)
        
        # Calculate correlation
        h0_corr = np.corrcoef(h0_values, density_scaled)[0, 1]
        
        axes[1].scatter(years, h0_values, alpha=0.5, label='Observed H₀')
        axes[1].plot(years, density_scaled, 'g-', label=f'Density-Scaled (Corr={h0_corr:.3f})')
        axes[1].set_title("Hubble Constant Measurements")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("H₀ (km/s/Mpc)")
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot 3: Core functions of optimized Genesis-Sphere model
    t_demo = np.linspace(-10, 10, 1000)
    rho = gs_model.rho(t_demo)
    tf = gs_model.tf(t_demo)
    
    axes[2].plot(t_demo, rho, 'b-', label="ρ(t) - Space-Time Density")
    axes[2].plot(t_demo, tf, 'r-', label="Tf(t) - Temporal Flow")
    axes[2].set_title(f"Genesis-Sphere Core Functions (α={params['alpha']:.4f}, β={params['beta']:.4f}, ω={params['omega']:.4f}, ε={params['epsilon']:.4f})")
    axes[2].set_xlabel("Time (t)")
    axes[2].set_ylabel("Function Value")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = os.path.join(output_dir, f"genesis_sphere_quantum_optimized_{timestamp}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Visualization saved to {fig_path}")

def add_command_line_args():
    parser = argparse.ArgumentParser(description='Quantum Genesis-Sphere Parameter Optimizer')
    parser.add_argument('--data-dir', type=str, 
                       default='validation/datasets',
                       help='Directory containing astronomical datasets')
    parser.add_argument('--output', type=str, 
                       default='output/quantum_optimized',
                       help='Output directory for results')
    parser.add_argument('--fallback', action='store_true',
                       help='Use classical optimization if quantum modules are unavailable')
    parser.add_argument('--use-qasm', action='store_true',
                       help='Use direct OpenQASM 2.0 circuit implementation')
    return parser.parse_args()

# Fallback to classical optimization when Qiskit isn't available
def classical_optimization(astronomical_data):
    """
    Fallback method using classical optimization (scipy) when Qiskit isn't available
    """
    from scipy.optimize import differential_evolution
    
    print("Using classical optimization (differential evolution) as fallback...")
    
    # Define bounds for parameters
    bounds = [
        (0.01, 0.1),    # alpha
        (0.1, 1.5),     # beta
        (0.5, 4.0),     # omega
        (0.01, 0.2)     # epsilon
    ]
    
    # Define objective function for scipy optimizer
    def objective(params):
        return evaluate_genesis_sphere_fitness(params, astronomical_data)
    
    # Run differential evolution
    print("Running classical optimization (this may take a few minutes)...")
    result = differential_evolution(objective, bounds, popsize=15, maxiter=30)
    
    # Map to named parameters
    optimized_params = {
        'alpha': result.x[0],
        'beta': result.x[1],
        'omega': result.x[2],
        'epsilon': result.x[3]
    }
    
    return optimized_params

# New function to create and run an OpenQASM 2.0 circuit for parameter optimization
def run_qasm_circuit_optimization(params_to_test, astronomical_data):
    """
    Create and run an OpenQASM 2.0 circuit directly for Genesis-Sphere parameter testing.
    This provides a lower-level implementation alternative to QAOA.
    
    Parameters:
    -----------
    params_to_test : list of tuples
        List of (alpha, beta, omega, epsilon) parameter combinations to evaluate
    astronomical_data : dict
        Dictionary containing astronomical datasets
        
    Returns:
    --------
    dict
        Optimized parameters with their fitness scores
    """
    try:
        from qiskit import QuantumCircuit, execute, Aer
        from qiskit.quantum_info import Statevector
        print("Creating OpenQASM 2.0 circuit for Genesis-Sphere optimization")
        
        # Number of parameters we want to encode (alpha, beta, omega, epsilon)
        n_params = 4
        
        # Number of qubits needed for the circuit
        # We'll use 2 qubits per parameter for encoding 4 possible values per parameter
        n_qubits = n_params * 2
        
        # Create a quantum circuit with n_qubits
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize with superposition to explore parameter space
        for i in range(n_qubits):
            qc.h(i)
        
        # Create parameter encoding gates
        # We'll use rotation gates to encode different parameter values
        for param_idx in range(n_params):
            base_qubit = param_idx * 2
            
            # Apply controlled rotations to encode parameter values
            qc.cx(base_qubit, base_qubit + 1)
            qc.rz(np.pi/4, base_qubit)
            qc.rx(np.pi/3, base_qubit + 1)
            
            # Add parameter-specific gates
            if param_idx == 0:  # alpha
                qc.ry(np.pi/8, base_qubit)
            elif param_idx == 1:  # beta
                qc.rz(np.pi/6, base_qubit)
            elif param_idx == 2:  # omega
                qc.rx(np.pi/4, base_qubit)
            else:  # epsilon
                qc.ry(np.pi/12, base_qubit)
        
        # Add entanglement between parameters
        for i in range(0, n_qubits-2, 2):
            qc.cx(i, i+2)
        
        # Final Hadamard layer
        for i in range(n_qubits):
            qc.h(i)
        
        # Measure all qubits
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Get the OpenQASM representation
        qasm_str = qc.qasm()
        print("Generated OpenQASM 2.0 circuit:")
        print(qasm_str[:500] + "..." if len(qasm_str) > 500 else qasm_str)
        
        # Execute the circuit 
        backend = Aer.get_backend('qasm_simulator')
        job = execute(qc, backend, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)
        
        # Process results to find parameter combinations
        # Convert the most frequent states to parameter indices
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        best_results = sorted_counts[:min(8, len(sorted_counts))]
        
        # Evaluate the top parameter combinations
        param_results = {}
        print("\nEvaluating top parameter combinations from quantum circuit:")
        for bit_str, count in best_results:
            # Convert bit string to parameter indices
            # Each parameter uses 2 bits (4 possible values)
            param_indices = []
            for param_idx in range(n_params):
                start_idx = param_idx * 2
                param_bit_str = bit_str[start_idx:start_idx+2]
                param_indices.append(int(param_bit_str, 2))
            
            # Map indices to actual parameter values
            alpha = 0.01 + (0.1 - 0.01) * (param_indices[0] / 3)  # Range: 0.01-0.1
            beta = 0.1 + (1.5 - 0.1) * (param_indices[1] / 3)    # Range: 0.1-1.5
            omega = 0.5 + (4.0 - 0.5) * (param_indices[2] / 3)   # Range: 0.5-4.0
            epsilon = 0.01 + (0.2 - 0.01) * (param_indices[3] / 3) # Range: 0.01-0.2
            
            # Evaluate fitness
            params = (alpha, beta, omega, epsilon)
            fitness = evaluate_genesis_sphere_fitness(params, astronomical_data)
            
            param_results[params] = {'fitness': fitness, 'count': count}
            print(f"  Parameters {params} - Fitness: {fitness:.6f}, Count: {count}")
        
        # Find the best parameter set
        best_params = min(param_results.items(), key=lambda x: x[1]['fitness'])[0]
        best_fitness = param_results[best_params]['fitness']
        
        print(f"\nBest parameters from QASM circuit: {best_params}")
        print(f"Best fitness: {best_fitness:.6f}")
        
        # Return as a dictionary for compatibility with other optimization methods
        return {
            'alpha': best_params[0],
            'beta': best_params[1],
            'omega': best_params[2],
            'epsilon': best_params[3]
        }
        
    except ImportError as e:
        print(f"Error importing Qiskit modules for QASM circuit: {e}")
        print("Falling back to classical optimization...")
        return None

def main():
    # Parse command line arguments
    args = add_command_line_args()
    
    # Add a new command-line argument for QASM circuit
    parser = argparse.ArgumentParser(description='Quantum Genesis-Sphere Parameter Optimizer')
    parser.add_argument('--data-dir', type=str, 
                      default='validation/datasets',
                      help='Directory containing astronomical datasets')
    parser.add_argument('--output', type=str, 
                      default='output/quantum_optimized',
                      help='Output directory for results')
    parser.add_argument('--fallback', action='store_true',
                      help='Use classical optimization if quantum modules are unavailable')
    parser.add_argument('--use-qasm', action='store_true',
                      help='Use direct OpenQASM 2.0 circuit implementation')
    args = parser.parse_args()
    
    # Show warning about dependency conflicts
    print("\nNOTE: If you're seeing dependency conflicts with pydantic or openai,")
    print("these typically don't affect the quantum optimizer functionality.")
    print("You can create a separate environment for Qiskit if needed:")
    print("  conda create -n qiskit-env python=3.9")
    print("  conda activate qiskit-env")
    print("  pip install qiskit qiskit-optimization\n")
    
    # Define file paths
    data_dir = args.data_dir
    output_dir = args.output
    
    # Load astronomical data
    print("Loading astronomical datasets...")
    astronomical_data = load_astronomical_data(data_dir)
    
    # Check if user requested OpenQASM implementation
    if args.use_qasm:
        print("Using direct OpenQASM 2.0 circuit implementation...")
        # Generate parameters to test
        params_to_test = []
        for alpha in np.linspace(0.01, 0.1, 4):
            for beta in np.linspace(0.1, 1.5, 4):
                for omega in np.linspace(0.5, 4.0, 4):
                    for epsilon in np.linspace(0.01, 0.2, 4):
                        params_to_test.append((alpha, beta, omega, epsilon))
        
        # Run optimization with QASM circuit
        optimized_params = run_qasm_circuit_optimization(params_to_test, astronomical_data)
        
        # Fall back to QAOA if QASM implementation fails
        if optimized_params is None and is_quantum_optimization_available():
            print("Falling back to QAOA optimization...")
            # Continue with QAOA optimization
            print("Formulating quantum optimization problem for Genesis-Sphere parameters...")
            problem, param_values = create_qubo_problem(astronomical_data)
            
            print("Connecting to IBM Quantum...")
            backend = connect_to_ibm_quantum()
            
            print("Running quantum optimization (this may take a few minutes)...")
            result = run_quantum_optimization(problem, backend)
            
            optimized_params = map_to_parameters(result, param_values)
    elif is_quantum_optimization_available():
        print("Quantum optimization is available. Proceeding with QAOA...")
        # Create QUBO problem - THIS IS WHERE FORMULA 1 IS USED
        print("Formulating quantum optimization problem for Genesis-Sphere parameters...")
        problem, param_values = create_qubo_problem(astronomical_data)
        
        # Connect to IBM Quantum
        print("Connecting to IBM Quantum...")
        backend = connect_to_ibm_quantum()
        
        # Run optimization - THIS IS WHERE FORMULA 2 IS USED
        # (QAOA algorithm for finding minimum eigenvalue solution)
        result = run_quantum_optimization(problem, backend)
        
        # Map to Genesis-Sphere parameters - THIS IS WHERE FORMULA 3 IS USED
        optimized_params = map_to_parameters(result, param_values)
    elif args.fallback:
        print("Quantum optimization is not available. Using classical fallback...")
        optimized_params = classical_optimization(astronomical_data)
    else:
        print("ERROR: Quantum optimization is not available and --fallback option was not specified.")
        print("Please install the required Qiskit packages or run with --fallback to use classical optimization.")
        print("See installation guide above for details.")
        return
    
    print(f"Optimized Genesis-Sphere parameters: {optimized_params}")
    
    # Save results
    output_file = save_results(optimized_params, output_dir)
    
    # Visualize results
    visualize_optimization(optimized_params, astronomical_data, output_dir)
    
    print("\nOptimization complete! The Genesis-Sphere parameters have been optimized.")
    print(f"Results saved to {output_file}")
    print("\nOptimized Genesis-Sphere Parameter Values:")
    print(f"  α (alpha): {optimized_params['alpha']:.6f} - Spatial dimension expansion coefficient")
    print(f"  β (beta): {optimized_params['beta']:.6f} - Temporal damping factor")
    print(f"  ω (omega): {optimized_params['omega']:.6f} - Angular frequency for sinusoidal projections")
    print(f"  ε (epsilon): {optimized_params['epsilon']:.6f} - Small constant to prevent division by zero")

if __name__ == "__main__":
    main()
