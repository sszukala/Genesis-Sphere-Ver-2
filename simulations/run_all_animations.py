import os
import time
import importlib
import sys

# List of animation modules to run
animation_modules = [
    'animation_3d_density',
    'animation_3d_parametric',
    'animation_spacetime_folding',
    'animation_4d_visualization'
]

def run_all_animations():
    """Run all animation scripts and track their outputs"""
    
    print("=" * 80)
    print("GENESIS-SPHERE 3D/4D ANIMATION GENERATOR")
    print("=" * 80)
    print(f"Starting generation of {len(animation_modules)} animations")
    print("This process will take some time as each animation requires rendering multiple frames")
    print("-" * 80)
    
    results = {}
    total_start_time = time.time()
    
    for module_name in animation_modules:
        print(f"\nRunning {module_name}...")
        start_time = time.time()
        
        try:
            # Add the current directory to the path (if not already there)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            
            # Import the module and run its main function
            module = importlib.import_module(module_name)
            output = module.generate_animation()
            
            # Store the result
            results[module_name] = {
                'status': 'success',
                'output': output,
                'time': time.time() - start_time
            }
            
            print(f"Completed {module_name} in {results[module_name]['time']:.2f} seconds")
            
        except Exception as e:
            # Log the error if the animation fails
            results[module_name] = {
                'status': 'error',
                'error': str(e),
                'time': time.time() - start_time
            }
            print(f"Error running {module_name}: {e}")
    
    total_time = time.time() - total_start_time
    
    # Print a summary report
    print("\n" + "=" * 80)
    print("ANIMATION GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds")
    print("-" * 80)
    
    for module_name, result in results.items():
        status = "✅ SUCCESS" if result['status'] == 'success' else "❌ ERROR"
        print(f"{module_name}: {status} ({result['time']:.2f}s)")
        
        if result['status'] == 'success':
            print(f"  Output: {result['output']}")
        else:
            print(f"  Error: {result['error']}")
    
    print("-" * 80)
    print("All animations stored in the '../output/' directory")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    run_all_animations()
