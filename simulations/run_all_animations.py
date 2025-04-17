import os
import time
import importlib
import sys
from tqdm import tqdm
import threading
import queue

# List of animation modules to run
animation_modules = [
    'animation_3d_density',
    'animation_3d_parametric',
    'animation_spacetime_folding',
    'animation_4d_visualization'
]

# Create a queue for processing animations in sequence
animation_queue = queue.Queue()

# Global progress tracking
current_module = ""
current_progress = 0
total_progress = 100
progress_lock = threading.Lock()

def update_progress_bar(progress_bar, module_name, frame, total_frames):
    """Update the progress bar for the current animation"""
    with progress_lock:
        global current_progress
        current_progress = int(100 * frame / total_frames)
        progress_bar.set_description(f"Processing {module_name}")
        progress_bar.update(1)

def run_animation_with_progress(module_name):
    """Run a single animation with progress tracking"""
    try:
        # Import the module
        module = importlib.import_module(module_name)
        
        # Get the original generate_animation function
        original_generate = module.generate_animation
        
        # Create a progress bar for this animation
        frames = 120  # Default frame count
        pbar = tqdm(total=frames, desc=f"Processing {module_name}", position=1, leave=False)
        
        # Monkey patch the frame saving function to update progress
        original_save_frame = None
        
        if hasattr(module, 'save_frame'):
            original_save_frame = module.save_frame
            
            def patched_save_frame(frame_num, *args, **kwargs):
                result = original_save_frame(frame_num, *args, **kwargs)
                update_progress_bar(pbar, module_name, frame_num, frames)
                return result
            
            module.save_frame = patched_save_frame
        
        # Start animation generation
        start_time = time.time()
        output = original_generate()
        elapsed_time = time.time() - start_time
        
        # Close progress bar
        pbar.close()
        
        # Restore original function
        if original_save_frame:
            module.save_frame = original_save_frame
        
        return {
            'status': 'success',
            'output': output,
            'time': elapsed_time
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'time': 0
        }

def process_queue():
    """Process animations from the queue one by one"""
    overall_pbar = tqdm(total=len(animation_modules), desc="Overall progress", position=0)
    results = {}
    
    while not animation_queue.empty():
        module_name = animation_queue.get()
        overall_pbar.set_description(f"Overall progress: Running {module_name}")
        
        # Run the animation with progress tracking
        result = run_animation_with_progress(module_name)
        results[module_name] = result
        
        # Update the overall progress
        overall_pbar.update(1)
        animation_queue.task_done()
    
    overall_pbar.close()
    return results

def run_all_animations():
    """Run all animation scripts with progress tracking and queuing"""
    
    print("=" * 80)
    print("GENESIS-SPHERE 3D/4D ANIMATION GENERATOR")
    print("=" * 80)
    print(f"Queuing {len(animation_modules)} animations for processing")
    print("This process will take some time as each animation requires rendering multiple frames")
    print("-" * 80)
    
    # Add all animations to the queue
    for module_name in animation_modules:
        animation_queue.put(module_name)
    
    # Set up the environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Process the queue and collect results
    total_start_time = time.time()
    results = process_queue()
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
