"""
Helper utilities for animation scripts, including progress tracking
"""
import os
import shutil
import subprocess
from tqdm import tqdm

def check_ffmpeg():
    """Check if FFmpeg is available in the system or common installation locations"""
    if shutil.which('ffmpeg'):
        return True
    
    # Common installation locations on Windows
    common_locations = [
        os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'
    ]
    
    for location in common_locations:
        if os.path.exists(location):
            print(f"FFmpeg found at {location} but not in system PATH")
            print("Temporarily adding to PATH for this session...")
            os.environ['PATH'] += os.pathsep + os.path.dirname(location)
            return True
    
    return False

def save_frame(i, total_frames, fig, frames_dir, prefix="frame"):
    """Save a single frame with progress tracking"""
    # Save the frame
    frame_file = os.path.join(frames_dir, f'{prefix}_{i:03d}.png')
    fig.savefig(frame_file, dpi=150)
    return frame_file

def run_ffmpeg(frames_dir, output_file, fps=30, silent=False):
    """Run FFmpeg with progress tracking"""
    # FFmpeg command to create a video from the frames
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite existing files
        '-framerate', str(fps),  # Frames per second
        '-i', os.path.join(frames_dir, 'frame_%03d.png'),
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '20',  # Quality (lower is better)
        '-pix_fmt', 'yuv420p',
        output_file
    ]
    
    if silent:
        cmd.extend(['-loglevel', 'error'])
    
    print(f"Running FFmpeg to create video...")
    
    # Run FFmpeg with progress updates
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Track FFmpeg progress
    with tqdm(total=100, desc="Encoding video", position=1, leave=False) as pbar:
        last_progress = 0
        for line in process.stdout:
            if "frame=" in line and "fps=" in line and "time=" in line:
                # FFmpeg progress line, extract time
                time_parts = line.split("time=")[1].split()[0].split(":")
                if len(time_parts) == 3:
                    hours, minutes, seconds = time_parts
                    seconds = float(seconds)
                    total_seconds = int(hours) * 3600 + int(minutes) * 60 + seconds
                    
                    # Estimate progress (assume 10 seconds video)
                    progress = min(int(total_seconds * 10), 100)
                    pbar.update(progress - last_progress)
                    last_progress = progress
    
    # Check if FFmpeg completed successfully
    return_code = process.wait()
    return return_code == 0

def generate_animation_with_progress(frames_dir, generate_frame_func, output_file, total_frames=120, fps=30):
    """Generic function to generate an animation with progress tracking"""
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate each frame with progress tracking
    with tqdm(total=total_frames, desc="Generating frames", position=1, leave=False) as pbar:
        for i in range(total_frames):
            generate_frame_func(i, total_frames)
            pbar.update(1)
    
    # Create video if FFmpeg is available
    if check_ffmpeg():
        try:
            success = run_ffmpeg(frames_dir, output_file, fps)
            if success:
                print(f"Animation saved successfully to {output_file}")
                return output_file
            else:
                print("Error creating video with FFmpeg")
                return frames_dir
        except Exception as e:
            print(f"Error creating video with FFmpeg: {e}")
            return frames_dir
    else:
        print("FFmpeg not found. Cannot create video.")
        return frames_dir
