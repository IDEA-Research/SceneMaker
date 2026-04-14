#!/usr/bin/env python3
"""
Recursively convert all mp4 files under a directory to gif format.
Preserve the original directory structure.
"""

import os
import argparse
from pathlib import Path
try:
    # moviepy 2.x
    from moviepy import VideoFileClip
except ImportError:
    # moviepy 1.x
    from moviepy.editor import VideoFileClip
from tqdm import tqdm


def find_mp4_files(root_dir):
    """Recursively find all mp4 files."""
    mp4_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.mp4'):
                mp4_path = os.path.join(root, file)
                mp4_files.append(mp4_path)
    return mp4_files


def convert_mp4_to_gif(mp4_path, output_path=None, fps=10, resize=None):
    """
    Convert an mp4 file to gif.
    
    Args:
        mp4_path: mp4 file path.
        output_path: output gif path; if None, replace the mp4 extension with gif.
        fps: gif frame rate, default is 10.
        resize: resize ratio (0-1) or None to keep original size.
    """
    if output_path is None:
        output_path = mp4_path.rsplit('.', 1)[0] + '.gif'
    
    try:
        # load video
        clip = VideoFileClip(mp4_path)
        
        # resize video (optional)
        if resize is not None and 0 < resize < 1:
            # get original size and compute new size
            new_width = int(clip.w * resize)
            new_height = int(clip.h * resize)
            # resize with PIL
            from PIL import Image
            def resize_frame(frame):
                img = Image.fromarray(frame)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                return np.array(img)
            
            import numpy as np
            clip = clip.fl_image(resize_frame)
        
        # convert to gif
        clip.write_gif(output_path, fps=fps)
        clip.close()
        
        return True, output_path
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Recursively convert all mp4 files under a directory to gif')
    parser.add_argument('--dir', type=str, default='.', 
                        help='Root directory to convert (default: current directory)')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='GIF output directory (default: same as mp4 directory)')
    parser.add_argument('--fps', type=int, default=10, 
                        help='GIF frame rate (default: 10)')
    parser.add_argument('--resize', type=float, default=None, 
                        help='Resize ratio, e.g. 0.5 means half size (default: no resize)')
    parser.add_argument('--keep-mp4', action='store_true', 
                        help='Keep original mp4 files (default: delete)')
    
    args = parser.parse_args()
    
    # find all mp4 files
    print(f"Scanning directory: {args.dir}")
    mp4_files = find_mp4_files(args.dir)
    
    if not mp4_files:
        print("No mp4 files found")
        return
    
    print(f"Found {len(mp4_files)} mp4 files")
    
    # convert each file
    success_count = 0
    failed_files = []
    
    for mp4_path in tqdm(mp4_files, desc="Conversion progress"):
        # compute output path
        if args.output_dir:
            # keep relative directory structure
            rel_path = os.path.relpath(mp4_path, args.dir)
            gif_path = os.path.join(args.output_dir, rel_path)
            gif_path = gif_path.rsplit('.', 1)[0] + '.gif'
            # create output directory
            os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        else:
            # same directory as mp4
            gif_path = mp4_path.rsplit('.', 1)[0] + '.gif'
        
        # skip if gif already exists
        if os.path.exists(gif_path):
            print(f"\nSkipped (gif already exists): {gif_path}")
            continue
        
        success, result = convert_mp4_to_gif(
            mp4_path, 
            output_path=gif_path,
            fps=args.fps, 
            resize=args.resize
        )
        
        if success:
            success_count += 1
            # delete original mp4 file (if specified)
            if not args.keep_mp4:
                try:
                    os.remove(mp4_path)
                except Exception as e:
                    print(f"\nWarning: failed to delete {mp4_path}: {e}")
        else:
            failed_files.append((mp4_path, result))
            print(f"\nConversion failed: {mp4_path}")
            print(f"Error: {result}")
    
    # print summary
    print("\n" + "="*50)
    print(f"Conversion completed: {success_count}/{len(mp4_files)} succeeded")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for mp4_path, error in failed_files:
            print(f"  - {mp4_path}")
            print(f"    Error: {error}")


if __name__ == "__main__":
    main()
