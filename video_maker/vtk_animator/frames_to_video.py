import os
import glob
from pathlib import Path
import argparse


def frames_to_video(frames_dir, output_file, fps=30, sort_numerically=True):
    """
    Convert a directory of PNG frames to an MP4 video file

    Args:
        frames_dir: Directory containing PNG frames
        output_file: Output MP4 file path
        fps: Frames per second
        sort_numerically: Whether to sort frames numerically (frame_0001.png, frame_0002.png, etc.)
    """
    try:
        import imageio
        print(f"Converting frames from {frames_dir} to video {output_file} at {fps} fps...")

        # Get all PNG files in the directory
        frames_pattern = os.path.join(frames_dir, "*.png")
        frame_files = glob.glob(frames_pattern)

        if not frame_files:
            print(f"No PNG files found in {frames_dir}")
            return False

        # Sort the frames
        if sort_numerically:
            # Extract numbers from filenames and sort based on that
            frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
        else:
            # Simple alphabetical sort
            frame_files.sort()

        print(f"Found {len(frame_files)} frames to process")

        # Read all images into a list
        images = []
        for filename in frame_files:
            try:
                img = imageio.imread(filename)
                images.append(img)
                if len(images) % 50 == 0:
                    print(f"Read {len(images)}/{len(frame_files)} frames...")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        if not images:
            print("Failed to read any valid image frames")
            return False

        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)

        # Write the video file
        print(f"Creating video with {len(images)} frames at {fps} fps...")
        imageio.mimsave(output_file, images, fps=fps)

        print(f"Video saved to {output_file}")
        return True

    except ImportError:
        print("Error: imageio module not found.")
        print("Please install it with: pip install imageio imageio-ffmpeg")
        return False

    except Exception as e:
        print(f"Error creating video: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PNG frames to MP4 video")
    parser.add_argument("--input", "-i", default="./temp_frames",
                        help="Directory containing PNG frames (default: ./temp_frames)")
    parser.add_argument("--output", "-o", default="animation.mp4",
                        help="Output MP4 file path (default: animation.mp4)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--no-sort", action="store_false", dest="sort_numerically",
                        help="Disable numerical sorting (use alphabetical)")

    args = parser.parse_args()

    success = frames_to_video(args.input, args.output, args.fps, args.sort_numerically)
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed.")


if __name__ == "__main__":
    main()