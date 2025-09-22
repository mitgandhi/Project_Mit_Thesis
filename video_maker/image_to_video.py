import os
import glob
from pathlib import Path
import argparse


def frames_to_video(frames_dir, output_file, fps=30, sort_numerically=True):
    """
    Convert a directory of JPG frames to an MP4 video file.

    Args:
        frames_dir: Directory containing JPG frames
        output_file: Output MP4 file path
        fps: Frames per second
        sort_numerically: Whether to sort frames numerically
    """
    try:
        import imageio
        print(f"Converting frames from: {frames_dir}")
        print(f"Output video: {output_file}")
        print(f"Frames per second: {fps}")

        # Get all JPG files
        frames_pattern = os.path.join(frames_dir, "*.jpg")
        frame_files = glob.glob(frames_pattern)

        if not frame_files:
            print(f"No JPG files found in {frames_dir}")
            return False

        # Sort the frames numerically (e.g. 001.jpg to 360.jpg)
        if sort_numerically:
            # Filter files where stem is a number, then sort by that number
            frame_files = [f for f in frame_files if Path(f).stem.isdigit()]
            frame_files.sort(key=lambda x: int(Path(x).stem))

        else:
            frame_files.sort()

        print(f"Found {len(frame_files)} frames to process.")

        images = []
        for idx, filename in enumerate(frame_files, start=1):
            try:
                img = imageio.imread(filename)
                images.append(img)
                if idx % 50 == 0 or idx == len(frame_files):
                    print(f"Read {idx}/{len(frame_files)} frames...")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        if not images:
            print("No valid images found. Aborting.")
            return False

        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(output_dir, exist_ok=True)

        print("Writing video...")
        imageio.mimsave(output_file, images, fps=fps)

        print(f"✅ Video saved to: {output_file}")
        return True

    except ImportError:
        print("❌ imageio module not found.")
        print("Install it with: pip install imageio imageio-ffmpeg")
        return False

    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert JPG frames to MP4 video")
    parser.add_argument("--input", "-i", required=True,
                        help="Directory containing JPG frames")
    parser.add_argument("--output", "-o", default="output_video.mp4",
                        help="Output MP4 file path (default: output_video.mp4)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--no-sort", action="store_false", dest="sort_numerically",
                        help="Disable numerical sorting (use alphabetical)")

    args = parser.parse_args()

    success = frames_to_video(args.input, args.output, args.fps, args.sort_numerically)
    if success:
        print("🎉 Conversion completed successfully!")
    else:
        print("⚠️ Conversion failed.")


if __name__ == "__main__":
    main()
