import cv2
import os
from src.fix_cur_frame import temporal_interpolation

def process_and_save_video(frame_dir='frames', output_video='output.mp4', fps=30):
    """
    Reads all frames from `frame_dir` in sorted order, applies temporal interpolation
    correction frame-by-frame, and writes the corrected frames into `output_video`.
    """
    # 1. Gather sorted frame paths
    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')]
    )
    frame_paths = [os.path.join(frame_dir, f) for f in frame_files]
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    # 2. Read first frame (no correction needed)
    prev = cv2.imread(frame_paths[0])
    if prev is None:
        raise IOError(f"Failed to load {frame_paths[0]}")
    corrected_frames = [prev]

    # 3. Iteratively correct subsequent frames
    for path in frame_paths[1:]:
        cur = cv2.imread(path)
        if cur is None:
            raise IOError(f"Failed to load {path}")
        fixed = temporal_interpolation(prev, cur)
        corrected_frames.append(fixed)
        prev = fixed

    # 4. Initialize video writer
    h, w = corrected_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # 5. Write all corrected frames
    for frame in corrected_frames:
        writer.write(frame)
    writer.release()
    print(f"Saved corrected video to {output_video}")


if __name__ == "__main__":
    process_and_save_video(frame_dir='frames', output_video='output.mp4', fps=30)

