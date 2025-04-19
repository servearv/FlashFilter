#!/usr/bin/env python3
import os
import cv2
import importlib.util

# Dynamically load your module
spec = importlib.util.spec_from_file_location("fix_cur_frame", "src/fix_cur_frame.py")
fix_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix_mod)
temporal_interpolation = fix_mod.temporal_interpolation

def process_and_save_video(frame_dir='frames',
                           output_video='output.mp4',
                           fps=30,
                           window_size=None):
    """
    Reads all frames from `frame_dir`, applies temporal interpolation
    using up to `window_size` past frames (default=all),
    and writes the corrected frames into `output_video`.
    """
    # 1. Gather sorted frame paths
    frame_files = sorted(f for f in os.listdir(frame_dir)
                         if f.startswith('frame_') and f.endswith('.png'))
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frame_dir}")
    frame_paths = [os.path.join(frame_dir, f) for f in frame_files]

    # 2. Read first frame
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise IOError(f"Failed to load {frame_paths[0]}")
    corrected = [first]

    # 3. Iteratively correct subsequent frames
    for path in frame_paths[1:]:
        cur = cv2.imread(path)
        if cur is None:
            raise IOError(f"Failed to load {path}")
        # choose last `window_size` frames or all if None
        prev_window = corrected if window_size is None else corrected[-window_size:]
        fixed = temporal_interpolation(prev_window, cur)
        corrected.append(fixed)

    # 4. Initialize video writer
    h, w = corrected[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # 5. Write out
    for frame in corrected:
        writer.write(frame)
    writer.release()
    print(f"Saved corrected video: {output_video}")

if __name__ == "__main__":
    # e.g. only use the last 3 frames for interpolation:
    process_and_save_video(frame_dir='frames',
                           output_video='output.mp4',
                           fps=30,
                           window_size=3)

