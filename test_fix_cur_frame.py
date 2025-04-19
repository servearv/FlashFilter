# test_fix_cur_frame.py

import os
import sys
import cv2
import numpy as np

# Add src directory to import path
sys.path.insert(0, os.path.abspath('src'))
from fix_cur_frame import temporal_interpolation

def temporal_interpolation_with_mask(prev_frame, cur_frame, alpha=0.7, flow_params=None, morph_kernel_size=5):
    """
    Run temporal_interpolation and also return the change mask.
    """
    # Grayscale conversion
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_cur  = cv2.cvtColor(cur_frame,  cv2.COLOR_BGR2GRAY)
    # Create change mask
    diff       = cv2.subtract(gray_cur, gray_prev)
    diff_eq    = cv2.equalizeHist(diff)
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    diff_clean = cv2.morphologyEx(diff_eq, cv2.MORPH_OPEN, kernel)
    _, mask    = cv2.threshold(diff_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Get corrected frame
    corrected = temporal_interpolation(prev_frame, cur_frame, alpha, flow_params, morph_kernel_size)
    return corrected, mask

def main(frame_dir='frames'):
    # Gather and sort frame files
    frames = sorted(f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png'))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frame_dir}")
    # Read first frame
    prev = cv2.imread(os.path.join(frame_dir, frames[0]))
    if prev is None:
        raise IOError(f"Failed to load {frames[0]}")
    prev_corrected = prev.copy()
    # Create display window
    cv2.namedWindow('Original | Corrected | Mask', cv2.WINDOW_NORMAL)
    # Loop through subsequent frames
    for fname in frames[1:]:
        path = os.path.join(frame_dir, fname)
        cur = cv2.imread(path)
        if cur is None:
            print(f"Warning: could not load {fname}, skipping")
            continue
        corrected, mask = temporal_interpolation_with_mask(prev_corrected, cur)
        prev_corrected = corrected
        # Prepare mask for 3-channel display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Resize to match if necessary
        h, w = cur.shape[:2]
        corrected_resized = cv2.resize(corrected, (w, h))
        mask_resized      = cv2.resize(mask_bgr, (w, h))
        # Concatenate side by side
        combined = np.hstack([cur, corrected_resized, mask_resized])
        cv2.imshow('Original | Corrected | Mask', combined)
        if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

