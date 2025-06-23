import cv2
import numpy as np

n = 5  # Number of previous frames to use


def temporal_interpolation(prev_frames, cur_frame, alpha=0.7, flow_params=None, morph_kernel_size=3):
    """
    Perform motion‑compensated interpolation to fill changed pixels in cur_frame
    using a list of previous frames as reference.

    Args:
        prev_frames (List[np.ndarray]): List of BGR images of previous frames (most recent first).
        cur_frame   (np.ndarray):        BGR image of the current frame to correct.
        alpha (float): Blend factor between warped and original pixels (0–1).
        flow_params (dict, optional): Parameters for Farneback optical flow.
        morph_kernel_size (int): Kernel size for morphological opening of the change mask.

    Returns:
        np.ndarray: Corrected version of cur_frame (BGR).
    """
    # Default Farneback parameters if none provided
    if flow_params is None:
        flow_params = {
            'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
            'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2, 'flags': 0
        }

    # 1. Generate flash bitmasks for cur_frame corresponding to each prev_frame
    masks = []
    for prev_frame in prev_frames:
        # Convert to grayscale
        gray_cur = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute absolute difference
        diff = cv2.absdiff(gray_cur, gray_prev)

        # Normalize the difference
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply threshold (you may need to adjust the threshold value)
        _, mask = cv2.threshold(diff_norm, 15, 255, cv2.THRESH_BINARY)

        # Morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks.append(mask)

    # 2. Combine masks (bitwise OR)
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 3. Pixel-wise averaging using previous frames
    corrected_frame = cur_frame.copy().astype(np.float32)
    for i in range(cur_frame.shape[0]):
        for j in range(cur_frame.shape[1]):
            if combined_mask[i, j] == 0:  # If not a flash pixel
                pixel_sum = np.zeros(3, dtype=np.float32)
                for prev_frame in prev_frames:
                    pixel_sum += prev_frame[i, j]
                corrected_frame[i, j] = pixel_sum / len(prev_frames)

    return corrected_frame.astype(np.uint8)

