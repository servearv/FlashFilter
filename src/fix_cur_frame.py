import cv2
import numpy as np

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

    # 1. Grayscale conversion of current frame
    gray_cur = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

    # 2. Build change mask using the most recent previous frame
    gray_prev0 = cv2.cvtColor(prev_frames[0], cv2.COLOR_BGR2GRAY)
    diff       = cv2.subtract(gray_cur, gray_prev0)
    diff_eq    = cv2.equalizeHist(diff)
    kernel     = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
    )
    diff_clean = cv2.morphologyEx(diff_eq, cv2.MORPH_OPEN, kernel)
    _, mask    = cv2.threshold(
        diff_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. For each previous frame, compute flow and warp to current
    warps = []
    for prev in prev_frames:
        gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_cur, None,
            **flow_params
        )
        h, w = gray_prev.shape
        coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float32)
        map_x = coords[..., 0] + flow[..., 0]
        map_y = coords[..., 1] + flow[..., 1]
        warped = cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        warps.append(warped.astype(np.float32))

    # 4. Combine warped frames by simple average
    combined_warp = np.mean(warps, axis=0).astype(np.uint8)

    # 5. Fill masked regions and blend with current frame
    result = cur_frame.copy().astype(np.float32)
    mask_bool = (mask == 255)
    # Apply channel-wise blending
    for c in range(3):
        channel = result[..., c]
        warped_chan = combined_warp[..., c]
        channel[mask_bool] = (
            alpha * warped_chan[mask_bool] +
            (1 - alpha) * channel[mask_bool]
        )
    result = result.astype(np.uint8)

    return result

