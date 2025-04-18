import cv2
import numpy as np

def temporal_interpolation(prev_frame, cur_frame, alpha=0.7, flow_params=None, morph_kernel_size=5):
    """
    Perform motion‑compensated interpolation to fill changed pixels in cur_frame
    using prev_frame as reference.
    
    Args:
        prev_frame (np.ndarray): BGR image of the previous frame.
        cur_frame  (np.ndarray): BGR image of the current frame to correct.
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

    # 1. Grayscale conversion
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_cur  = cv2.cvtColor(cur_frame,  cv2.COLOR_BGR2GRAY)
    
    # 2. Change mask via differencing + cleanup
    diff       = cv2.subtract(gray_cur, gray_prev)
    diff_eq    = cv2.equalizeHist(diff)
    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    diff_clean = cv2.morphologyEx(diff_eq, cv2.MORPH_OPEN, kernel)
    _, mask    = cv2.threshold(diff_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Optical flow from prev → cur
    flow = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_cur, None,
        **flow_params
    )
    
    # 4. Warp prev_frame to align with cur_frame
    h, w = gray_prev.shape
    coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float32)
    map_x = coords[..., 0] + flow[..., 0]
    map_y = coords[..., 1] + flow[..., 1]
    warped_prev = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # 5. Fill masked regions and blend
    result = cur_frame.copy()
    for c in range(3):
        # copy warped pixels where change detected
        result[..., c][mask == 255] = warped_prev[..., c][mask == 255]
        # blend for smooth transition
        result[..., c][mask == 255] = (
            alpha * warped_prev[..., c][mask == 255] +
            (1 - alpha) * cur_frame[..., c][mask == 255]
        )
    
    return result
