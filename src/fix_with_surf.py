import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def temporal_interpolation(prev_frames, cur_frame, alpha=0.7, flow_params=None, morph_kernel_size=3):
    ref_frame = prev_frames[0]
    gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_cur = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        kp_ref, des_ref = surf.detectAndCompute(gray_ref, None)
        kp_cur, des_cur = surf.detectAndCompute(gray_cur, None)
    except:
        orb = cv2.ORB_create(1000)
        kp_ref, des_ref = orb.detectAndCompute((gray_ref * 255).astype(np.uint8), None)
        kp_cur, des_cur = orb.detectAndCompute((gray_cur * 255).astype(np.uint8), None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_ref, des_cur)
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_ref, des_cur, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        matches = good

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

    aligned_cur_gray = cv2.warpPerspective(gray_cur, H, (gray_ref.shape[1], gray_ref.shape[0]))
    aligned_cur_color = cv2.warpPerspective(cur_frame, H, (gray_ref.shape[1], gray_ref.shape[0]))

    diff = np.abs(gray_ref - aligned_cur_gray)
    _, mask = cv2.threshold(diff, 0.1, 1.0, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    flow = cv2.calcOpticalFlowFarneback(
        gray_ref, aligned_cur_gray, None,
        *(flow_params or dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)).values()
    )

    h, w = gray_ref.shape
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + flow[..., 0]).astype(np.float32)
    map_y = (map_y + flow[..., 1]).astype(np.float32)

    interpolated = cv2.remap(ref_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    corrected = aligned_cur_color.copy()
    for c in range(3):
        corrected[..., c][mask == 1] = (
            alpha * interpolated[..., c][mask == 1] + (1 - alpha) * aligned_cur_color[..., c][mask == 1]
        ).astype(np.uint8)

    return corrected

