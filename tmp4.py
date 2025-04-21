import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to frame images
frame_dir = 'frames'
frame0_path = os.path.join(frame_dir, 'frame_00003.png')
frame5_path = os.path.join(frame_dir, 'frame_00004.png')

# Load frames
frame0 = cv2.imread(frame0_path)
frame5 = cv2.imread(frame5_path)

# Convert to grayscale float
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
gray5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

# Feature detection and matching (SURF or fallback to ORB)
try:
    surf = cv2.xfeatures2d.SURF_create(400)
    kp0, des0 = surf.detectAndCompute(gray0, None)
    kp5, des5 = surf.detectAndCompute(gray5, None)
except:
    print("SURF not available, using ORB instead.")
    orb = cv2.ORB_create(1000)
    kp0, des0 = orb.detectAndCompute((gray0*255).astype(np.uint8), None)
    kp5, des5 = orb.detectAndCompute((gray5*255).astype(np.uint8), None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des5)
else:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des5, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    matches = good

# Align images using matched keypoints
src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp5[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

aligned5_gray = cv2.warpPerspective(gray5, H, (gray0.shape[1], gray0.shape[0]))
aligned5_color = cv2.warpPerspective(frame5, H, (gray0.shape[1], gray0.shape[0]))

# Step 4: Absolute difference
diff = np.abs(gray0 - aligned5_gray)

# Equalize the histogram of the difference image
diff_eq = cv2.equalizeHist((diff * 255).astype(np.uint8))

# Step 5: Threshold to get bitmask (flash regions)
_, mask = cv2.threshold(diff_eq, 0.1, 1.0, cv2.THRESH_BINARY)
mask = mask.astype(np.uint8)

# Step 6: Interpolate masked region using optical flow
flow = cv2.calcOpticalFlowFarneback(gray0, aligned5_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
h, w = gray0.shape
flow_map = np.meshgrid(np.arange(w), np.arange(h))
map_x = (flow_map[0] + flow[..., 0]).astype(np.float32)
map_y = (flow_map[1] + flow[..., 1]).astype(np.float32)
interpolated_color = cv2.remap(frame0, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# Apply correction only on masked region
corrected_color = aligned5_color.copy()
for c in range(3):
    corrected_color[..., c][mask == 1] = interpolated_color[..., c][mask == 1]

# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.title("Frame 0")
plt.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Frame 5 (aligned)")
plt.imshow(cv2.cvtColor(aligned5_color, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Flash Bitmask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Corrected Frame 5")
plt.imshow(cv2.cvtColor(corrected_color, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
