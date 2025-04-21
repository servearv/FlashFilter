import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where frames are stored
frame_dir = 'frames'

# Paths to the reference (frame0) and target (frame5) frames
ref_path = os.path.join(frame_dir, 'frame_00000.png')
tar_path = os.path.join(frame_dir, 'frame_00060.png')

# Load the frames
ref = cv2.imread(ref_path)
tar = cv2.imread(tar_path)
if ref is None or tar is None:
    raise FileNotFoundError("Could not load the frames. Check your 'frames' directory and filenames.")

# Convert to grayscale for mask creation and optical flow
gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
gray_tar = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

# Compute binary mask of changed pixels (as before)
diff = cv2.subtract(gray_tar, gray_ref)
diff_eq = cv2.equalizeHist(diff)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
diff_clean = cv2.morphologyEx(diff_eq, cv2.MORPH_OPEN, kernel)
_, mask = cv2.threshold(diff_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Compute dense optical flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(
    gray_ref, gray_tar, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# Warp ref frame to align with tar frame
h, w = gray_ref.shape
coords = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float32)
map_x = coords[..., 0] + flow[..., 0]
map_y = coords[..., 1] + flow[..., 1]
warped_ref = cv2.remap(ref, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# Create output by copying warped pixels into masked regions of tar
result = tar.copy()
for c in range(3):
    result[..., c][mask == 255] = warped_ref[..., c][mask == 255]

# Optional: Blend warped and original for smooth transition
alpha = 0.7
for c in range(3):
    result[..., c][mask == 255] = (
        alpha * warped_ref[..., c][mask == 255] +
        (1 - alpha) * tar[..., c][mask == 255]
    )

# Convert to RGB for display
tar_rgb = cv2.cvtColor(tar, cv2.COLOR_BGR2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

# Display the original target, mask, and result
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(tar_rgb); axs[0].set_title('Frame 00005 (Original)'); axs[0].axis('off')
axs[1].imshow(mask_rgb); axs[1].set_title('Changed Pixels Mask'); axs[1].axis('off')
axs[2].imshow(ref); axs[2].set_title('OG Image'); axs[2].axis('off')
axs[3].imshow(result_rgb); axs[3].set_title('Interpolated via Optical Flow'); axs[2].axis('off')
plt.tight_layout()
plt.show()

