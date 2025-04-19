import cv2
import os

# Path to the input video
video_path = 'videoplayback.mp4'

# If frames directory exists, remove it
if os.path.exists('frames'):
    import shutil
    shutil.rmtree('frames')

# Directory where frames will be saved
output_dir = 'frames'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video file: {video_path}")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Save frame as a numbered PNG (e.g., frame_00001.png)
    filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(filename, frame)
    frame_idx += 1

cap.release()
print(f"Saved {frame_idx} frames to '{output_dir}'")


