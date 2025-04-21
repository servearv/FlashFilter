import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

video_path = 'videoplayback.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video file: {video_path}")

frames, hists = [], []
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    frames.append(frame); hists.append(hist)
cap.release()

fig, (ax_frame, ax_hist) = plt.subplots(1, 2, figsize=(12,5))
im = ax_frame.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
ax_frame.axis('off')
bars = ax_hist.bar(range(256), hists[0])
ax_hist.set_xlim(0,255)
ax_hist.set_ylim(0, max(h.max() for h in hists)*1.1)

def init():
    """Draw the first frame & histogram."""
    im.set_data(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    for bar, h in zip(bars, hists[0]):
        bar.set_height(h)
    return [im] + list(bars)

def update(i):
    im.set_data(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
    ax_frame.set_title(f"Frame {i+1}/{len(frames)}")
    for bar, h in zip(bars, hists[i]):
        bar.set_height(h)
    ax_hist.set_title(f"Histogram {i+1}/{len(frames)}")
    return [im] + list(bars)

ani = FuncAnimation(
    fig, update,
    frames=len(frames),
    init_func=init,
    interval=50,
    blit=True
)

plt.tight_layout()
plt.show()

