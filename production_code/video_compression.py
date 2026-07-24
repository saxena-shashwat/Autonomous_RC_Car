import cv2
import numpy as np

# Read the video
video_path = r'C:\Users\g3sha\OneDrive\Documents\Electronics\IIIT Vadodara\Advanced-Image-and-Video-Processing\Code\videos\movingDot.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read all frames into a list
video_array = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_array.append(frame)

cap.release()

# Convert list to numpy array
video_array = np.array(video_array)
n = len(video_array)

# Setup video writer
output_path = r'C:\Users\g3sha\OneDrive\Documents\Electronics\IIIT Vadodara\Advanced-Image-and-Video-Processing\Code\videos\test_sliding_box.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter(output_path, fourcc, frame_rate, (640, 480))

# Process frames with sliding window
for i in range(n - 5):
    idx = list(range(i, i + 6))
    grp = video_array[idx].astype(np.float64)
    out = np.sum(grp, axis=0) / 6
    out = out.astype(np.uint8)
    resized_img = cv2.resize(out, (640, 480))
    vw.write(resized_img)

vw.release()
