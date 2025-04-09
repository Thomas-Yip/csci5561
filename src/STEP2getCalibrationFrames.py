import cv2
import numpy as np
import os
import shutil

# Checkerboard detection
def detect_checkerboard(frame, checkerboard_size=(7, 10)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    return ret, corners if ret else None

# Folder setup
def setup_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Setup save directories
left_save_dir = "left_calib"
right_save_dir = "right_calib"
setup_folder(left_save_dir)
setup_folder(right_save_dir)

# Video paths
left_video_path = 'media/left.mp4'
right_video_path = 'media/right.mp4'

# Open videos
cap_left = cv2.VideoCapture(left_video_path)
cap_right = cv2.VideoCapture(right_video_path)

fps_left = cap_left.get(cv2.CAP_PROP_FPS)
fps_right = cap_right.get(cv2.CAP_PROP_FPS)

frame_interval_left = int(fps_left * 4)
frame_interval_right = int(fps_right * 4)

# Get total frame count
total_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
total_time = min(total_frames_left / fps_left, total_frames_right / fps_right)

print(f"Processing every 4s up to {total_time:.2f}s")

# Loop through at 4-second intervals
t = 0
while True:
    frame_idx_left = int(t * fps_left)
    frame_idx_right = int(t * fps_right)

    if frame_idx_left >= total_frames_left or frame_idx_right >= total_frames_right:
        break

    # Seek directly to the frame
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_left)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_right)

    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        break

    # Detect checkerboard
    found_left, _ = detect_checkerboard(frame_left)
    found_right, _ = detect_checkerboard(frame_right)

    if found_left and found_right:
        timestamp = int(t)
        cv2.imwrite(os.path.join(left_save_dir, f"left_calib_{timestamp}s.jpg"), frame_left)
        cv2.imwrite(os.path.join(right_save_dir, f"right_calib_{timestamp}s.jpg"), frame_right)
        print(f"Saved checkerboard at {timestamp}s")

    t += 4  # Move to next interval

# Clean up
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
