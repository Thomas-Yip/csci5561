import cv2
import numpy as np

# --- Config ---
timestamp = 168  # seconds
left_video_path = 'media/left.mp4'
right_video_path = 'media/right.mp4'
left_frame_path = 'media/testLeft.jpg'
right_frame_path = 'media/testRight.jpg'

# --- Frame Extraction ---
def save_frame_at_timestamp(video_path, timestamp, output_filename):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(timestamp * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()

    cap.release()

    if ret:
        cv2.imwrite(output_filename, frame)
        print(f"Saved frame at {timestamp}s from '{video_path}' as '{output_filename}'")
        return frame
    else:
        print(f"Failed to retrieve frame at {timestamp}s from '{video_path}'")
        return None

# --- Mouse Callback ---
def get_pixel_value(event, x, y, flags, param):
    image, points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = image[y, x]
        print(f"Clicked: ({x}, {y}) - RGB: ({r}, {g}, {b}), HEX: #{r:02x}{g:02x}{b:02x}")
        points.append((x, y))

# --- Interactive Click Tool ---
def display_image_for_pixel_extraction(image, window_title='Image'):
    img_display = image.copy()
    points = []

    # Scale window to fit screen
    screen_res = (1920, 1080)
    img_h, img_w = img_display.shape[:2]
    scale = min((screen_res[0] * 0.8) / img_w, (screen_res[1] * 0.8) / img_h)
    window_size = (int(img_w * scale), int(img_h * scale))

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, *window_size)
    cv2.setMouseCallback(window_title, get_pixel_value, (img_display, points))

    print("Click on points, press ESC when done.")
    while True:
        disp_img = img_display.copy()
        for pt in points:
            cv2.circle(disp_img, pt, 3, (0, 0, 255), -1)
        cv2.imshow(window_title, disp_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return np.array(points)

# --- Main Flow ---
def main():
    # Extract and save frames
    frame_left = save_frame_at_timestamp(left_video_path, timestamp, left_frame_path)
    frame_right = save_frame_at_timestamp(right_video_path, timestamp, right_frame_path)

    if frame_left is None or frame_right is None:
        print("Aborting due to missing frame(s).")
        return

    # Get clicked points
    points_left = display_image_for_pixel_extraction(frame_left, 'Left Frame')
    points_right = display_image_for_pixel_extraction(frame_right, 'Right Frame')

    # Save points
    with open('CalibrationData/clicked_points.txt', 'w') as f:
        f.write("uvs1 = [\n")
        for pt in points_left:
            f.write(f"    {pt.tolist()},\n")
        f.write("]\n\n")

        f.write("uvs2 = [\n")
        for pt in points_right:
            f.write(f"    {pt.tolist()},\n")
        f.write("]\n")

    print("Clicked points saved to 'clicked_points.txt'.")

if __name__ == "__main__":
    main()
