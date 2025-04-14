import cv2
import os
num_cams = 1 # choose number of cameras
vid_caps = []
# Enter your camera IP addresses here
cam_ip_addr = ['http://10.143.10.97:4747/video', 
               'http://10.143.10.207:4747/video'
]
outs = []
ws_path = os.getcwd()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i in range(num_cams):
    cap = cv2.VideoCapture(cam_ip_addr[i])
    # cap = cv2.VideoCapture(0)
    vid_caps.append(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = ws_path+"/media/output_cam"f"{i+1}.mp4"
    print(output_path)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    outs.append(out)

while all([cap.isOpened() for cap in vid_caps]):
    frames = []
    for i in range(num_cams):
        ret, frame = vid_caps[i].read()
        if ret:
            frames.append(frame)
            outs[i].write(frame)
    show_frame = frames[0] 

    for i in range(1, len(frames)):
        show_frame = cv2.hconcat([show_frame, frames[i]])

    cv2.imshow('Synced Video', show_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("exit")
        break

for i in range(num_cams):
    vid_caps[i].release()
    outs[i].release()

cv2.destroyAllWindows()
