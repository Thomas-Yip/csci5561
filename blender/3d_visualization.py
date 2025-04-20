import bpy
import numpy as np
import os

#FIRST OPEN  MODEL
# Clear existing objects
#bpy.ops.object.select_all(action='SELECT')
#bpy.ops.object.delete()

# Set FPS and frame spacing
bpy.context.scene.render.fps = 60
frame_spacing = 10

#ONLY LOADS IN ARMS AND HIPS RN
# Load keypoints 2 through 7 (indices 1–6)
selected_indices = list(range(0, 9))

# Path to your keypoints file
keypoints_file = r"C:\Users\nipun_p4ey3oc\OneDrive\Desktop\ComputerVision\FInalProject\3dpart\kpts_3d.dat"

# Read keypoints
def read_keypoints(filename):
    with open(filename, 'r') as fin:
        kpts = []
        for line in fin:
            if line.strip() == '':
                break
            line = list(map(float, line.split()))
            frame = []
            for idx in selected_indices:
                frame.append(line[idx*3 : idx*3+3])
            kpts.append(np.array(frame))
    return np.array(kpts)

# Load and preprocess
p3ds = read_keypoints(keypoints_file)
p3ds = p3ds[:, :, [0, 2, 1]]  # Swap Y and Z
p3ds[:, :, 2] *= -1           # Flip Z

# Normalize and scale
valid = p3ds.reshape(-1, 3)
valid = valid[~np.any(valid == -1, axis=1)]
min_vals = np.min(valid, axis=0)
max_vals = np.max(valid, axis=0)
scale = 8.0 / np.max(max_vals - min_vals)
p3ds = (p3ds - min_vals) * scale

# Create spheres
joints = {}
for i in range(len(selected_indices)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05)
    sphere = bpy.context.active_object
    sphere.name = f"Joint_{selected_indices[i]}"
    joints[i] = sphere

# Animate
current_frame = 1
for frame_pts in p3ds:
    for i, pos in enumerate(frame_pts):
        if not np.any(pos == -1):
            joints[i].location = pos
            joints[i].keyframe_insert(data_path="location", frame=current_frame)
    current_frame += frame_spacing

# Scene settings
bpy.context.scene.frame_end = current_frame
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_set(1)

print("Animated keypoints 2–7 (indices 1–6). Press Alt+A to preview.")
