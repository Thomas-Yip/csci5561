# stereo_and_triangulation.py
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
import ast

def read_clicked_points(file_path):
    """
    Reads uvs1 and uvs2 point arrays from a file with Python list format.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: (uvs1, uvs2) as numpy arrays
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Extract uvs1 and uvs2 blocks
    uvs1_block = content.split('uvs1 =')[1].split('uvs2 =')[0].strip()
    uvs2_block = content.split('uvs2 =')[1].strip()

    # Use `ast.literal_eval` for safe parsing
    uvs1 = np.array(ast.literal_eval(uvs1_block))
    uvs2 = np.array(ast.literal_eval(uvs2_block))

    return uvs1, uvs2

# Load intrinsics
def load_calibration(filename):
    data = np.load(filename)
    return data['mtx'], data['dist']

mtx1, dist1 = load_calibration('CalibrationData/c0_calib_data.npz')
mtx2, dist2 = load_calibration('CalibrationData/c1_calib_data.npz')

# Stereo calibration
import cv2 as cv
import glob
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import os

def stereo_calibrate(mtx1, dist1, mtx2, dist2, left_folder, right_folder):
    # Sorted lists of left and right image paths
    c1_images_names = sorted(glob.glob(os.path.join(left_folder, '*.jpg')))
    c2_images_names = sorted(glob.glob(os.path.join(right_folder, '*.jpg')))

    # Load images
    c1_images = [cv.imread(name, 1) for name in c1_images_names]
    c2_images = [cv.imread(name, 1) for name in c2_images_names]

    # Checkerboard configuration
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    rows, columns = 7, 10
    world_scaling = 0.24

    # Prepare 3D object points
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2) * world_scaling

    imgpoints_left, imgpoints_right, objpoints = [], [], []

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 and c_ret2:
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            objpoints.append(objp)

    ret, _, _, _, _, R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
        mtx2, dist2, c1_images[0].shape[1::-1],
        criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC)

    print("Stereo calibration completed. Reprojection Error:", ret)
    np.savez('CalibrationData/stereoExtrinsics.npz', R=R, T=T)
    return R, T

# Example usage:
R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'left_calib', 'right_calib')

uvs1, uvs2 = read_clicked_points("CalibrationData/clicked_points.txt")
 
uvs1 = np.array(uvs1)
uvs2 = np.array(uvs2)

frame1 = cv.imread('media/testLeft.jpg')
frame2 = cv.imread('media/testRight.jpg')

plt.imshow(frame1[..., ::-1]); plt.scatter(uvs1[:,0], uvs1[:,1]); plt.title('Left Image'); plt.show()
plt.imshow(frame2[..., ::-1]); plt.scatter(uvs2[:,0], uvs2[:,1]); plt.title('Right Image'); plt.show()

# Triangulation
RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
RT2 = np.hstack((R, T))

P1 = mtx1 @ RT1
P2 = mtx2 @ RT2

def DLT(P1, P2, point1, point2):
    A = [
        point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ]
    A = np.array(A)
    _, _, Vh = linalg.svd(A)
    X = Vh[-1]
    return X[:3] / X[3]

p3ds = np.array([DLT(P1, P2, pt1, pt2) for pt1, pt2 in zip(uvs1, uvs2)])
np.save('Outputs/triangulated_points.npy', p3ds)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p3ds[:, 0], p3ds[:, 2], -p3ds[:, 1], c='blue')
plt.title("Triangulated 3D Points")
plt.show()
