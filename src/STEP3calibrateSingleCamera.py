# calibrate_individual_cameras.py
import cv2 as cv
import glob
import numpy as np

def calibrate_camera(images_folder, save_prefix):
    images_names = sorted(glob.glob(images_folder))
    images = [cv.imread(imname, 1) for imname in images_names]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows, columns = 7, 10
    world_scaling = 0.24
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2) * world_scaling

    width, height = images[0].shape[1], images[0].shape[0]
    imgpoints, objpoints = [], []

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)

    print(f"{save_prefix} Camera RMSE: {ret}")
    print(f"{save_prefix} Camera Matrix:\n{mtx}")
    print(f"{save_prefix} Distortion Coefficients:\n{dist.ravel()}")

    np.savez(f"CalibrationData/{save_prefix}_calib_data.npz", mtx=mtx, dist=dist)
    return mtx, dist

# Calibrate and save both cameras
calibrate_camera('left_calib/*', 'c0')
calibrate_camera('right_calib/*', 'c1')