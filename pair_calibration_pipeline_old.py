import numpy as np
import cv2
import json
import sys
from sys import platform
import os
from glob import glob
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0' # To avoid CUDNN errors


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/openpose/build/python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose/build/x64/Release;' +  dir_path + '/openpose/build/bin;'
        import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Load camera calibration parameters for each camera

calib_file_path = "data/intrinsic/*.npz"
calib_files = glob(os.path.join(calib_file_path))
calib_files = [f'{calib_file_path}/Calib_Parameters517.npz',
               f'{calib_file_path}/Calib_Parameters518.npz',
               f'{calib_file_path}/Calib_Parameters536.npz',
               f'{calib_file_path}/Calib_Parameters520.npz']
cameras = []
for calib_file in calib_files:
    with np.load(calib_file) as data:
        mtx, dist, frame_size, pxsize = [data[f] for f in ('mtx', 'dist', 'frame_size', 'pxsize')]
        cameras.append({'matrix': mtx, 'distortion': dist, 'frame_size': frame_size, 'pxsize': pxsize})
print(cameras)


# Load the OpenPose .json files for both camera views
json_file_1 = 'data/json/posed_000000000034_keypoints.json'
json_file_2 = 'data/json/posed_000000000034_keypoints.json'
with open(json_file_1, 'r') as f1, open(json_file_2, 'r') as f2:
    data_1 = json.load(f1)
    data_2 = json.load(f2)

# Define the 3D world coordinates of the keypoints
# This will depend on the pose estimation model and the joints detected
world_points = np.array([
    # Example world points for COCO model
    (0.0, 0.0, 0.0),  # Nose
    (0.0, -1.0, 0.0),  # Neck
    (-0.5, -1.5, 0.0),  # Left Shoulder
    (0.5, -1.5, 0.0),  # Right Shoulder
    (-1.0, -2.5, 0.0),  # Left Elbow
    (1.0, -2.5, 0.0),  # Right Elbow
    (-1.5, -3.5, 0.0),  # Left Wrist
    (1.5, -3.5, 0.0),  # Right Wrist
    (-0.5, -4.5, 0.0),  # Left Hip
    (0.5, -4.5, 0.0),  # Right Hip
    (-1.0, -6.0, 0.0),  # Left Knee
    (1.0, -6.0, 0.0),  # Right Knee
    (-1.5, -7.5, 0.0),  # Left Ankle
    (1.5, -7.5, 0.0)  # Right Ankle
])

# Extract the 2D image coordinates of the keypoints from the OpenPose .json files
image_points_1 = []
image_points_2 = []
for i in range(len(data_1['people'])):
    keypoints_1 = data_1['people'][i]['pose_keypoints_2d']
    keypoints_2 = data_2['people'][i]['pose_keypoints_2d']
    for j in range(0, len(keypoints_1), 3):
        x_1 = keypoints_1[j]
        y_1 = keypoints_1[j + 1]
        x_2 = keypoints_2[j]
        y_2 = keypoints_2[j + 1]
        image_points_1.append((x_1, y_1))
        image_points_2.append((x_2, y_2))

# Load the camera intrinsic parameters from the .npz file
npz_file = 'data/intrinsic/Calib_Parameters517.npz'
with np.load(npz_file) as data:
    K1 = data['mtx']
    dist_coeffs_1 = data['dist']
    frame_size = data['frame_size']
    pxsize = data['pxsize']

# Define the camera matrix for the second camera view (assuming same intrinsic parameters)
K2 = K1

# Convert the pixel size to millimeters
pxsize_mm = pxsize * 1000.0

# Estimate the essential matrix between the two camera views
E, _ = cv2.findEssentialMat(np.array(image_points_1), np.array(image_points_2), K1, cv2.RANSAC, 0.999, 1.0)

# Estimate the rotation and translation vectors between the two camera views
_, R, t, _ = cv2.recoverPose(E, np.array(image_points_1), np.array(image_points_2), K2)

# Convert the translation vector from pixels to millimeters
t = t * pxsize_mm

# Print the rotation and translation vectors
print("Rotation matrix:\n", R)
print("Translation vector:\n", t)