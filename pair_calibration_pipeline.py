import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0' # To avoid CUDNN errors
from glob import glob
from sys import platform
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from numpy.linalg import svd
# Import OpenPose
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

def load_camera_params(calib_file_path, cam_num_1, cam_num_2, cam_num_3, cam_num_4):
    """
    Load camera calibration parameters for a set of cameras from npz files
    """
    camera_files = [
        f"{calib_file_path}/Calib_Parameters{cam_num_1}.npz",
        f"{calib_file_path}/Calib_Parameters{cam_num_2}.npz",
        f"{calib_file_path}/Calib_Parameters{cam_num_3}.npz",
        f"{calib_file_path}/Calib_Parameters{cam_num_4}.npz"
    ]
    cameras = []
    for camera_file in camera_files:
        with np.load(camera_file) as data:
            mtx, dist, frame_size, pxsize = [data[f] for f in ('mtx', 'dist', 'frame_size', 'pxsize')]
            cameras.append({'matrix': mtx, 'distortion': dist, 'frame_size': frame_size, 'pxsize': pxsize})
    print(f"Loaded intrinsics for cameras {cam_num_1}, {cam_num_2}, {cam_num_3}, and {cam_num_4}")
    print(cameras)
    return cameras

def extract_keypoints_pair1_517_518(images_pair1_517, images_pair1_518, opWrapper):
    """
    Extract keypoints and confidences for each image
    """
    images_cam1 = images_pair1_517
    images_cam2 = images_pair1_518

    keypoints_cam1 = []
    keypoints_cam2 = []
    confidences_cam1 = []
    confidences_cam2 = []

    mapping = op.getPoseBodyPartMapping(op.PoseModel.BODY_25)
    strings = np.array([value for value in mapping.values()]).reshape(-1, 1)[:-1]

    # handle images from camera 1
    for image in images_cam1:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam1.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam1.append(average_confidence)

    # handle images from camera 2
    for image in images_cam2:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam2.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam2.append(average_confidence)

    # Transform the list of keypoints into arrays
    keypoints1 = np.vstack(keypoints_cam1)
    keypoints2 = np.vstack(keypoints_cam2)

    confidences1 = np.array(confidences_cam1)
    confidences2 = np.array(confidences_cam2)

    return keypoints1, keypoints2, confidences1, confidences2

def extract_keypoints_pair2_517_536(images_pair2_517, images_pair2_536, opWrapper):
    """
    Extract keypoints and confidences for each image
    """

    images_cam1 = images_pair2_517
    images_cam2 = images_pair2_536
    keypoints_cam1 = []
    keypoints_cam2 = []
    confidences_cam1 = []
    confidences_cam2 = []

    mapping = op.getPoseBodyPartMapping(op.PoseModel.BODY_25)
    strings = np.array([value for value in mapping.values()]).reshape(-1, 1)[:-1]

    # handle images from camera 1
    for image in images_cam1:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam1.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam1.append(average_confidence)

    # handle images from camera 2
    for image in images_cam2:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam2.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam2.append(average_confidence)

    # Transform the list of keypoints into arrays
    keypoints1 = np.vstack(keypoints_cam1)
    keypoints2 = np.vstack(keypoints_cam2)

    confidences1 = np.array(confidences_cam1)
    confidences2 = np.array(confidences_cam2)

    return keypoints1, keypoints2, confidences1, confidences2

def extract_keypoints_pair3_517_520(images_pair3_517, images_pair3_520, opWrapper):
    """
    Extract keypoints and confidences for each image
    """

    images_cam1 = images_pair3_517
    images_cam2 = images_pair3_520
    keypoints_cam1 = []
    keypoints_cam2 = []
    confidences_cam1 = []
    confidences_cam2 = []

    mapping = op.getPoseBodyPartMapping(op.PoseModel.BODY_25)
    strings = np.array([value for value in mapping.values()]).reshape(-1, 1)[:-1]

    # handle images from camera 1
    for image in images_cam1:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam1.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam1.append(average_confidence)

    # handle images from camera 2
    for image in images_cam2:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam2.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam2.append(average_confidence)

    # Transform the list of keypoints into arrays
    keypoints1 = np.vstack(keypoints_cam1)
    keypoints2 = np.vstack(keypoints_cam2)

    confidences1 = np.array(confidences_cam1)
    confidences2 = np.array(confidences_cam2)

    return keypoints1, keypoints2, confidences1, confidences2

def detect_charuco_diamonds(images):
    """
    Detects Charuco diamonds in images and extracts 2D positions of ArUco IDs
    """
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    square_length = 24.
    marker_length = 16.
    charuco_board = cv2.aruco.CharucoBoard(
        (3, 3),
        squareLength=24,
        markerLength=16,
        dictionary=charuco_dict)

    charuco_corners_all = []
    aruco_ids_coords_all = []

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_dict)
        if len(corners) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

            charuco_corners_all.append(charuco_corners)

            aruco_ids_coords = []
            for id in charuco_ids.flatten():
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, charuco_dict)

                for marker_corner, marker_id in zip(marker_corners, marker_ids):
                    if marker_id == id:
                        if marker_corner.shape[0] == 1:
                            marker_corner = np.squeeze(marker_corner, axis=0)
                        aruco_ids_coords.append(marker_corner[0])

            aruco_ids_coords_all.append(aruco_ids_coords)
        else:
            charuco_corners_all.append(None)
            aruco_ids_coords_all.append(None)

    charuco_corn1 = np.array(charuco_corners_all[:1])
    charuco_corn2 = np.array(charuco_corners_all[1:])
    aruco_ids_coords_all = np.array(aruco_ids_coords_all, dtype=float)

    # Create a new array to hold the new elements with the desired shape
    new_elements = np.expand_dims(aruco_ids_coords_all, axis=2)
    # Concatenate charuco_corn1 with new_elements along the second axis
    combined_corn1 = np.concatenate((charuco_corn1, new_elements[:1]), axis=1)
    combined_corn2 = np.concatenate((charuco_corn2, new_elements[1:]), axis=1)
    # Update charuco_corn1 with the combined result
    charuco_corn1 = combined_corn1
    charuco_corn2 = combined_corn2
    charuco_corn1 = charuco_corn1[:, :, 0]
    charuco_corn2 = charuco_corn2[:, :, 0]
    charuco_corn1 = np.squeeze(charuco_corn1, axis=0)
    charuco_corn2 = np.squeeze(charuco_corn2, axis=0)

    return charuco_corn1, charuco_corn2, aruco_ids_coords_all

def filter_keypoints_pair1_517_518(keypoints_pair1_517, keypoints_pair1_518, threshold_pair1):
    """
    Filter keypoints based on the confidence score threshold and keep only the keypoints with matching IDs.

    Args:
        keypoints1 (np.ndarray): First keypoints array with shape (N, 4) where N is the number of keypoints.
        keypoints2 (np.ndarray): Second keypoints array with shape (N, 4) where N is the number of keypoints.
        threshold (float): Confidence score threshold, keypoints with scores below this value will be removed.

    Returns:
        np.ndarray, np.ndarray: Filtered keypoints arrays for keypoints1 and keypoints2 with matching IDs.
    """
    keypoints1 = keypoints_pair1_517
    keypoints2 = keypoints_pair1_518
    threshold = threshold_pair1

    confidence_scores1 = keypoints1[:, 3].astype(np.float64)
    confidence_scores2 = keypoints2[:, 3].astype(np.float64)

    # Filter by confidence score threshold
    valid_keypoints1 = confidence_scores1 >= threshold
    valid_keypoints2 = confidence_scores2 >= threshold

    # Ensure the same keypoints are valid in both images
    valid_keypoints = valid_keypoints1 & valid_keypoints2

    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float64)


    return filtered_keypoints1, filtered_keypoints2

def filter_keypoints_pair2_517_536(keypoints_pair2_517, keypoints_pair2_536, threshold_pair2):
    """
    Filter keypoints based on the confidence score threshold and keep only the keypoints with matching IDs.

    Args:
        keypoints1 (np.ndarray): First keypoints array with shape (N, 4) where N is the number of keypoints.
        keypoints2 (np.ndarray): Second keypoints array with shape (N, 4) where N is the number of keypoints.
        threshold (float): Confidence score threshold, keypoints with scores below this value will be removed.

    Returns:
        np.ndarray, np.ndarray: Filtered keypoints arrays for keypoints1 and keypoints2 with matching IDs.
    """
    keypoints1 = keypoints_pair2_517
    keypoints2 = keypoints_pair2_536
    threshold = threshold_pair2

    confidence_scores1 = keypoints1[:, 3].astype(np.float64)
    confidence_scores2 = keypoints2[:, 3].astype(np.float64)

    # Filter by confidence score threshold
    valid_keypoints1 = confidence_scores1 >= threshold
    valid_keypoints2 = confidence_scores2 >= threshold

    # Ensure the same keypoints are valid in both images
    valid_keypoints = valid_keypoints1 & valid_keypoints2

    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float64)

    return filtered_keypoints1, filtered_keypoints2

def filter_keypoints_pair3_517_520(keypoints_pair3_517, keypoints_pair3_520, threshold_pair3):
    """
    Filter keypoints based on the confidence score threshold and keep only the keypoints with matching IDs.

    Args:
        keypoints1 (np.ndarray): First keypoints array with shape (N, 4) where N is the number of keypoints.
        keypoints2 (np.ndarray): Second keypoints array with shape (N, 4) where N is the number of keypoints.
        threshold (float): Confidence score threshold, keypoints with scores below this value will be removed.

    Returns:
        np.ndarray, np.ndarray: Filtered keypoints arrays for keypoints1 and keypoints2 with matching IDs.
    """
    keypoints1 = keypoints_pair3_517
    keypoints2 = keypoints_pair3_520
    threshold = threshold_pair3

    confidence_scores1 = keypoints1[:, 3].astype(np.float64)
    confidence_scores2 = keypoints2[:, 3].astype(np.float64)

    # Filter by confidence score threshold
    valid_keypoints1 = confidence_scores1 >= threshold
    valid_keypoints2 = confidence_scores2 >= threshold

    # Ensure the same keypoints are valid in both images
    valid_keypoints = valid_keypoints1 & valid_keypoints2

    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float64)

    return filtered_keypoints1, filtered_keypoints2

# recoverPose
def estimate_relative_pose_pair1_517_518(cameras, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters - MAKE SURE THEY ARE FOR CORRECT CAMERAS
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[1]['distortion']

    # Redefine the keypoints
    keypoints1 = filtered_keypoints_pair1_517
    keypoints2 = filtered_keypoints_pair1_518

    keypoints1 = keypoints1.reshape(-1, 2).astype(np.float64)
    keypoints2 = keypoints2.reshape(-1, 2).astype(np.float64)

    _ ,E , R_recoverPose, t_recoverPose, _= cv2.recoverPose(keypoints1, keypoints2, K1,dist1, K2, dist2)

    return R_recoverPose, t_recoverPose

def estimate_relative_pose_pair2_517_536(cameras, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters - MAKE SURE THEY ARE FOR CORRECT CAMERAS
    K1 = cameras[0]['matrix']
    K2 = cameras[2]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[2]['distortion']

    # Redefine the keypoints
    keypoints1 = filtered_keypoints_pair2_517
    keypoints2 = filtered_keypoints_pair2_536

    keypoints1 = keypoints1.reshape(-1, 2).astype(np.float64)
    keypoints2 = keypoints2.reshape(-1, 2).astype(np.float64)

    _, E, R_recoverPose, t_recoverPose, _ = cv2.recoverPose(keypoints1, keypoints2, K1, dist1, K2, dist2)

    return R_recoverPose, t_recoverPose

def estimate_relative_pose_pair3_517_520(cameras, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters - MAKE SURE THEY ARE FOR CORRECT CAMERAS
    K1 = cameras[0]['matrix']
    K2 = cameras[3]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[3]['distortion']

    # Redefine the keypoints
    keypoints1 = filtered_keypoints_pair3_517
    keypoints2 = filtered_keypoints_pair3_520

    keypoints1 = keypoints1.reshape(-1, 2).astype(np.float64)
    keypoints2 = keypoints2.reshape(-1, 2).astype(np.float64)

    _, E, R_recoverPose, t_recoverPose, _ = cv2.recoverPose(keypoints1, keypoints2, K1, dist1, K2, dist2)

    return R_recoverPose, t_recoverPose

# Triangulation
def triangulate_points_pair1_517_518(R_recoverPose_pair1_517_518, t_recoverPose_pair1_517_518, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518):

    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[1]['distortion']

    filtered_keypoints1 = filtered_keypoints_pair1_517
    filtered_keypoints2 = filtered_keypoints_pair1_518
    R_recoverPose = R_recoverPose_pair1_517_518
    t_recoverPose = t_recoverPose_pair1_517_518

    # Step 1: Construct the projection matrices for the two cameras
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))) # [I|0] for the first camera
    P2 = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose))) # [R|t] for the second camera

    # Step 2: Unpack the keypoint matches to separate arrays
    points1 = np.float64(filtered_keypoints1).reshape(-1, 2).T
    points2 = np.float64(filtered_keypoints2).reshape(-1, 2).T

    # Step 3: Call the triangulatePoints function
    homogeneous_3d_points = cv2.triangulatePoints(P1, P2, points1, points2)

    # Step 4: Convert the points from homogeneous to euclidean coordinates
    euclidean_3d_points = homogeneous_3d_points[:3, :] / homogeneous_3d_points[3, :]

    # Step 5: Reshape the points to a Nx3 array
    triangulated_points = euclidean_3d_points.T

    return triangulated_points

def triangulate_points_pair2_517_536(R_recoverPose_pair2_517_536, t_recoverPose_pair2_517_536, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536):

    K1 = cameras[0]['matrix']
    K2 = cameras[2]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[2]['distortion']

    filtered_keypoints1 = filtered_keypoints_pair2_517
    filtered_keypoints2 = filtered_keypoints_pair2_536
    R_recoverPose = R_recoverPose_pair2_517_536
    t_recoverPose = t_recoverPose_pair2_517_536

    # Step 1: Construct the projection matrices for the two cameras
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))) # [I|0] for the first camera
    P2 = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose))) # [R|t] for the second camera

    # Step 2: Unpack the keypoint matches to separate arrays
    points1 = np.float64(filtered_keypoints1).reshape(-1, 2).T
    points2 = np.float64(filtered_keypoints2).reshape(-1, 2).T

    # Step 3: Call the triangulatePoints function
    homogeneous_3d_points = cv2.triangulatePoints(P1, P2, points1, points2)

    # Step 4: Convert the points from homogeneous to euclidean coordinates
    euclidean_3d_points = homogeneous_3d_points[:3, :] / homogeneous_3d_points[3, :]

    # Step 5: Reshape the points to a Nx3 array
    triangulated_points = euclidean_3d_points.T

    return triangulated_points

def triangulate_points_pair3_517_520(R_recoverPose_pair3_517_520, t_recoverPose_pair3_517_520, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520):

    K1 = cameras[0]['matrix']
    K2 = cameras[3]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[3]['distortion']

    filtered_keypoints1 = filtered_keypoints_pair3_517
    filtered_keypoints2 = filtered_keypoints_pair3_520
    R_recoverPose = R_recoverPose_pair3_517_520
    t_recoverPose = t_recoverPose_pair3_517_520

    # Step 1: Construct the projection matrices for the two cameras
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))) # [I|0] for the first camera
    P2 = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose))) # [R|t] for the second camera

    # Step 2: Unpack the keypoint matches to separate arrays
    points1 = np.float64(filtered_keypoints1).reshape(-1, 2).T
    points2 = np.float64(filtered_keypoints2).reshape(-1, 2).T

    # Step 3: Call the triangulatePoints function
    homogeneous_3d_points = cv2.triangulatePoints(P1, P2, points1, points2)

    # Step 4: Convert the points from homogeneous to euclidean coordinates
    euclidean_3d_points = homogeneous_3d_points[:3, :] / homogeneous_3d_points[3, :]

    # Step 5: Reshape the points to a Nx3 array
    triangulated_points = euclidean_3d_points.T

    return triangulated_points
def estimate_solvePnP(triangulated_points, filtered_keypoints2, cameras):
    # Using OpenCV's solvePnP function to estimate pose.

    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Processing data for solvePnP
    triangulated_points = triangulated_points[:, :3]
    triangulated_points = np.float64(triangulated_points)
    filtered_keypoints2 = np.float64(filtered_keypoints2)

    # Solving PnP
    ret, rvec, tvec_solvePnP = cv2.solvePnP(triangulated_points, filtered_keypoints2, K2, distCoeffs=dist)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnP, _ = cv2.Rodrigues(rvec)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnP, tvec_solvePnP

def estimate_solvePnPRansac(points_3d, filtered_keypoints2, cameras):
    # Using OpenCV's solvePnPRansac function to estimate pose.

    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Processing data for solvePnPRansac
    points_3d = points_3d[:, :3]
    points_3d = np.float64(points_3d)
    filtered_keypoints2 = np.float64(filtered_keypoints2)

    # Solving PnPRansac
    ret, rvec, tvec_solvePnPRansac, inliers = cv2.solvePnPRansac(points_3d, filtered_keypoints2, K2, distCoeffs=dist)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRansac, _ = cv2.Rodrigues(rvec)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRansac, tvec_solvePnPRansac
def estimate_solvePnPRefineLM_pair1_517_518(triangulated_points_pair1_517_518, filtered_keypoints_pair1_518, cameras):
    # Using OpenCV's solvePnPRefineLM function to estimate pose.

    # Extract intrinsic parameters
    K2 = cameras[1]['matrix']
    dist2 = cameras[1]['distortion']

    points_3d = triangulated_points_pair1_517_518
    filtered_keypoints2 = filtered_keypoints_pair1_518

    # Processing data for solvePnPRefineLM
    points_3d = points_3d[:, :3]
    points_3d = np.float64(points_3d)
    filtered_keypoints2 = np.float64(filtered_keypoints2)

    # Initial pose estimation with solvePnP
    _, rvec_init, tvec_init = cv2.solvePnP(points_3d, filtered_keypoints2, K2, distCoeffs=dist2)

    # Refine pose estimation with solvePnPRefineLM
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(points_3d, filtered_keypoints2, K2, distCoeffs=dist2, rvec=rvec_init, tvec=tvec_init)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRefineLM, _ = cv2.Rodrigues(rvec_refined)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRefineLM, tvec_refined

def estimate_solvePnPRefineLM_pair2_517_536( triangulated_points_pair2_517_536, filtered_keypoints_pair2_536, cameras):
    # Using OpenCV's solvePnPRefineLM function to estimate pose.

    # Extract intrinsic parameters
    K2 = cameras[2]['matrix']
    dist2 = cameras[2]['distortion']

    points_3d = triangulated_points_pair2_517_536
    filtered_keypoints2 = filtered_keypoints_pair2_536

    # Processing data for solvePnPRefineLM
    points_3d = points_3d[:, :3]
    points_3d = np.float64(points_3d)
    filtered_keypoints2 = np.float64(filtered_keypoints2)

    # Initial pose estimation with solvePnP
    _, rvec_init, tvec_init = cv2.solvePnP(points_3d, filtered_keypoints2, K2, distCoeffs=dist2)

    # Refine pose estimation with solvePnPRefineLM
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(points_3d, filtered_keypoints2, K2, distCoeffs=dist2,
                                                      rvec=rvec_init, tvec=tvec_init)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRefineLM, _ = cv2.Rodrigues(rvec_refined)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRefineLM, tvec_refined

def estimate_solvePnPRefineLM_pair3_517_520( triangulated_points_pair3_517_520, filtered_keypoints_pair3_520, cameras):
    # Using OpenCV's solvePnPRefineLM function to estimate pose.

    # Extract intrinsic parameters
    K2 = cameras[3]['matrix']
    dist2 = cameras[3]['distortion']

    points_3d = triangulated_points_pair3_517_520
    filtered_keypoints2 = filtered_keypoints_pair3_520

    # Processing data for solvePnPRefineLM
    points_3d = points_3d[:, :3]
    points_3d = np.float64(points_3d)
    filtered_keypoints2 = np.float64(filtered_keypoints2)

    # Initial pose estimation with solvePnP
    _, rvec_init, tvec_init = cv2.solvePnP(points_3d, filtered_keypoints2, K2, distCoeffs=dist2)

    # Refine pose estimation with solvePnPRefineLM
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(points_3d, filtered_keypoints2, K2, distCoeffs=dist2,rvec=rvec_init, tvec=tvec_init)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRefineLM, _ = cv2.Rodrigues(rvec_refined)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRefineLM, tvec_refined

# Reprojections
def calculate_reprojection_error_solvePnPRefine_pair1_517_518(triangulated_points_pair1_517_518, filtered_keypoints_pair1_518, R_solvePnPRefineLM_pair1_517_518, tvec_refined_pair1_517_518, cameras):

    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']
    filtered_keypoints2 = filtered_keypoints_pair1_518
    triangulated_points = triangulated_points_pair1_517_518
    R_recoverPose = R_solvePnPRefineLM_pair1_517_518
    t_recoverPose = tvec_refined_pair1_517_518

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, R_recoverPose,t_recoverPose, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape)

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error, projected_points

def calculate_reprojection_error_solvePnPRefine_pair2_517_536(triangulated_points_pair2_517_536, filtered_keypoints_pair2_536, R_solvePnPRefineLM_pair2_517_536, tvec_refined_pair2_517_536, cameras):

    K2 = cameras[2]['matrix']
    dist = cameras[2]['distortion']
    filtered_keypoints2 = filtered_keypoints_pair2_536
    triangulated_points = triangulated_points_pair2_517_536
    R_recoverPose = R_solvePnPRefineLM_pair2_517_536
    t_recoverPose = tvec_refined_pair2_517_536

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, R_recoverPose,t_recoverPose, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape)

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error

def calculate_reprojection_error_solvePnPRefine_pair3_517_520(triangulated_points_pair3_517_520, filtered_keypoints_pair3_520, R_solvePnPRefineLM_pair3_517_520, tvec_refined_pair3_517_520, cameras):

    K2 = cameras[3]['matrix']
    dist = cameras[3]['distortion']
    filtered_keypoints2 = filtered_keypoints_pair3_520
    triangulated_points = triangulated_points_pair3_517_520
    R_recoverPose = R_solvePnPRefineLM_pair3_517_520
    t_recoverPose = tvec_refined_pair3_517_520

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, R_recoverPose,t_recoverPose, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape)

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error

def calculate_reprojection_error_solvePnP(triangulated_points, filtered_keypoints2, R_solvePnP, tvec_solvePnP, cameras):
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, R_solvePnP,tvec_solvePnP, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape)

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error

def calculate_reprojection_error_solvePnPRansac(triangulated_points, filtered_keypoints2, R_solvePnPRansac, tvec_solvePnPRansac, cameras):
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, R_solvePnPRansac,tvec_solvePnPRansac, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape)

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error

def calculate_reprojection_error_solvePnPLM(triangulated_points, filtered_keypoints2, R_solvePnPRefineLM, tvec_refined, cameras):
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, R_solvePnPRefineLM, tvec_refined, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape) #  np.prod(squared_errors.shape) gives the total number of elements in the array

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error


def visualize_3d_points(triangulated_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = triangulated_points[:, 0]
    ys = triangulated_points[:, 1]
    zs = triangulated_points[:, 2]

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def visualize_reprojection_error(initial_points, reprojected_points):
    # Compute the Euclidean distance between initial points and reprojected points
    errors = np.linalg.norm(initial_points - reprojected_points, axis=1)

    # Plotting the reprojection error
    plt.figure()
    plt.hist(errors, bins='auto')
    plt.xlabel('Reprojection Error')
    plt.ylabel('Frequency')
    plt.title('Reprojection Error Distribution')
    plt.show()

def visualize_reprojection_error_heatmap(initial_points, reprojected_points):
    # Compute the Euclidean distance between initial points and reprojected points
    errors = np.linalg.norm(initial_points - reprojected_points, axis=1)

    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        initial_points[:, 0], initial_points[:, 1], bins=50
    )

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(
        heatmap.T, cmap='hot', origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
    )

    # Add a colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('Frequency')

    # Scatter plot of initial points with error as color
    scatter = ax.scatter(
        initial_points[:, 0], initial_points[:, 1],
        c=errors, cmap='cool', alpha=0.7
    )

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Reprojection Error Heatmap')

    # Show the plot
    plt.show()

def visualize_reprojection_error_scatter(initial_points, reprojected_points):
    # Compute the Euclidean distance between initial points and reprojected points
    errors = np.linalg.norm(initial_points - reprojected_points, axis=1)

    # Duplicate the errors for each point
    errors = np.repeat(errors, 2)

    # Duplicate the initial points for plotting
    points = initial_points.reshape(-1, 1)
    points = np.repeat(points, 2, axis=1)
    points = points.flatten()

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Plot the reprojection error as scatter points
    scatter = ax.scatter(points[::2], points[1::2], c=errors, cmap='cool', alpha=0.7)

    # Add a colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Reprojection Error')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Reprojection Error Scatter Plot')

    # Show the plot
    plt.show()

def getHomography(
        points1,
        points2,
):
    """

    :param points1:
    :param points2:

    :return:

    """
    h, _ = cv2.findHomography(
        srcPoints=points1,
        dstPoints=points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    return h






if __name__ == '__main__':
    parser = ArgumentParser(description="Pairwise camera calibration using OpenPose")

    parser.add_argument("--image_path_pair1_517_518", type=str, default="data/Calibration_sets/517_518/", help="Path to the directory containing input images")
    parser.add_argument("--image_path_pair2_517_536", type=str, default="data/Calibration_sets/517_536/",help="Path to the directory containing input images")
    parser.add_argument("--image_path_pair3_517_520", type=str, default="data/Calibration_sets/517_520/",help="Path to the directory containing input images")

    parser.add_argument("--calib_path", type=str, default="data/intrinsic/", help="Path to the directory containing calibration files")
    args = parser.parse_args()

    # Load camera calibration parameters for each camera
    cameras = load_camera_params(args.calib_path, 517,518,536,520) # pick which camera pair you are calibrating
    print('Intrinsics loaded!')

    # Load OpenPose
    params = dict()
    params["model_folder"] = "openpose/models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Load images and extract keypoints for each camera pair from cam1 to cam2,3,4
    # images_cam517 = sorted(glob(f"{args.image_path}/517*.png"))
    # images_cam518 = sorted(glob(f"{args.image_path}/518*.png"))
    # images_cam536 = sorted(glob(f"{args.image_path}/536*.png"))
    # images_cam520 = sorted(glob(f"{args.image_path}/520*.png"))

    images_pair1_517 = sorted(glob(f"{args.image_path_pair1_517_518}/517*.png"))
    images_pair1_518 = sorted(glob(f"{args.image_path_pair1_517_518}/518*.png"))

    images_pair2_517 = sorted(glob(f"{args.image_path_pair2_517_536}/517*.png"))
    images_pair2_536 = sorted(glob(f"{args.image_path_pair2_517_536}/536*.png"))

    images_pair3_517 = sorted(glob(f"{args.image_path_pair3_517_520}/517*.png"))
    images_pair3_520 = sorted(glob(f"{args.image_path_pair3_517_520}/520*.png"))

    keypoints_pair1_517, keypoints_pair1_518, confidences_pair1_517, confidences_pair1_518 = extract_keypoints_pair1_517_518(images_pair1_517, images_pair1_518, opWrapper)
    keypoints_pair2_517, keypoints_pair2_536, confidences_pair2_517, confidences_pair2_536 = extract_keypoints_pair2_517_536(images_pair2_517, images_pair2_536, opWrapper)
    keypoints_pair3_517, keypoints_pair3_520, confidences_pair3_517, confidences_pair3_520 = extract_keypoints_pair3_517_520(images_pair3_517,images_pair3_520, opWrapper)
    print('Keypoints extracted for all pairs!')

    # Filter keypoints based on confidence level and presence in both keypoint sets from cam1 to cam2,3,4
    threshold_pair1 = 0.7
    threshold_pair2 = 0.75
    threshold_pair3 = 0.7

    filtered_keypoints_pair1_517, filtered_keypoints_pair1_518 = filter_keypoints_pair1_517_518(keypoints_pair1_517, keypoints_pair1_518, threshold_pair1)
    filtered_keypoints_pair2_517, filtered_keypoints_pair2_536 = filter_keypoints_pair2_517_536(keypoints_pair2_517, keypoints_pair2_536, threshold_pair2)
    filtered_keypoints_pair3_517, filtered_keypoints_pair3_520 = filter_keypoints_pair3_517_520(keypoints_pair3_517, keypoints_pair3_520, threshold_pair3)
    print('Keypoints filtered for all pairs!')

    # Estimate relative pose for each camera pair
    R_recoverPose_pair1_517_518, t_recoverPose_pair1_517_518 = estimate_relative_pose_pair1_517_518(cameras, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518)
    R_recoverPose_pair2_517_536, t_recoverPose_pair2_517_536 = estimate_relative_pose_pair2_517_536(cameras, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536)
    R_recoverPose_pair3_517_520, t_recoverPose_pair3_517_520 = estimate_relative_pose_pair3_517_520(cameras, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520)
    print('recoverPose successful for all pairs!')

    """ Triangulation """#################
    # Assuming K1, K2, R, t, filtered_keypoints1, and filtered_keypoints2 are defined
    triangulated_points_pair1_517_518 = triangulate_points_pair1_517_518(R_recoverPose_pair1_517_518, t_recoverPose_pair1_517_518, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518)
    triangulated_points_pair2_517_536 = triangulate_points_pair2_517_536(R_recoverPose_pair2_517_536, t_recoverPose_pair2_517_536, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536)
    triangulated_points_pair3_517_520 = triangulate_points_pair3_517_520(R_recoverPose_pair3_517_520, t_recoverPose_pair3_517_520, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520)
    print('Triangulation successful for all pairs!')

    '''Estimate functions''' ######################
    #R_solvePnP, tvec_solvePnP = estimate_solvePnP(triangulated_points,filtered_keypoints2, cameras)
    #R_solvePnPRansac, tvec_solvePnPRansac = estimate_solvePnPRansac(triangulated_points, filtered_keypoints2, cameras)

    R_solvePnPRefineLM_pair1_517_518, tvec_refined_pair1_517_518 = estimate_solvePnPRefineLM_pair1_517_518(triangulated_points_pair1_517_518, filtered_keypoints_pair1_518, cameras)
    R_solvePnPRefineLM_pair2_517_536, tvec_refined_pair2_517_536 = estimate_solvePnPRefineLM_pair2_517_536(triangulated_points_pair2_517_536, filtered_keypoints_pair2_536, cameras)
    R_solvePnPRefineLM_pair3_517_520, tvec_refined_pair3_517_520 = estimate_solvePnPRefineLM_pair3_517_520(triangulated_points_pair3_517_520, filtered_keypoints_pair3_520, cameras)
    print('solvePnPRefineLM successful for all cameras!')

    '''Reprojection error''' #######################
    reprojection_error_solvePnPRefine_pair1_517_518, reprojected_points = calculate_reprojection_error_solvePnPRefine_pair1_517_518(triangulated_points_pair1_517_518, filtered_keypoints_pair1_518, R_solvePnPRefineLM_pair1_517_518, tvec_refined_pair1_517_518, cameras)
    reprojection_error_solvePnPRefine_pair2_517_536 = calculate_reprojection_error_solvePnPRefine_pair2_517_536(triangulated_points_pair2_517_536, filtered_keypoints_pair2_536, R_solvePnPRefineLM_pair2_517_536, tvec_refined_pair2_517_536, cameras)
    reprojection_error_solvePnPRefine_pair3_517_520 = calculate_reprojection_error_solvePnPRefine_pair3_517_520(triangulated_points_pair3_517_520, filtered_keypoints_pair3_520, R_solvePnPRefineLM_pair3_517_520, tvec_refined_pair3_517_520, cameras)


    #print('Reprojection error recoverPose:', reprojection_error_recov)

    # reprojection_error_pnp = calculate_reprojection_error_solvePnP(triangulated_points, filtered_keypoints2, R_solvePnP, tvec_solvePnP, cameras)
    # print('Reprojection error solvepnp:', reprojection_error_pnp)
    #
    # reprojection_error_PnPRansac = calculate_reprojection_error_solvePnPRansac(triangulated_points, filtered_keypoints2, R_solvePnPRansac, tvec_solvePnPRansac, cameras)
    # print('Reprojection error solvepnpRansac:', reprojection_error_PnPRansac)
    #
    # reprojection_error_PnPLM = calculate_reprojection_error_solvePnPLM(triangulated_points, filtered_keypoints2, R_solvePnPRefineLM, tvec_refined, cameras)
    # print('Reprojection error solvepnpLM:', reprojection_error_PnPLM)

    # Scaling with the real distance between sensors
    real_distance_cm_pair1 = 592
    real_distance_cm_pair2 = 762
    real_distance_cm_pair3 = 481

    estimated_distance_solvePnPRefine_pair1_517_518 = np.linalg.norm(tvec_refined_pair1_517_518)
    estimated_distance_solvePnPRefine_pair2_517_536 = np.linalg.norm(tvec_refined_pair2_517_536)
    estimated_distance_solvePnPRefine_pair3_517_520 = np.linalg.norm(tvec_refined_pair3_517_520)

    scale_factor_pair1_517_518 = real_distance_cm_pair1 / estimated_distance_solvePnPRefine_pair1_517_518
    scale_factor_pair2_517_536 = real_distance_cm_pair2 / estimated_distance_solvePnPRefine_pair2_517_536
    scale_factor_pair3_517_520 = real_distance_cm_pair3 / estimated_distance_solvePnPRefine_pair3_517_520



    T_scaled_cm_pair1_517_518 = scale_factor_pair1_517_518 * tvec_refined_pair1_517_518
    T_scaled_cm_pair2_517_536 = scale_factor_pair2_517_536 * tvec_refined_pair2_517_536
    T_scaled_cm_pair3_517_520 = scale_factor_pair3_517_520 * tvec_refined_pair3_517_520

    print('Scale factor pair1: ', scale_factor_pair1_517_518, '/ Scale factor pair2:', scale_factor_pair2_517_536, '/ Scale factor pair3:', scale_factor_pair3_517_520)
    print("Pair 1 - translation from camera 517 to 518 - UBIQISENSE PIPELINE:", "[-483.67437729 -122.77347367  348.15576036] (cm) ")
    print("Pair 1 - translation from camera 517 to 518 - solvePnPRefineLM:", T_scaled_cm_pair1_517_518.ravel())
    print('')

    print("Pair 2 - translation from camera 517 to 536 - UBIQISENSE PIPELINE:", "[-188.76428849 -211.59882113  731.79469696] (cm) ")
    print("Pair 2 - translation from camera 517 to 536 - solvePnPRefineLM:", T_scaled_cm_pair2_517_536.ravel())
    print('')

    print("Pair 3 - translation from camera 517 to 520 - UBIQISENSE PIPELINE:", "[306.34257461 -94.6791296  380.83349388] (cm) ")
    print("Pair 3 - translation from camera 517 to 520 - solvePnPRefineLM:", T_scaled_cm_pair3_517_520.ravel())
    print('')
    # Results
    rot_decomp_pair1_recoverPose, _, _, _, _, _ = cv2.RQDecomp3x3(R_recoverPose_pair1_517_518)
    rot_decomp_pair2_recoverPose, _, _, _, _, _ = cv2.RQDecomp3x3(R_recoverPose_pair2_517_536)
    rot_decomp_pair3_recoverPose, _, _, _, _, _ = cv2.RQDecomp3x3(R_recoverPose_pair3_517_520)

    rot_decomp_pair1_solvePnPRefine, _, _, _, _, _ = cv2.RQDecomp3x3(R_solvePnPRefineLM_pair1_517_518)
    rot_decomp_pair2_solvePnPRefine, _, _, _, _, _ = cv2.RQDecomp3x3(R_solvePnPRefineLM_pair2_517_536)
    rot_decomp_pair3_solvePnPRefine, _, _, _, _, _ = cv2.RQDecomp3x3(R_solvePnPRefineLM_pair3_517_520)
    num_keypoints_pair1 = filtered_keypoints_pair1_517.shape[0]
    num_keypoints_pair2 = filtered_keypoints_pair2_517.shape[0]
    num_keypoints_pair3 = filtered_keypoints_pair3_517.shape[0]

    print("Pair 1 - rotation from camera 517 to 518 - UBIQISENSE PIPELINE:", "( 47.95943056 -67.13253629 -62.56659299) (deg.) ")
    print("Pair 1 - rotation from camera 517 to 518 - recoverPose:", rot_decomp_pair1_recoverPose)
    print("Pair 1 - rotation from camera 517 to 518 - solvePnPRefineLM:", rot_decomp_pair1_solvePnPRefine)
    print('Number of keypoints:', num_keypoints_pair1, "/ Threshold: ", threshold_pair1, "/ Reprojection error:", reprojection_error_solvePnPRefine_pair1_517_518 )
    print('')

    print("Pair 2 - rotation from camera 517 to 536 - UBIQISENSE PIPELINE:","( 150.39854946   13.33519403 -177.79742376) (deg.) ")
    print("Pair 2 - rotation from camera 517 to 536 - recoverPose:", rot_decomp_pair2_recoverPose)
    print("Pair 2 - rotation from camera 517 to 536 - solvePnPRefineLM:", rot_decomp_pair2_solvePnPRefine)
    print('Number of keypoints:', num_keypoints_pair2, "/ Threshold: ", threshold_pair2, "/ Reprojection error:", reprojection_error_solvePnPRefine_pair2_517_536)
    print('')

    print("Pair 3 - rotation from camera 517 to 520 - UBIQISENSE PIPELINE:","( 132.62900666  69.60734076 151.97386786) (deg.) ")
    print("Pair 3 - rotation from camera 517 to 520 - recoverPose:", rot_decomp_pair3_recoverPose)
    print("Pair 3 - rotation from camera 517 to 520 - solvePnPRefineLM:", rot_decomp_pair3_solvePnPRefine)
    print('Number of keypoints:', num_keypoints_pair3, "/ Threshold: ", threshold_pair3, "/ Reprojection error:", reprojection_error_solvePnPRefine_pair3_517_520)

    #Visualize
    # visualize_3d_points(triangulated_points_pair1_517_518)
    # visualize_3d_points(triangulated_points_pair2_517_536)
    # visualize_3d_points(triangulated_points_pair3_517_520)

    #print(reprojected_points) remove a dim!!
    # visualize_reprojection_error(filtered_keypoints_pair1_518, reprojected_points)
    #visualize_reprojection_error_heatmap(filtered_keypoints_pair1_518, reprojected_points)
    #visualize_reprojection_error_scatter(filtered_keypoints_pair1_518, reprojected_points)


