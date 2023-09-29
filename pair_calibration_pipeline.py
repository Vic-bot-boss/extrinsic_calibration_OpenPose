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
import g2o
from g2o import RobustKernelHuber
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

### Camera parameters
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

### OpenPose keypoints extraction
def extract_keypoints(images_cam1, images_cam2, opWrapper):
    """
    Extract keypoints and confidences for each image
    """

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

        # Create an array of IDs from 1 to 25 for camera 2
        ids_cam1 = np.arange(1, 26).reshape(-1, 1)

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1), ids_cam1))  # append strings and confidence scores to keypoints
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

        # Create an array of IDs from 1 to 25 for camera 2
        ids_cam2 = np.arange(1, 26).reshape(-1, 1)

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1), ids_cam2))  # append strings and confidence scores to keypoints
        keypoints_cam2.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam2.append(average_confidence)

    # Transform the list of keypoints into arrays
    keypoints1 = np.vstack(keypoints_cam1)
    keypoints2 = np.vstack(keypoints_cam2)
    print(keypoints1)
    confidences1 = keypoints1[:, 3].astype(np.float64)
    confidences2 = keypoints2[:, 3].astype(np.float64)

    return keypoints1, keypoints2, confidences1, confidences2
def extract_keypoints_common(images_cam1, images_cam2, images_cam3, images_cam4, opWrapper):
    """
    Extract keypoints and confidences for each image
    """

    keypoints_cam1 = []
    keypoints_cam2 = []
    keypoints_cam3 = []
    keypoints_cam4 = []
    confidences_cam1 = []
    confidences_cam2 = []
    confidences_cam3 = []
    confidences_cam4 = []

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

    # handle images from camera 3
    for image in images_cam3:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]  # X, Y
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()  # Confidence score each keypoint

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam3.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam3.append(average_confidence)

    # handle images from camera 4
    for image in images_cam4:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints[0][:, :2]
        confidence_scores = datum.poseKeypoints[0][:, 2].flatten()

        keypoints_with_strings = np.hstack((keypoints, strings, confidence_scores.reshape(-1, 1)))  # append strings and confidence scores to keypoints
        keypoints_cam4.append(keypoints_with_strings)

        average_confidence = datum.poseScores[0]  # Average confidence
        confidences_cam4.append(average_confidence)



    # Transform the list of keypoints into arrays
    keypoints1 = np.vstack(keypoints_cam1)
    keypoints2 = np.vstack(keypoints_cam2)
    keypoints3 = np.vstack(keypoints_cam3)
    keypoints4 = np.vstack(keypoints_cam4)

    confidences1 = np.array(confidences_cam1)
    confidences2 = np.array(confidences_cam2)
    confidences3 = np.array(confidences_cam3)
    confidences4 = np.array(confidences_cam4)

    return keypoints1, keypoints2, keypoints3, keypoints4, confidences1, confidences2, confidences3, confidences4


### Charuco detection
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


### Filtering keypoints
def filter_keypoints(keypoints1, keypoints2, threshold):

    """
    Filter keypoints based on the confidence score threshold and keep only the keypoints with matching IDs.

    Args:
        keypoints1 (np.ndarray): First keypoints array with shape (N, 4) where N is the number of keypoints.
        keypoints2 (np.ndarray): Second keypoints array with shape (N, 4) where N is the number of keypoints.
        threshold (float): Confidence score threshold, keypoints with scores below this value will be removed.

    Returns:
        np.ndarray, np.ndarray: Filtered keypoints arrays for keypoints1 and keypoints2 with matching IDs.
    """

    # Extract confidence scores
    confidence_scores1 = keypoints1[:, 3].astype(np.float64)
    confidence_scores2 = keypoints2[:, 3].astype(np.float64)

    # Filter by confidence score threshold
    valid_keypoints1 = confidence_scores1 >= threshold
    valid_keypoints2 = confidence_scores2 >= threshold

    # Ensure the same keypoints are valid in both images
    valid_keypoints = valid_keypoints1 & valid_keypoints2

    # Filtered keypoints without confidence scores
    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float64)

    # Filtered confidence scores
    filtered_scores1 = keypoints1[valid_keypoints, 3].astype(np.float64)
    filtered_scores2 = keypoints2[valid_keypoints, 3].astype(np.float64)

    return filtered_keypoints1, filtered_keypoints2, filtered_scores1, filtered_scores2
def filter_keypoints_common(keypoints1, keypoints2, keypoints3, keypoints4, threshold):

    """
    Filter keypoints based on the confidence score threshold and keep only the keypoints with matching IDs.

    Args:
        keypoints1 (np.ndarray): First keypoints array with shape (N, 4) where N is the number of keypoints.
        keypoints2 (np.ndarray): Second keypoints array with shape (N, 4) where N is the number of keypoints.
        threshold (float): Confidence score threshold, keypoints with scores below this value will be removed.

    Returns:
        np.ndarray, np.ndarray: Filtered keypoints arrays for keypoints1 and keypoints2 with matching IDs.
    """

    # Extract confidence scores
    confidence_scores1 = keypoints1[:, 3].astype(np.float64)
    confidence_scores2 = keypoints2[:, 3].astype(np.float64)
    confidence_scores3 = keypoints3[:, 3].astype(np.float64)
    confidence_scores4 = keypoints4[:, 3].astype(np.float64)

    # Filter by confidence score threshold
    valid_keypoints1 = confidence_scores1 >= threshold
    valid_keypoints2 = confidence_scores2 >= threshold
    valid_keypoints3 = confidence_scores3 >= threshold
    valid_keypoints4 = confidence_scores4 >= threshold

    # Ensure the same keypoints are valid in both images
    valid_keypoints = valid_keypoints1 & valid_keypoints2 & valid_keypoints3 & valid_keypoints4

    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints3 = keypoints3[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints4 = keypoints4[valid_keypoints, :2].astype(np.float64)

    return filtered_keypoints1, filtered_keypoints2, filtered_keypoints3, filtered_keypoints4


### Pose estimation functions
def estimate_recoverPose(cam1, cam2, keypoints1, keypoints2):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters - MAKE SURE THEY ARE FOR CORRECT CAMERAS
    K1 = cam1['matrix']
    K2 = cam2['matrix']
    dist1 = cam1['distortion']
    dist2 = cam2['distortion']

    keypoints1 = keypoints1.reshape(-1, 2).astype(np.float64)
    keypoints2 = keypoints2.reshape(-1, 2).astype(np.float64)

    # Find the Essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(keypoints1, keypoints2, K1, method=cv2.RANSAC)

    # Recover the relative pose using the Essential matrix
    retval, E, R_recoverPose, t_recoverPose, mask_out = cv2.recoverPose(
        keypoints1, keypoints2, K1, dist1, K2, dist2, E,
    method=cv2.RANSAC, #cv2.RANSAC
    prob=0.999,
    threshold=3.0,
    mask=mask)


    return R_recoverPose, t_recoverPose

def estimate_solvePnP(triangulated_points, filtered_keypoints2, cam2):
    # Using OpenCV's solvePnP function to estimate pose.

    # Extract intrinsic parameters
    K2 = cam2['matrix']
    dist = cam2['distortion']

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

def estimate_solvePnPRansac(triangulated_points, filtered_keypoints2, cam2):
    # Using OpenCV's solvePnPRansac function to estimate pose.

    # Extract intrinsic parameters
    K2 = cam2['matrix']
    dist = cam2['distortion']

    # Processing data for solvePnP
    triangulated_points = triangulated_points[:, :3]
    triangulated_points = np.float64(triangulated_points)
    filtered_keypoints2 = np.float64(filtered_keypoints2)


    # Solving PnPRansac
    ret, rvec, tvec_solvePnPRansac, inliers = cv2.solvePnPRansac(triangulated_points, filtered_keypoints2, K2, distCoeffs=dist)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRansac, _ = cv2.Rodrigues(rvec)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRansac, tvec_solvePnPRansac
def estimate_solvePnPRefineLM(triangulated_points, filtered_keypoints2, cam2):
    # Using OpenCV's solvePnPRefineLM function to estimate pose.

    # Extract intrinsic parameters
    K2 = cam2['matrix']
    dist = cam2['distortion']

    # Processing data for solvePnP
    triangulated_points = triangulated_points[:, :3]
    triangulated_points = np.float64(triangulated_points)
    filtered_keypoints2 = np.float64(filtered_keypoints2)

    # Initial pose estimation with solvePnP
    _, rvec_init, tvec_init = cv2.solvePnP(triangulated_points, filtered_keypoints2, K2, distCoeffs=dist)

    # Refine pose estimation with solvePnPRefineLM
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(triangulated_points, filtered_keypoints2, K2, distCoeffs=dist, rvec=rvec_init, tvec=tvec_init)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRefineLM, _ = cv2.Rodrigues(rvec_refined)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRefineLM, tvec_refined


### Triangulation
def triangulate_points(cam1, cam2, filtered_keypoints1, filtered_keypoints2, R_recoverPose, t_recoverPose):
    # Extract intrinsic parameters - MAKE SURE THEY ARE FOR CORRECT CAMERAS
    K1 = cam1['matrix']
    K2 = cam2['matrix']
    dist1 = cam1['distortion']
    dist2 = cam2['distortion']

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


### Reprojections
def calculate_reprojection_error(cam2, triangulated_points, filtered_keypoints_cam2, rot, tvec):
    # computes the Root Mean Square (RMS) error between the original and projected 2D keypoints
    # Pixel units

    K2 = cam2['matrix']
    dist = cam2['distortion']


    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, rot, tvec, K2, dist)

    # Compute the difference between the original and projected 2D points
    reprojection_error = filtered_keypoints_cam2 - projected_points.reshape(-1, 2)

    # Square the errors
    squared_errors = reprojection_error**2

    # Compute the sum of the squared errors
    sum_of_squared_errors = np.sum(squared_errors)

    # Compute the mean squared error
    mean_squared_error = sum_of_squared_errors / np.prod(squared_errors.shape)

    # Compute the root mean square (RMS) reprojection error
    rms_reprojection_error = np.sqrt(mean_squared_error)

    return rms_reprojection_error

def visualize_reprojections(image, cam, R, t, points_3d, original_keypoints, modified_path):
    camera_matrix = cam['matrix']
    dist_coeffs = cam['distortion']

    # Project the 3D points onto the 2D image plane
    projected_points, _ = cv2.projectPoints(points_3d, R, t, camera_matrix, dist_coeffs)

    # Convert projected points to a suitable format
    projected_points = projected_points.reshape(-1, 2)

    # Create a copy of the original image to draw on
    vis_image = image.copy()

    # Draw the original keypoints in blue, the projected points in red, and a line between them in green
    for i in range(len(original_keypoints)):
        cv2.circle(vis_image, tuple(np.int32(original_keypoints[i])), 3, (255, 0, 0), -1)
        cv2.circle(vis_image, tuple(np.int32(projected_points[i])), 3, (0, 0, 255), -1)
        cv2.line(vis_image, tuple(np.int32(original_keypoints[i])), tuple(np.int32(projected_points[i])), (0, 255, 0), 1)

    # Save the original and modified images

    cv2.imwrite(modified_path, vis_image)
    return vis_image

def calculate_reprojection_error_euclidean(cam2, triangulated_points, filtered_keypoints_cam2, rot, tvec):
    # Euclidean distance
    K2 = cam2['matrix']
    dist = cam2['distortion']

    # Project the 3D object points back onto the image plane
    projected_points, _ = cv2.projectPoints(triangulated_points, rot, tvec, K2, dist)
    projected_points = projected_points.reshape(-1, 2)

    # Compute the Euclidean distances between the original and projected 2D points
    euclidean_distances = np.linalg.norm(filtered_keypoints_cam2 - projected_points, axis=1)

    # Compute the mean of the Euclidean distances
    mean_euclidean_distance = np.mean(euclidean_distances)

    return mean_euclidean_distance

### Visualization
def visualize_3d_points(triangulated_points, title="Triangulated Points"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = triangulated_points[:, 0]
    ys = triangulated_points[:, 1]
    zs = triangulated_points[:, 2]

    # Color mapping based on Z value (depth)
    colors = zs - min(zs)
    colors = colors / max(colors)
    colmap = plt.cm.get_cmap('jet')  # 'jet' color map ranges from blue to red
    colors = colmap(colors)

    # Point size modulation based on depth for a pseudo-depth effect
    sizes = 50 - (colors[:, -1] * 45)

    scatter = ax.scatter(xs, ys, zs, c=colors, s=sizes, depthshade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Optional: Set consistent axis limits for comparability
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])

    ax.grid(True)

    # Display colorbar to represent depth
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, orientation='vertical')
    cbar.set_label('Depth (Z value)')

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

def scale_translation(tvec, real_distance_cm):
    # print(tvec)
    estimated_distance = np.linalg.norm(tvec)
    # print(estimated_distance)
    scale_factor = real_distance_cm / estimated_distance

    # Scale the translation vector
    tvec = tvec * scale_factor

    return scale_factor, tvec

def decompose_rotation(R):
    rot_decomp, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    return rot_decomp

def invert_transformation(R, t):
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R_recoverPose_pair2
    T[0:3, 3] = t_recoverPose_pair2.ravel()
    T[3, 3] = 1
    T_inv = np.linalg.inv(T)
    R_inv = T_inv[:3, :3]  # Rotation matrix is the top-left 3x3 submatrix
    t_inv = T_inv[:3, 3]  # Translation vector is the right-most column

    return R_inv, t_inv
def draw_openpose_skeleton(image, keypoints):
    # Define colors for drawing keypoints
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85), (255, 85, 85), (255, 85, 0),
        (255, 170, 0), (255, 170, 85), (255, 170, 170), (255, 255, 170),
        (170, 255, 170), (85, 255, 170), (85, 255, 255), (170, 255, 255)
    ]

    # Iterate over keypoints
    for point in keypoints:
        # Get the x, y coordinates of the keypoint
        x = int(float(point[0]))
        y = int(float(point[1]))

        # Draw circle for the keypoint
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    return image


def overlay_images(image_paths, keypoints):
    transparency = 0.5
    # Load the first image as the base image
    base_image = cv2.imread(image_paths[0])

    # Iterate over the remaining images
    for path in image_paths[1:]:
        # Read the next image
        next_image = cv2.imread(path)

        # Resize the next image to match the base image size
        next_image = cv2.resize(next_image, (base_image.shape[1], base_image.shape[0]))

        # Apply transparency to the next image
        overlay = cv2.addWeighted(base_image, 1 - transparency, next_image, transparency, 0)

        # Set the overlay as the new base image for the next iteration
        base_image = overlay

    # Draw the OpenPose skeleton on the final image
    final_image_with_skeleton = draw_openpose_skeleton(base_image, keypoints)

    # Display the overlayed image with skeleton
    cv2.imshow('Overlay with Skeleton', final_image_with_skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_reprojection_error_scatter(initial_points, reprojected_points, outliers=None):
    """
    Visualizes the reprojection error as a scatter plot.

    Parameters:
    - initial_points: The original 2D keypoints.
    - reprojected_points: The reprojected 2D keypoints after triangulation and projection.
    - outliers: A boolean array indicating which points are outliers.
    """

    # Compute the Euclidean distance between initial points and reprojected points
    errors = np.linalg.norm(initial_points - reprojected_points, axis=1)

    fig, ax = plt.subplots()

    # Plot the initial points
    ax.scatter(initial_points[:, 0], initial_points[:, 1], color='blue', label='Original Points')

    # Plot the reprojected points with color indicating error magnitude
    scatter = ax.scatter(reprojected_points[:, 0], reprojected_points[:, 1], c=errors, cmap='coolwarm', s=50,
                         label='Reprojected Points')

    # If outlier information is provided, modify the size/shape of outliers
    if outliers is not None:
        ax.scatter(initial_points[outliers, 0], initial_points[outliers, 1], color='red', marker='x', label='Outliers')

    # Add a colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('Reprojection Error')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Reprojection Error Scatter Plot')
    ax.legend()

    # Show the plot
    plt.show()

# BA and other code improvements
def pipeline_for_pair(images_path_1, images_path_2, cam1, cam2, opWrapper, threshold, real_distance_cm):
    # Load images
    images_1 = sorted(glob(images_path_1))
    images_2 = sorted(glob(images_path_2))

    # Extract keypoints
    keypoints_1, keypoints_2, confidences_1, confidences_2 = extract_keypoints(images_1, images_2, opWrapper)

    # Filter keypoints
    filtered_keypoints_1, filtered_keypoints_2 = filter_keypoints(keypoints_1, keypoints_2, threshold)

    # Estimate relative pose
    R_recoverPose, t_recoverPose = estimate_recoverPose(cam1, cam2, filtered_keypoints_1, filtered_keypoints_2)

    # Triangulate the points
    points = triangulate_points(cam1, cam2, filtered_keypoints_1, filtered_keypoints_2, R_recoverPose, t_recoverPose)

    # Compute reprojection error
    reproj_error = calculate_reprojection_error(cam2, points, filtered_keypoints_2, R_recoverPose, t_recoverPose)

    # Scale translation
    scale_factor, scaled_translation = scale_translation(t_recoverPose, real_distance_cm)

    # Decompose rotation matrix
    rot_decomp = decompose_rotation(R_recoverPose)

    # Number of keypoints
    num_keypoints = len(filtered_keypoints_1)

    return {
        "reproj_error": reproj_error,
        "scale_factor": scale_factor,
        "scaled_translation": scaled_translation,
        "rot_decomp": rot_decomp,
        "num_keypoints": num_keypoints
    }

class BundleAdjustment_monocular(g2o.SparseOptimizer):
    def __init__(self, solver_type, linear_solver):
        super().__init__()
        solver_block = solver_type(linear_solver())
        solver = g2o.OptimizationAlgorithmLevenberg(solver_block)
        super().set_algorithm(solver)
        self.set_verbose(True)  # Add this line to enable verbosity
        # print(dir(g2o))

    def optimize(self, max_iterations):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, fixed=False):
        print("Input Pose:", pose)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(pose_id * 2)

        # Convert Isometry3d to SE3Quat
        se3quat = g2o.SE3Quat(pose.rotation(), pose.translation())

        v_se3.set_estimate(se3quat)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

        print("Isometry3d Pose:", pose)
        print("SE3Quat Pose:", se3quat)
    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, measurement, information, robust_kernel_threshold):
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(np.array(measurement).reshape(2, 1))
        edge.set_information(information)

        if robust_kernel_threshold is not None:
            robust_kernel = g2o.RobustKernelHuber(robust_kernel_threshold)
            edge.set_robust_kernel(robust_kernel)

        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        vertex = self.vertex(point_id * 2 + 1)
        if isinstance(vertex, g2o.VertexPointXYZ):
            return np.array(vertex.estimate())
        else:
            print(f"Warning: Point with ID {point_id} was not retrieved correctly!")
            return None

class BundleAdjustment_stereo_1st_try(g2o.SparseOptimizer):
    def __init__(self, solver_type, linear_solver):
        super().__init__()
        solver_block = solver_type(linear_solver())
        solver = g2o.OptimizationAlgorithmLevenberg(solver_block)
        super().set_algorithm(solver)
        self.set_verbose(True)  # Add this line to enable verbosity
        # print(dir(g2o))

    def optimize(self, max_iterations):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(cam[0, 0], cam[1, 1], cam[0, 2], cam[1, 2], 481)  # baseline set to 0 for now

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)
    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, measurement, information, robust_kernel_threshold):
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(np.array(measurement).reshape(2, 1))
        edge.set_information(information)

        if robust_kernel_threshold is not None:
            robust_kernel = g2o.RobustKernelHuber(robust_kernel_threshold)
            edge.set_robust_kernel(robust_kernel)

        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        vertex = self.vertex(point_id * 2 + 1)
        if isinstance(vertex, g2o.VertexPointXYZ):
            return np.array(vertex.estimate())
        else:
            print(f"Warning: Point with ID {point_id} was not retrieved correctly!")
            return None

# class L2RegularizationEdge(g2o.BaseUnaryEdge):
#     def __init__(self, lambda_reg):
#         super().__init__()
#         self.lambda_reg = lambda_reg
#
#     def computeError(self):
#         # Assuming a 6D pose vertex (x, y, z, roll, pitch, yaw)
#         self._error[0] = self.lambda_reg * self._vertices[0].estimate().to_vector().norm()
#

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, solver_type, linear_solver):
        super().__init__()
        solver_block = solver_type(linear_solver())
        solver = g2o.OptimizationAlgorithmLevenberg(solver_block) # Levenberg(solver_block) # GaussNewton(solver_block) # Dogleg(solver_block)

        # # Set convergence criteria
        # stop_threshold = 1e-6
        # max_iterations = 100
        # solver.setStopThreshold(stop_threshold)
        # solver.setMaxIterations(max_iterations)


        super().set_algorithm(solver)
        self.set_verbose(True)

    def optimize(self, max_iterations):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(cam[0, 0], cam[1, 1], cam[0, 2], cam[1, 2], 481)  # including baseline as distance between cameras
        # Print camera parameters
        print("Camera Parameters:")
        print(f"fx: {cam[0, 0]}")
        print(f"fy: {cam[1, 1]}")
        print(f"cx: {cam[0, 2]}")
        print(f"cy: {cam[1, 2]}")
        print(f"baseline: {100}")
        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, measurement, information, robust_kernel_threshold):
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(np.array(measurement).reshape(2, 1))
        edge.set_information(information)

        # Print out information about the edge
        print(f"Adding edge between point {point_id} and pose {pose_id}")
        print(f"Measurement: {measurement}")
        print(f"Information matrix: {information}")

        if robust_kernel_threshold is not None:
            robust_kernel = g2o.RobustKernelHuber(robust_kernel_threshold)
            edge.set_robust_kernel(robust_kernel)

        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        vertex = self.vertex(point_id * 2 + 1)
        if isinstance(vertex, g2o.VertexPointXYZ):
            return np.array(vertex.estimate())
        else:
            print(f"Warning: Point with ID {point_id} was not retrieved correctly!")
            return None

def perform_bundle_adjustment_with_class(points, cam_matrices, rotations, translations, keypoints_list,
                                         num_iterations, robust_kernel_threshold, information_matrices_list,
                                         solver_type, linear_solver):
    ba = BundleAdjustment(solver_type=solver_type, linear_solver=linear_solver)

    for i, (R, t) in enumerate(zip(rotations, translations)):
        pose = g2o.Isometry3d(R, t)
        fixed = (i == 0)
        ba.add_pose(pose_id=i, pose=pose, cam=cam_matrices[i], fixed=fixed)

    for point_id, point in enumerate(points):
        ba.add_point(point_id=point_id, point=point)
        for cam_id in [0, 1]:
            measurement = keypoints_list[cam_id][point_id]
            #information = information_matrices_list[cam_id][point_id]
            information = np.identity(2)
            ba.add_edge(point_id=point_id, pose_id=cam_id, measurement=measurement,
                        information=information, robust_kernel_threshold=robust_kernel_threshold)

    ba.optimize(max_iterations=num_iterations)

    optimized_poses = [ba.get_pose(pose_id=i) for i in range(len(rotations))]
    optimized_points = [np.array(ba.get_point(point_id=i)).reshape(-1) for i in range(len(points)) if ba.get_point(point_id=i) is not None]

    final_error = ba.chi2()
    print("Final error after optimization:", final_error)
    return optimized_points, optimized_poses


def perform_bundle_adjustment_with_class_monocular(points, cam_matrices, rotations, translations, keypoints_list,
                                                    num_iterations, robust_kernel_threshold, information_matrices_list,
                                                    solver_type, linear_solver):
    # Create an instance of the HyperparamBundleAdjustment class
    ba = BundleAdjustment(solver_type=solver_type, linear_solver=linear_solver)

    # Add camera poses
    for i, (R, t) in enumerate(zip(rotations, translations)):
        pose = g2o.Isometry3d(R, t)
        #fx, fy, cx, cy = cam_matrices[i][0, 0], cam_matrices[i][1, 1], cam_matrices[i][0, 2], cam_matrices[i][1, 2]
        # baseline = 0  # Modify as needed
        fixed = (i == 0)
        ba.add_pose(pose_id=i, pose=pose, fixed=fixed)

    # Add 3D points and their observations
    for point_id, point in enumerate(points):
        ba.add_point(point_id=point_id, point=point)
        for cam_id in [0, 1]:
            measurement = keypoints_list[cam_id][point_id]
            #information = np.identity(2)
            information = information_matrices_list[cam_id][point_id]
            ba.add_edge(point_id=point_id, pose_id=cam_id, measurement=measurement, information=information, robust_kernel_threshold=robust_kernel_threshold)

    # Optimize
    ba.optimize(max_iterations=num_iterations)

    # Extract the optimized camera poses and 3D points
    optimized_poses = [ba.get_pose(pose_id=i) for i in range(len(rotations))]
    optimized_points = [np.array(ba.get_point(point_id=i)).reshape(-1) for i in range(len(points)) if ba.get_point(point_id=i) is not None]

    final_error = ba.chi2()
    print("Final error after optimization:", final_error)
    return optimized_points, optimized_poses
def perform_bundle_adjustment_with_class_stereo_1st_try(points, cam_matrices, rotations, translations, keypoints_list,
                                                    num_iterations, robust_kernel_threshold, information_matrices_list,
                                                    solver_type, linear_solver):
    # Create an instance of the BundleAdjustmentStereo class
    ba = BundleAdjustment(solver_type=solver_type, linear_solver=linear_solver)

    # Add camera poses
    for i, (R, t) in enumerate(zip(rotations, translations)):
        pose = g2o.Isometry3d(R, t)
        fixed = (i == 0)
        ba.add_pose(pose_id=i, pose=pose, cam=cam_matrices[i], fixed=fixed)

    # Add 3D points and their observations
    for point_id, point in enumerate(points):
        ba.add_point(point_id=point_id, point=point)
        for cam_id in [0, 1]:
            measurement = keypoints_list[cam_id][point_id]
            #information = information_matrices_list[cam_id][point_id]
            information = np.identity(2)

            ba.add_edge(point_id=point_id, pose_id=cam_id, measurement=measurement,
                        information=information, robust_kernel_threshold=robust_kernel_threshold)

    # Optimize
    ba.optimize(max_iterations=num_iterations)

    # Extract the optimized camera poses and 3D points
    optimized_poses = [ba.get_pose(pose_id=i) for i in range(len(rotations))]
    optimized_points = [np.array(ba.get_point(point_id=i)).reshape(-1) for i in range(len(points)) if ba.get_point(point_id=i) is not None]

    final_error = ba.chi2()
    print("Final error after optimization:", final_error)
    return optimized_points, optimized_poses

def list_g2o_attributes():
    attributes = dir(g2o)
    for attr in attributes:
        if "Solver" in attr or "Block" in attr or 'OptimizationAlgorithm' in attr or 'RobustKernel' in attr:
            print(attr)

def confidence_to_information(confidence_levels, max_std_dev=1.0, min_std_dev=0.1):
    # Define standard deviations based on confidence levels
    std_devs = max_std_dev - confidence_levels * (max_std_dev - min_std_dev)

    # Convert to variance
    variances = std_devs ** 2

    # Create 2x2 information matrices, assuming the same uncertainty in x and y
    information_matrices = [np.array([[1 / var, 0], [0, 1 / var]]) for var in variances]

    return np.array(information_matrices)

def write_to_npz(rotation_matrix, translation_vector, filename):
    # Check if rotation_matrix and translation_vector are numpy arrays, if not, convert them.
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.array(rotation_matrix)

    # Ensure translation_vector is a numpy array, even if a single number is passed
    # Flatten the array to ensure it is a row vector
    translation_vector = np.array(translation_vector).flatten()
    translation_vector = np.atleast_2d(translation_vector)

    # Create an identity matrix of a given size
    dirVect = np.eye(3)

    # Save to .npz file
    np.savez(filename, R=rotation_matrix, P=translation_vector, dirVect=dirVect)


def rotationMatrixToEulerAngles(R):
    """
    Convert a rotation matrix to Euler angles.

    Args:
    - R (3x3 np.array): Rotation matrix.

    Returns:
    - angles (np.array): Euler angles [roll, pitch, yaw].
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):
    """
    Convert Euler angles to a rotation matrix.

    Args:
    - theta (np.array): Euler angles [roll, pitch, yaw].

    Returns:
    - R (3x3 np.array): Rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def add_noise_to_rotation(rotation_matrix, sigma=0.5):
    """
    Add noise to a rotation matrix.

    Args:
    - rotation_matrix (3x3 np.array): Original rotation matrix.
    - sigma (float): Standard deviation of Gaussian noise (in radians) added to Euler angles.

    Returns:
    - perturbed_rotation_matrix (3x3 np.array): Rotation matrix after adding noise.
    """
    # Convert rotation matrix to Euler angles
    angles = rotationMatrixToEulerAngles(rotation_matrix)

    # Add Gaussian noise to the angles
    noisy_angles = angles + np.random.normal(0, sigma, 1)

    # Convert back to rotation matrix
    perturbed_rotation_matrix = eulerAnglesToRotationMatrix(noisy_angles)

    return perturbed_rotation_matrix


def add_noise_to_translation(translation_vector, sigma=5):
    """
    Add noise to a translation vector.

    Args:
    - translation_vector (3x1 np.array): Original translation vector.
    - sigma (float): Standard deviation of Gaussian noise added to each component of the translation vector.

    Returns:
    - perturbed_translation_vector (3x1 np.array): Translation vector after adding noise.
    """
    noisy_translation = translation_vector + np.random.normal(0, sigma, 3).reshape(3, 1)
    return noisy_translation


def add_noise_to_points(points, sigma=5):
    """
    Add noise to 3D points.

    Args:
    - points (list of 3D np.array): Original 3D points.
    - sigma (float): Standard deviation of Gaussian noise added to each coordinate of the 3D points.

    Returns:
    - perturbed_points (list of 3D np.array): 3D points after adding noise.
    """
    perturbed_points = [point + np.random.normal(0, sigma, 1) for point in points]
    return perturbed_points

def apply_transformation_to_room_reference(rotation_matrices, translation_vectors):
    # Define the translation (200 cm along the positive Z-axis for Cam1)
    translation_vector1 = np.array([0, 0, 200]).reshape(3, 1)

    # Define the rotation (8 degrees around X-axis for Cam1)
    angle_deg = 8
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix1 = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # Apply the transformation to all cameras (assuming rotation_matrices and translation_vectors are lists of extrinsic parameters)
    new_rotation_matrices = []
    new_translation_vectors = []
    for i, (R, t) in enumerate(zip(rotation_matrices, translation_vectors)):
        # For Cam1, set the specific rotation and translation
        if i == 0:
            new_rotation_matrices.append(rotation_matrix1)
            new_translation_vectors.append(translation_vector1)
        else:
            # Apply inverse transformation to the other cameras to align with the new reference frame
            new_rotation_matrices.append(rotation_matrix1.T @ R)
            new_translation_vectors.append(rotation_matrix1.T @ (t - translation_vector1))

    return new_rotation_matrices, new_translation_vectors

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
    parser.add_argument("--image_path_pair3_517_520", type=str, default="data/Calibration_sets/517_520/more_corr/",help="Path to the directory containing input images")

    parser.add_argument("--image_path_common", type=str, default="data/Calibration_sets/common_correspondence/",help="Path to the directory containing input images")
    parser.add_argument("--image_path_cam2_cam3", type=str, default="data/Calibration_sets/518_536/",help="Path to the directory containing input images")

    parser.add_argument("--calib_path", type=str, default="data/intrinsic/", help="Path to the directory containing calibration files")
    args = parser.parse_args()

    # Load camera calibration parameters for each camera
    cameras = load_camera_params(args.calib_path, 517,518,536,520) # pick which camera pair you are calibrating
    cam517 = cameras[0]
    cam518 = cameras[1]
    cam536 = cameras[2]
    cam520 = cameras[3]

    cam517_mtx = cam517['matrix']
    cam518_mtx = cam518['matrix']
    cam536_mtx = cam536['matrix']
    cam520_mtx = cam520['matrix']
    print('Intrinsics loaded!')

    # Load OpenPose
    params = dict()
    params["model_folder"] = "openpose/models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print('OpenPose loaded!')

    # Load images
    images_pair1_517 = sorted(glob(f"{args.image_path_pair1_517_518}/517*.png"))
    images_pair1_518 = sorted(glob(f"{args.image_path_pair1_517_518}/518*.png"))

    images_pair2_517 = sorted(glob(f"{args.image_path_pair2_517_536}/517*.png"))
    images_pair2_536 = sorted(glob(f"{args.image_path_pair2_517_536}/536*.png"))
    images_pair2_536_viz = sorted(glob(f"{args.image_path_pair2_517_536}/536_1.png"))[0]
    images_pair2_536_viz = cv2.imread(images_pair2_536_viz)

    images_pair3_517 = sorted(glob(f"{args.image_path_pair3_517_520}/517*.png"))
    images_pair3_520 = sorted(glob(f"{args.image_path_pair3_517_520}/520*.png"))
    images_pair2_520_viz = sorted(glob(f"{args.image_path_pair3_517_520}/520_1.png"))[0]
    images_pair3_520_viz = cv2.imread(images_pair2_520_viz)

    print('Approach 1-2, 1-3, 1-4 #############################################################################################################')
    # Extract keypoints for the pair 1 using the extract_keypoints() function
    keypoints_pair1_517, keypoints_pair1_518, confidences_pair1_517, confidences_pair1_518 = extract_keypoints(images_pair1_517, images_pair1_518, opWrapper)
    keypoints_pair2_517, keypoints_pair2_536, confidences_pair2_517, confidences_pair2_536 = extract_keypoints(images_pair2_517, images_pair2_536, opWrapper)
    keypoints_pair3_517, keypoints_pair3_520, confidences_pair3_517, confidences_pair3_520 = extract_keypoints(images_pair3_517,images_pair3_520, opWrapper)

    # Apply filtering to the extracted keypoints with the specific threshold
    threshold_pair1 = 0.6
    threshold_pair2 = 0.7
    threshold_pair3 = 0.6

    filtered_keypoints_pair1_517, filtered_keypoints_pair1_518, filtconf_keypoints_pair1_517, filtconf_keypoints_pair1_518 = filter_keypoints(keypoints_pair1_517, keypoints_pair1_518, threshold_pair1)
    filtered_keypoints_pair2_517, filtered_keypoints_pair2_536, filtconf_keypoints_pair2_517, filtconf_keypoints_pair2_536 = filter_keypoints(keypoints_pair2_517, keypoints_pair2_536, threshold_pair2)
    filtered_keypoints_pair3_517, filtered_keypoints_pair3_520, filtconf_keypoints_pair3_517, filtconf_keypoints_pair3_520 = filter_keypoints(keypoints_pair3_517, keypoints_pair3_520, threshold_pair3)

    #print(filtconf_keypoints_pair3_517)
    # Estimate relative position using the filtered keypoints
    R_recoverPose_pair1, t_recoverPose_pair1 = estimate_recoverPose(cam517, cam518, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518)
    R_recoverPose_pair2, t_recoverPose_pair2 = estimate_recoverPose(cam517, cam536, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536)
    R_recoverPose_pair3, t_recoverPose_pair3 = estimate_recoverPose(cam517, cam520, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520)


    # Hardcoded keypoints
    #filtered_keypoints1_charuco = np.array([[289, 392], [250, 380], [233, 406], [193, 392], [214,434], [378, 392], [321,379], [309,404]])
    #print(filtered_keypoints1_charuco)
    #filtered_keypoints2_charuco = np.array([[1036.6193, 478.9927], [1075.4508, 467.73816], [982.38214, 459.42395], [1023.3541, 451.7], [892,449], [1051,510],[1092,493],[991,482]])

    # CUBE #############################################################################################################
    import seaborn as sns

    filtered_keypoints1_charuco = np.array([[793,104], [793,508], [521,139], [521,473], [992,140], [992,472]])
    print(filtered_keypoints1_charuco)
    filtered_keypoints2_charuco = np.array([[398,175], [398,578], [199,211], [199,542], [667,211], [667,541]])


    # Extend the add_controlled_noise function to support Gaussian noise
    def add_controlled_noise(points, noise_increment, iteration, noise_type='linear'):
        if noise_type == 'linear':
            noise = np.linspace(0, noise_increment, len(points)) * iteration
            noise = noise.reshape(-1, 1)
            noisy_points = points + noise
        elif noise_type == 'gaussian':
            # Gaussian noise with mean 0 and std deviation as noise_increment * iteration
            noise = np.random.normal(0, noise_increment * iteration, points.shape)
            noisy_points = points + noise
        return noisy_points

    def modify_intrinsics(K, error_factor):
        K_modified = K.copy()  # Create a copy of K to avoid modifying the original
        K_modified *= np.array([[error_factor, 1, error_factor],
                                [1, error_factor, error_factor],
                                [1, 1, 1]])
        return K_modified


    # Placeholder arrays to store metrics and additional data
    noise_levels = []
    rmse_rotations = []
    rmse_translations = []
    reproj_errors = []
    noisy_kps1 = []  # List to store noisy keypoints1
    noisy_kps2 = []  # List to store noisy keypoints2
    condition_numbers = []  # List to store condition numbers of Essential matrix E

    # Ground truth for comparison
    R_charuco, t_charuco = estimate_recoverPose(cam517, cam520, filtered_keypoints1_charuco,
                                                filtered_keypoints2_charuco)
    points_pair4 = triangulate_points(cam517, cam520, filtered_keypoints1_charuco, filtered_keypoints2_charuco,
                                      R_charuco, t_charuco)

    # Loop through different levels of noise
    num_iterations = 100
    noise_increment = 0.5  # Adjust this value based on how much noise you want to add at each step

    for i in range(0, num_iterations + 1):
        noise_level = noise_increment * i

        noisy_keypoints1 = add_controlled_noise(filtered_keypoints1_charuco, noise_increment, i)
        noisy_keypoints2 = add_controlled_noise(filtered_keypoints2_charuco, noise_increment, i)

        noisy_kps1.append(noisy_keypoints1)
        noisy_kps2.append(noisy_keypoints2)

        R_noisy, t_noisy = estimate_recoverPose(cam517, cam520, noisy_keypoints1, noisy_keypoints2)

        E, mask = cv2.findEssentialMat(noisy_keypoints1, noisy_keypoints2, cam517['matrix'], method=cv2.RANSAC)
        U, s, V = np.linalg.svd(E)
        cond_number = s[0] / s[-1]
        condition_numbers.append(cond_number)

        rmse_rotation = np.sqrt(np.mean((R_noisy - R_charuco) ** 2))
        rmse_translation = np.sqrt(np.mean((t_noisy - t_charuco) ** 2))
        reproj_error_noisy = calculate_reprojection_error(cam520, points_pair4, noisy_keypoints2, R_noisy, t_noisy)

        noise_levels.append(noise_level)
        rmse_rotations.append(rmse_rotation)
        rmse_translations.append(rmse_translation)
        reproj_errors.append(reproj_error_noisy)

        print(
            f"Iteration: {i}, Noise Level: {noise_level}, RMSE Rotation: {rmse_rotation}, RMSE Translation: {rmse_translation}, Reprojection Error: {reproj_error_noisy}")

    # Analyze spikes
    spike_indices = [i for i, reproj_error in enumerate(reproj_errors) if
                     reproj_error > 10]  # Change threshold as needed

    for spike_index in spike_indices:
        print(f"Spike observed at iteration: {spike_index}, Noise Level: {noise_levels[spike_index]}")
        print(f"Noisy Keypoints from Image 1: \n{noisy_kps1[spike_index]}")
        print(f"Noisy Keypoints from Image 2: \n{noisy_kps2[spike_index]}")
        print(f"Condition number of Essential matrix E: {condition_numbers[spike_index]}")
        print("-----")

    # Plotting the metrics
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(noise_levels, rmse_rotations, label='RMSE Rotation', color='r')
    plt.xlabel('Noise Level')
    plt.ylabel('RMSE Rotation')
    plt.title('Impact of Noise on RMSE Rotation')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(noise_levels, rmse_translations, label='RMSE Translation', color='g')
    plt.xlabel('Noise Level')
    plt.ylabel('RMSE Translation')
    plt.title('Impact of Noise on RMSE Translation')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(noise_levels, reproj_errors, label='Reprojection Error', color='b')
    plt.xlabel('Noise Level')
    plt.ylabel('Reprojection Error')
    plt.title('Impact of Noise on Reprojection Error')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # # Loop through different calibration error levels
    # error_factors = [1, 1.01, 1.05, 1.1, 1.2, 2]
    #
    # for error_factor in error_factors:
    #     cam517['matrix'] = modify_intrinsics(cam517['matrix'], error_factor)
    #     cam520['matrix'] = modify_intrinsics(cam520['matrix'], error_factor)
    #
    #
    #     R_calib_error, t_calib_error = estimate_recoverPose(cam517, cam520,
    #                                                         filtered_keypoints1_charuco, filtered_keypoints2_charuco)
    #
    #     # Calculate RMSE for rotation and translation
    #     rmse_rotation = np.sqrt(np.mean((R_calib_error - R_charuco) ** 2))
    #     rmse_translation = np.sqrt(np.mean((t_calib_error - t_charuco) ** 2))
    #
    #     print(
    #         f"Calibration Error Factor: {error_factor}, RMSE Rotation: {rmse_rotation}, RMSE Translation: {rmse_translation}")


#######################################
    R_charuco, t_charuco = estimate_recoverPose(cam517, cam520, filtered_keypoints1_charuco, filtered_keypoints2_charuco)
    R_charuco_decomp = decompose_rotation(R_charuco)
    print('R_charuco:')
    print(R_charuco_decomp)
    real_distance_cm_pair3 = 100
    scale_factor3, t_charuco = scale_translation(t_charuco, real_distance_cm_pair3)
    print('t_charuco:')
    print(t_charuco)
    points_pair4 = triangulate_points(cam517, cam520, filtered_keypoints1_charuco, filtered_keypoints2_charuco,
                                      R_charuco, t_charuco)
    reproj_error_pair1 = calculate_reprojection_error(cam520, points_pair4, filtered_keypoints2_charuco,
                                                      R_charuco, t_charuco)
    print('Reprojection error CUBE:')
    print(reproj_error_pair1)
###################################  Bundle adjustment and hyperparameters
    # Initialize reference camera
    rotation_vector1 = np.zeros((3, 3))
    translation_vector1 = np.array([0, 0, 0])

    # Customizable parameters for bundle adjustment
    num_iterations = 1000
    robust_kernel_threshold = np.sqrt(5.991) # 95% confidence interval
    solver_type = g2o.BlockSolverSE3
    linear_solver = g2o.LinearSolverEigenSE3
    # g2o solver types
    list_g2o_attributes()

    # Scaling with the real distance between sensors
    real_distance_cm_pair1 = 592
    real_distance_cm_pair2 = 762
    real_distance_cm_pair3 = 481 # 206 for synth

    # Scale translation for all pairs
    scale_factor_pair1, scaled_translation_pair1 = scale_translation(t_recoverPose_pair1, real_distance_cm_pair1)
    scale_factor_pair2, scaled_translation_pair2 = scale_translation(t_recoverPose_pair2, real_distance_cm_pair2)
    scale_factor_pair3, scaled_translation_pair3 = scale_translation(t_recoverPose_pair3, real_distance_cm_pair3)

    print('Scaled translation for pair 1:')
    print(scaled_translation_pair1)
    print('Scaled translation for pair 2:')
    print(scaled_translation_pair2)
    print('Scaled translation for pair 3:')
    print(scaled_translation_pair3)

#######################################
    from math import radians, cos, sin


    def cam1_to_room_transformation():
        # 8-degree forward tilt around the X-axis
        theta_x = np.radians(98)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        # 45-degree rotation around the Y-axis
        theta_y = np.radians(45)
        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # Combine rotations
        R_cam1_room = Ry @ Rx

        # Translation along Z-axis (200 cm high)
        t_cam1_room = np.array([0, 0, 200])

        return R_cam1_room, t_cam1_room

    def inverse_transformation(R, t):
        R_inv = R.T
        t_inv = t.reshape(-1)
        return R_inv, t_inv

    R_cam1_room, t_cam1_room = cam1_to_room_transformation()
    print('R_cam1_room:')
    print(R_cam1_room)
    print('t_cam1_room:')
    print(t_cam1_room)

    R_cam1_cam2, t_cam1_cam2 = inverse_transformation(R_recoverPose_pair1, scaled_translation_pair1)
    R_cam1_cam3, t_cam1_cam3 = inverse_transformation(R_recoverPose_pair2, scaled_translation_pair2)
    R_cam1_cam4, t_cam1_cam4 = inverse_transformation(R_recoverPose_pair3, scaled_translation_pair3)

    #t_cam1_cam4[0] = -t_cam1_cam4[0]
    print(t_cam1_cam4)

    tranz = t_cam1_cam4 + t_cam1_room
    p2 =R_recoverPose_pair3 @ tranz
    print('p2:')
    print(p2)
    def transform_to_room(R_cam1_camX, t_cam1_camX, R_cam1_room, t_cam1_room):
        R_room_camX = R_cam1_room @ R_cam1_camX
        t_room_camX = R_cam1_room @ (t_cam1_camX + t_cam1_room)
        return R_room_camX, t_room_camX


    # Transformations from room to cam2, cam3, cam4
    R_room_cam2, t_room_cam2 = transform_to_room(R_cam1_cam2, t_cam1_cam2, R_cam1_room, t_cam1_room)
    R_room_cam3, t_room_cam3 = transform_to_room(R_cam1_cam3, t_cam1_cam3, R_cam1_room, t_cam1_room)
    R_room_cam4, t_room_cam4 = transform_to_room(R_cam1_cam4, t_cam1_cam4, R_cam1_room, t_cam1_room)

    print('ROOM TO CAM')
    print('R_room_cam2:')
    print(R_room_cam2)
    print('t_room_cam2:')
    print(t_room_cam2)
    print('R_room_cam3:')
    print(R_room_cam3)
    print('t_room_cam3:')
    print(t_room_cam3)
    print('R_room_cam4:')
    print(R_room_cam4)
    print('t_room_cam4:')
    print(t_room_cam4)






    # Triangulate the points
    points_pair1 = triangulate_points(cam517, cam518, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518,
                                      R_recoverPose_pair1, scaled_translation_pair1)
    points_pair2 = triangulate_points(cam517, cam536, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536,
                                      R_recoverPose_pair2, scaled_translation_pair2)
    points_pair3 = triangulate_points(cam517, cam520, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520,
                                      R_recoverPose_pair3, scaled_translation_pair3)

    # Information matrices for each pair of cameras based on confidence levels
    information_matrix_p1cam1 = confidence_to_information(filtconf_keypoints_pair1_517)
    information_matrix_p1cam2 = confidence_to_information(filtconf_keypoints_pair1_518)

    information_matrix_p2cam1 = confidence_to_information(filtconf_keypoints_pair2_517)
    information_matrix_p2cam2 = confidence_to_information(filtconf_keypoints_pair2_536)

    information_matrix_p3cam1 = confidence_to_information(filtconf_keypoints_pair3_517)
    information_matrix_p3cam2 = confidence_to_information(filtconf_keypoints_pair3_520)

    translation_vector1 = translation_vector1.reshape(3, 1)
    # Print all bundle adjustment inputs
    print('Points:')
    print(points_pair3[:5])
    print(len(points_pair3))

    print('Cameras:')
    print(cam517_mtx)
    print(cam520_mtx)
    print('Rotations:')
    print(rotation_vector1)
    print(R_recoverPose_pair3)
    print('Translations:')
    print(translation_vector1)
    print(scaled_translation_pair3)
    print('Keypoints:')
    print(filtered_keypoints_pair3_517[:5])
    print(filtered_keypoints_pair3_520[:5])
    print(len(filtered_keypoints_pair3_517))
    print('Information matrices:')
    print(information_matrix_p3cam1[:5])
    print(len(information_matrix_p3cam1))
    print(information_matrix_p3cam2[:5])

    #Add noise to the inputs
    # noisy_rotations = [add_noise_to_rotation(R) for R in [rotation_vector1, R_recoverPose_pair3]]
    # noisy_translations = [add_noise_to_translation(t) for t in [translation_vector1, scaled_translation_pair3]]
    noisy_points = add_noise_to_points(points_pair3)
    noisy_rotation_cam2 = add_noise_to_rotation(R_recoverPose_pair3)
    noisy_translation_cam2 = add_noise_to_translation(scaled_translation_pair3)


    #print(translation_vector1)
    noisy_points = np.array(noisy_points)

    print ('Noisy points:')
    print(noisy_points[:5])
    print('Noisy cameras:')
    print(noisy_rotation_cam2)
    print(noisy_translation_cam2)
    # Print initial inputs
    print('Initial points and cameras:')
    print(points_pair3)
    print(R_recoverPose_pair3)
    print(scaled_translation_pair3)


    # print(translation_vector1)
    # def simulate_camera_setup(num_points=10, baseline=481, fx=1120, fy=1120, cx=600, cy=400,
    #                           point_distance=1000, point_variance=200, noise_level=5.0):
    #     """
    #     Simulate a camera setup.
    #
    #     Args:
    #     - num_points: number of 3D points to simulate.
    #     - baseline: distance between the two cameras.
    #     - fx, fy, cx, cy: intrinsic parameters of the cameras.
    #     - point_distance: average distance of 3D points from the cameras.
    #     - point_variance: variance in the distribution of 3D points around the average distance.
    #     - noise_level: noise to add in the 2D projections to simulate real-world observations.
    #
    #     Returns:
    #     - keypoints1, keypoints2: noisy 2D projections in the two cameras.
    #     - points_3d: true 3D points.
    #     - rotations: rotations of the two cameras.
    #     - translations: translations of the two cameras.
    #     """
    #
    #     # Camera matrices (same for both cameras in this simulation)
    #     K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    #
    #     # Simulate 3D points in front of the camera
    #     points_3d = np.hstack((np.random.randn(num_points, 2) * point_variance,
    #                            point_distance + np.random.randn(num_points, 1) * point_variance))
    #
    #     # Camera poses
    #     # First camera at origin and second camera translated by baseline along x-axis
    #     rotations = [np.eye(3), np.eye(3)]
    #     translations = [np.array([[0], [0], [0]]), np.array([[baseline], [0], [0]])]
    #
    #     # Project 3D points to 2D image planes of the cameras
    #     P1 = K @ np.hstack([rotations[0], translations[0]])
    #     P2 = K @ np.hstack([rotations[1], translations[1]])
    #
    #     # Convert 3D points to homogeneous coordinates for projection
    #     points_3d_homogeneous = np.hstack([points_3d, np.ones((num_points, 1))])
    #
    #     keypoints1 = (P1 @ points_3d_homogeneous.T).T
    #     keypoints1 /= keypoints1[:, 2:3]
    #
    #     keypoints2 = (P2 @ points_3d_homogeneous.T).T
    #     keypoints2 /= keypoints2[:, 2:3]
    #
    #     # Add noise to the 2D keypoints
    #     keypoints1[:, :2] += np.random.randn(num_points, 2) * noise_level
    #     keypoints2[:, :2] += np.random.randn(num_points, 2) * noise_level
    #
    #     return keypoints1[:, :2], keypoints2[:, :2], points_3d, rotations, translations
    #
    #
    # keypoints1, keypoints2, points_3d, rotations, translations = simulate_camera_setup()
    #
    # optimized_points, optimized_cameras = perform_bundle_adjustment_with_class(
    #     points=points_3d,
    #     cam_matrices=[cam517_mtx, cam520_mtx],
    #     rotations=rotations,
    #     translations=translations,
    #     keypoints_list=[keypoints1, keypoints2],
    #
    #     num_iterations=num_iterations,
    #     robust_kernel_threshold=robust_kernel_threshold,
    #     information_matrices_list=[information_matrix_p3cam1, information_matrix_p3cam2],
    #     solver_type=solver_type,
    #     linear_solver=linear_solver
    # )

    # # Perform bundle adjustment
    # optimized_points, optimized_cameras = perform_bundle_adjustment_with_class(
    #     points=noisy_points,
    #     cam_matrices=[cam517_mtx, cam520_mtx],
    #     rotations=[rotation_vector1, noisy_rotation_cam2],
    #     translations=[translation_vector1, noisy_translation_cam2],
    #     keypoints_list=[filtered_keypoints_pair3_517, filtered_keypoints_pair3_520],
    #
    #     num_iterations=num_iterations,
    #     robust_kernel_threshold=robust_kernel_threshold,
    #     information_matrices_list=[information_matrix_p3cam1, information_matrix_p3cam2],
    #     solver_type=solver_type,
    #     linear_solver=linear_solver
    # )
    #scaled_translation_pair3[0] = -scaled_translation_pair3[0]
    # optimized_points, optimized_cameras = perform_bundle_adjustment_with_class(
    #     points=points_pair3,
    #     cam_matrices=[cam517_mtx, cam520_mtx],  # Your camera parameters here
    #     rotations=[rotation_vector1, R_recoverPose_pair3],
    #     translations=[translation_vector1, scaled_translation_pair3],
    #     keypoints_list=[filtered_keypoints_pair3_517, filtered_keypoints_pair3_520],
    #
    #     num_iterations=num_iterations,
    #     robust_kernel_threshold=robust_kernel_threshold,
    #     information_matrices_list=[information_matrix_p3cam1, information_matrix_p3cam2],
    #     solver_type=solver_type,
    #     linear_solver=linear_solver
    # )
    optimized_points, optimized_cameras = perform_bundle_adjustment_with_class(
        points=points_pair4,
        cam_matrices=[cam517_mtx,cam520_mtx],  # Your camera parameters here
        rotations=[rotation_vector1, R_charuco],
        translations=[translation_vector1, t_charuco],
        keypoints_list=[filtered_keypoints1_charuco,filtered_keypoints2_charuco],

        num_iterations=num_iterations,
        robust_kernel_threshold=robust_kernel_threshold,
        information_matrices_list=[information_matrix_p3cam1, information_matrix_p3cam2],
        solver_type=solver_type,
        linear_solver=linear_solver
    )

    optimized_points = np.array(optimized_points)

    # Extracting the optimized rotation matrices and translation vectors for each camera
    cam1_BArot, cam2_BArot = [camera.rotation().matrix() for camera in optimized_cameras]
    cam1_BAtrans, cam2_BAtrans = [camera.translation() for camera in optimized_cameras]

    print('cam1_BArot:')
    print(cam1_BArot)
    print('cam2_BArot:')
    print(cam2_BArot)
    print('cam1_BAtrans:')
    print(cam1_BAtrans)
    print('cam2_BAtrans:')
    print(cam2_BAtrans)

    # Points before / after BA
    print('Optimized points and cameras:')
    print(points_pair4[:10])
    print(len(optimized_points))
    print(optimized_points[:10])

    #print(missing_points)
    #points_pair3 = np.delete(points_pair3, missing_points, axis=0)
    #filtered_keypoints_pair3_520 = np.delete(filtered_keypoints_pair3_520, missing_points, axis=0)

    # Visualize the 3D points
    visualize_3d_points(points_pair3)
    visualize_3d_points(optimized_points)


    #BA_scale_factor_pair1, BA_scaled_translation_pair1 = scale_translation(cam2_BAtrans, 100)
    # BA_scale_factor_pair2, BA_scaled_translation_pair2 = scale_translation(cam2_BAtrans, real_distance_cm_pair2)
    # BA_scale_factor_pair3, BA_scaled_translation_pair3 = scale_translation(cam2_BAtrans, real_distance_cm_pair3)

    # Reprojection error before BA
    reproj_error_pair1 = calculate_reprojection_error(cam518, points_pair1, filtered_keypoints_pair1_518, R_recoverPose_pair1, scaled_translation_pair1)
    reproj_error_pair2 = calculate_reprojection_error(cam536, points_pair2, filtered_keypoints_pair2_536, R_recoverPose_pair2, scaled_translation_pair2)
    reproj_error_pair3 = calculate_reprojection_error(cam520, points_pair3, filtered_keypoints_pair3_520, R_recoverPose_pair3, scaled_translation_pair3)

    # Reprojection error after BA
    #BA_reproj_error_pair1 = calculate_reprojection_error(cam520, optimized_points, filtered_keypoints_pair3_520, cam2_BArot, BA_scaled_translation_pair1)
    BA_reproj_error_pair1 = calculate_reprojection_error(cam520, optimized_points, filtered_keypoints2_charuco,
                                                         cam2_BArot, cam2_BAtrans)
    # BA_reproj_error_pair2 = calculate_reprojection_error(cam536, points_cam1_cam3, filtered_keypoints_pair2_536, cam3_BArot, cam3_BAtrans)
    # BA_reproj_error_pair3 = calculate_reprojection_error(cam520, points_cam1_cam4, filtered_keypoints_pair3_520, cam4_BArot, cam4_BAtrans)

    # Visualize reprojections
    vis_initial = visualize_reprojections(images_pair3_520_viz,cam520 , R_recoverPose_pair3, scaled_translation_pair3, points_pair3, filtered_keypoints_pair3_520,'visualizations/initial.png')
    vis_optimized = visualize_reprojections(images_pair3_520_viz, cam520, cam2_BArot, cam2_BAtrans, optimized_points, filtered_keypoints1_charuco,'visualizations/BA.png')

    # Display the images
    cv2.imshow("Initial Projections", vis_initial)
    cv2.imshow("Optimized Projections", vis_optimized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print reprojection errors
    print(f"Reprojection error for pair 1: {reproj_error_pair1}")
    print(f"Reprojection error for pair 2: {reproj_error_pair2}")
    print(f"Reprojection error for pair 3: {reproj_error_pair3}")

    print(f"Reprojection error for pair 1 after BA: {BA_reproj_error_pair1}")
    # print(f"Reprojection error for pair 2 after BA: {BA_reproj_error_pair2}")
    # print(f"Reprojection error for pair 3 after BA: {BA_reproj_error_pair3}")

    # Results
    # # Scaling with the real distance between sensors
    # real_distance_cm_pair1 = 592 #206  #594
    # real_distance_cm_pair2 = 762 #764
    # real_distance_cm_pair3 = 481  #479
    #
    # # Scale translation for all pairs
    # scale_factor_pair1, scaled_translation_pair1 = scale_translation(t_recoverPose_pair1, real_distance_cm_pair1)
    # #scale_factor_pair2, scaled_translation_pair2 = scale_translation(t_recoverPose_pair2, real_distance_cm_pair2)
    # scale_factor_pair3, scaled_translation_pair3 = scale_translation(t_recoverPose_pair3, real_distance_cm_pair3)
    #
    # BA_scale_factor_pair1, BA_scaled_translation_pair1 = scale_translation(cam2_BAtrans, real_distance_cm_pair2)
    # # BA_scale_factor_pair2, BA_scaled_translation_pair2 = scale_translation(cam2_BAtrans, real_distance_cm_pair2)
    # # BA_scale_factor_pair3, BA_scaled_translation_pair3 = scale_translation(cam2_BAtrans, real_distance_cm_pair3)

    #write_to_npz(R_recoverPose_pair2, scaled_translation_pair2, 'visualize_3d/Sensor_Parameters_13_536.npz')

    # Decompose rotation matrix
    rot_decomp_pair1 = decompose_rotation(R_recoverPose_pair1)
    rot_decomp_pair2 = decompose_rotation(R_recoverPose_pair2)
    rot_decomp_pair3 = decompose_rotation(R_recoverPose_pair3)

    BA_rot_decomp_pair1 = decompose_rotation(cam2_BArot)
    # BA_rot_decomp_pair2 = decompose_rotation(cam3_BArot)
    # BA_rot_decomp_pair3 = decompose_rotation(cam4_BArot)

    # Number of keypoints
    num_keypoints_pair1 = len(filtered_keypoints_pair1_517)
    num_keypoints_pair2 = len(filtered_keypoints_pair2_517)
    num_keypoints_pair3 = len(filtered_keypoints_pair3_517)

    #Print results
    print('Pair 1 - translation from camera 517 to 518 - UBIQISENSE PIPELINE:', "[-483.67437729 -122.77347367  348.15576036] (cm) ")
    print('Pair 1 - translation from camera 517 to 518 - recoverPose:', scaled_translation_pair1.ravel())
    print('Pair 1 - rotation from camera 517 to 518 - UBIQISENSE PIPELINE:', "( 47.95943056 -67.13253629 -62.56659299) (deg.) ")
    print('Pair 1 - rotation from camera 517 to 518 - recoverPose:', rot_decomp_pair1)
    print('Pair 1 - scale factor:', scale_factor_pair1)
    print('Number of keypoints:', num_keypoints_pair1, "/ Threshold: ", threshold_pair1, "/ Reprojection error:", reproj_error_pair1)
    print('')

    print('Pair 2 - translation from camera 517 to 536 - UBIQISENSE PIPELINE:', "[-188.76428849 -211.59882113  731.79469696] (cm) ")
    print('Pair 2 - translation from camera 517 to 536 - recoverPose:', scaled_translation_pair2.ravel())
    print('Pair 2 - rotation from camera 517 to 536 - UBIQISENSE PIPELINE:', "( 150.39854946   13.33519403 -177.79742376) (deg.) ")
    print('Pair 2 - rotation from camera 517 to 536 - recoverPose:', rot_decomp_pair2)
    print('Pair 2 - scale factor:', scale_factor_pair2)
    print('Number of keypoints:', num_keypoints_pair2, "/ Threshold: ", threshold_pair2, "/ Reprojection error:", reproj_error_pair2)
    print('')

    print('Pair 3 - translation from camera 517 to 520 - UBIQISENSE PIPELINE:', "[306.34257461 -94.6791296  380.83349388] (cm) ")
    print('Pair 3 - rotation from camera 517 to 520 - UBIQISENSE PIPELINE:', "( 132.62900666  69.60734076 151.97386786) (deg.) ")
    print('Pair 3 - translation from camera 517 to 520 - recoverPose:', scaled_translation_pair3.ravel())
    print('Pair 3 - rotation from camera 517 to 520 - recoverPose:', rot_decomp_pair3)
    print('Pair 3 - scale factor:', scale_factor_pair3)
    print('Number of keypoints:', num_keypoints_pair3, "/ Threshold: ", threshold_pair3, "/ Reprojection error:", reproj_error_pair3)
    print('')

    # Bundle Adjustment Results
    print('Bundle Adjustment Results')
    print('Pair 1 - translation from camera 517 to 520 - BA:', cam2_BAtrans.ravel())
    print('Pair 1 - rotation from camera 517 to 520 - BA:', BA_rot_decomp_pair1)
    #print('Pair 1 - scale factor:', BA_scale_factor_pair1)
    print('')
    # print('Pair 2 - translation from camera 517 to 536 - BA:', BA_scaled_translation_pair2.ravel())
    # print('Pair 2 - rotation from camera 517 to 536 - BA:', BA_rot_decomp_pair2)
    # print('Pair 2 - scale factor:', BA_scale_factor_pair2)
    # print('')
    # print('Pair 3 - translation from camera 517 to 520 - BA:', BA_scaled_translation_pair3.ravel())
    # print('Pair 3 - rotation from camera 517 to 520 - BA:', BA_rot_decomp_pair3)
    # print('Pair 3 - scale factor:', BA_scale_factor_pair3)
    # print('')

    # Absolution position solvePnP
    D3_points = np.array([[0, 0, 0], [3, 3, 200], [300, 130, 0], [300,130,70]])
    D2_points = np.array([[81, 365], [55, 45], [411, 548], [418, 510]])

    rvec, tvec = estimate_solvePnP( D3_points, D2_points, cam536)
    # rvec = decompose_rotation(rvec)
    # R_recoverPose_pair2, t_recoverPose_pair2

    # Given data
    R_rel = R_recoverPose_pair2
    t_rel = scaled_translation_pair2
    R_abs_cam2 = rvec
    t_abs_cam2 = tvec

    # Step 1: Inverse of the relative transformation from recoverPose
    R_rel_inverse = R_rel.T
    t_rel_inverse = -np.dot(R_rel.T, t_rel)
    # Step 2: Compute the absolute rotation and translation of Camera 1
    R_abs_cam1 = np.dot(R_rel, R_abs_cam2)
    t_abs_cam1 = t_rel + np.dot(R_rel, t_abs_cam2)


    R_abs_cam1 = decompose_rotation(R_abs_cam1)
    print('rvec:')
    print(R_abs_cam1)
    print('tvec:')
    print(t_abs_cam1)



    #overlay_images(sorted(glob(f"{args.image_path_pair1_517_518}/517*.png")),filtered_keypoints_pair1_517)
    #draw_skeleton(images_common_517, filtered_common_517,  )

    #Visualize
    # visualize_3d_points(triangulated_points_pair1_517_518)
    # visualize_3d_points(triangulated_points_pair2_517_536)
    # visualize_3d_points(triangulated_points_pair3_517_520)

    #print(reprojected_points) remove a dim!!
    #visualize_reprojection_error(filtered_keypoints_pair1_518, reprojected_points)
    #visualize_reprojection_error_heatmap(filtered_keypoints_pair1_518, reprojected_points)
    #visualize_reprojection_error_scatter(filtered_keypoints_pair1_518, reprojected_points)


