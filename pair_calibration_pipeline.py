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

    confidences1 = np.array(confidences_cam1)
    confidences2 = np.array(confidences_cam2)

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

    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float64)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float64)

    return filtered_keypoints1, filtered_keypoints2
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

    _ ,E , R_recoverPose, t_recoverPose, _= cv2.recoverPose(keypoints1, keypoints2, K1, dist1, K2, dist2)

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


### Visualization
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

def scale_translation(tvec, real_distance_cm):
    print(tvec)
    estimated_distance = np.linalg.norm(tvec)
    print(estimated_distance)
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

    images_pair3_517 = sorted(glob(f"{args.image_path_pair3_517_520}/517*.png"))
    images_pair3_520 = sorted(glob(f"{args.image_path_pair3_517_520}/520*.png"))

    print('Approach 1-2, 1-3, 1-4 #############################################################################################################')
    # Extract keypoints for the pair 1 using the extract_keypoints() function
    keypoints_pair1_517, keypoints_pair1_518, confidences_pair1_517, confidences_pair1_518 = extract_keypoints(images_pair1_517, images_pair1_518, opWrapper)
    keypoints_pair2_517, keypoints_pair2_536, confidences_pair2_517, confidences_pair2_536 = extract_keypoints(images_pair2_517, images_pair2_536, opWrapper)
    keypoints_pair3_517, keypoints_pair3_520, confidences_pair3_517, confidences_pair3_520 = extract_keypoints(images_pair3_517,images_pair3_520, opWrapper)

    # Apply filtering to the extracted keypoints with the specific threshold
    threshold_pair1 = 0.5
    threshold_pair2 = 0.6
    threshold_pair3 = 0.7

    filtered_keypoints_pair1_517, filtered_keypoints_pair1_518 = filter_keypoints(keypoints_pair1_517, keypoints_pair1_518, threshold_pair1)
    filtered_keypoints_pair2_517, filtered_keypoints_pair2_536 = filter_keypoints(keypoints_pair2_517, keypoints_pair2_536, threshold_pair2)
    filtered_keypoints_pair3_517, filtered_keypoints_pair3_520 = filter_keypoints(keypoints_pair3_517, keypoints_pair3_520, threshold_pair3)

    # Estimate relative position using the filtered keypoints
    R_recoverPose_pair1, t_recoverPose_pair1 = estimate_recoverPose(cam517, cam518, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518)
    R_recoverPose_pair2, t_recoverPose_pair2 = estimate_recoverPose(cam517, cam536, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536)
    R_recoverPose_pair3, t_recoverPose_pair3 = estimate_recoverPose(cam517, cam520, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520)

    # Triangulate the points
    points_pair1 = triangulate_points(cam517, cam518, filtered_keypoints_pair1_517, filtered_keypoints_pair1_518, R_recoverPose_pair1, t_recoverPose_pair1)
    points_pair2 = triangulate_points(cam517, cam536, filtered_keypoints_pair2_517, filtered_keypoints_pair2_536, R_recoverPose_pair2, t_recoverPose_pair2)
    points_pair3 = triangulate_points(cam517, cam520, filtered_keypoints_pair3_517, filtered_keypoints_pair3_520, R_recoverPose_pair3, t_recoverPose_pair3)

    # Compute reprojection error
    reproj_error_pair1 = calculate_reprojection_error(cam518, points_pair1, filtered_keypoints_pair1_518, R_recoverPose_pair1, t_recoverPose_pair1)
    reproj_error_pair2 = calculate_reprojection_error(cam536, points_pair2, filtered_keypoints_pair2_536, R_recoverPose_pair2, t_recoverPose_pair2)
    reproj_error_pair3 = calculate_reprojection_error(cam520, points_pair3, filtered_keypoints_pair3_520, R_recoverPose_pair3, t_recoverPose_pair3)

    print(f"Reprojection error for pair 1: {reproj_error_pair1}")
    print(f"Reprojection error for pair 2: {reproj_error_pair2}")
    print(f"Reprojection error for pair 3: {reproj_error_pair3}")

    # Results
    # Scaling with the real distance between sensors
    real_distance_cm_pair1 = 592 #206  #594
    real_distance_cm_pair2 = 762 #764
    real_distance_cm_pair3 = 481  #479

    # Scale translation for all pairs
    scale_factor_pair1, scaled_translation_pair1 = scale_translation(t_recoverPose_pair1, real_distance_cm_pair1)
    scale_factor_pair2, scaled_translation_pair2 = scale_translation(t_recoverPose_pair2, real_distance_cm_pair2)
    scale_factor_pair3, scaled_translation_pair3 = scale_translation(t_recoverPose_pair3, real_distance_cm_pair3)

    # Decompose rotation matrix
    rot_decomp_pair1 = decompose_rotation(R_recoverPose_pair1)
    rot_decomp_pair2 = decompose_rotation(R_recoverPose_pair2)
    rot_decomp_pair3 = decompose_rotation(R_recoverPose_pair3)

    # Number of keypoints
    num_keypoints_pair1 = len(filtered_keypoints_pair1_517)
    num_keypoints_pair2 = len(filtered_keypoints_pair2_517)
    num_keypoints_pair3 = len(filtered_keypoints_pair3_517)

    # Print results
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
    print('Pair 3 - translation from camera 517 to 520 - recoverPose:', scaled_translation_pair3.ravel())
    print('Pair 3 - rotation from camera 517 to 520 - UBIQISENSE PIPELINE:', "( 132.62900666  69.60734076 151.97386786) (deg.) ")
    print('Pair 3 - rotation from camera 517 to 520 - recoverPose:', rot_decomp_pair3)
    print('Pair 3 - scale factor:', scale_factor_pair3)
    print('Number of keypoints:', num_keypoints_pair3, "/ Threshold: ", threshold_pair3, "/ Reprojection error:", reproj_error_pair3)
    print('')

    print('Approach 1-2, 2-3, 1-3 Triangulation #############################################################################################################')

    images_pair3_518 = sorted(glob(f"{args.image_path_cam2_cam3}/518*.png"))
    images_pair3_536 = sorted(glob(f"{args.image_path_cam2_cam3}/536*.png"))

    # Extract keypoints
    keypoints_pair3_518, keypoints_pair3_536, confidences_pair3_518, confidences_pair3_536 = extract_keypoints(images_pair3_518, images_pair3_536, opWrapper)

    # Filter keypoints
    threshold_approach2 = 0.4
    filtered_keypoints_pair3_518, filtered_keypoints_pair3_536 = filter_keypoints(keypoints_pair3_518, keypoints_pair3_536, threshold_approach2)

    # Recover pose
    R_recoverPose_pair3, t_recoverPose_pair3 = estimate_recoverPose(cam518, cam536, filtered_keypoints_pair3_518, filtered_keypoints_pair3_536)

    T32 = np.zeros((4, 4))
    T32[0:3, 0:3] = R_recoverPose_pair3
    T32[0:3, 3] = t_recoverPose_pair3.ravel()
    T32[3, 3] = 1
    T23 = np.linalg.inv(T32)

    T21 = np.zeros((4, 4))
    T21[0:3, 0:3] = R_recoverPose_pair1
    T21[0:3, 3] = t_recoverPose_pair1.ravel()
    T21[3, 3] = 1
    print(T21)
    T12 = np.linalg.inv(T21)
    #print(T12)

    # T21 = np.zeros((4, 4))
    # T21[0:3, 0:3] = np.linalg.inv(R_recoverPose_pair1)
    # T21[0:3, 3] = - np.dot(np.linalg.inv(R_recoverPose_pair1), t_recoverPose_pair1.ravel())
    # T21[3, 3] = 1
    # print(T21)
    # #T12 = np.linalg.inv(T21)
    # #print(T12)


    T13 = np.dot(T12,T23)
    T13 = np.linalg.inv(T13)
    R_13 = T13[:3, :3]  # Rotation matrix is the top-left 3x3 submatrix
    t_13 = T13[:3, 3]  # Translation vector is the right-most column

    # Triangulation
    points_pair3 = triangulate_points(cam518, cam536, filtered_keypoints_pair3_518, filtered_keypoints_pair3_536, R_recoverPose_pair3, t_recoverPose_pair3)

    # Reprojection error
    reproj_error_pair3 = calculate_reprojection_error(cam536, points_pair3, filtered_keypoints_pair3_536, R_13, t_13)

    # Scaling
    real_distance_cm_pair3 = 762
    # real_distance_cm_pair3 = 481 + 592
    scale_factor_pair3, scaled_translation_pair3 = scale_translation(t_13, real_distance_cm_pair3)

    # Decomp rotation
    rot_decomp_pair3 = decompose_rotation(R_13)

    num_keypoints_pair3 = len(filtered_keypoints_pair3_518)


    # Print results
    print('Pair 1 - translation from camera 517 to 518 - UBIQISENSE PIPELINE:',
          "[-483.67437729 -122.77347367  348.15576036] (cm) ")
    print('Pair 1 - translation from camera 517 to 518 - recoverPose:', scaled_translation_pair1.ravel())
    print('Pair 1 - rotation from camera 517 to 518 - UBIQISENSE PIPELINE:',
          "( 47.95943056 -67.13253629 -62.56659299) (deg.) ")
    print('Pair 1 - rotation from camera 517 to 518 - recoverPose:', rot_decomp_pair1)
    print('Pair 1 - scale factor:', scale_factor_pair1)
    print('Number of keypoints:', num_keypoints_pair1, "/ Threshold: ", threshold_pair1, "/ Reprojection error:",
          reproj_error_pair1)
    print('')

    print('Pair 2 - translation from camera 517 to 536 - UBIQISENSE PIPELINE:',"[-188.76428849 -211.59882113  731.79469696] (cm) ")
    print('Pair 2 - translation from camera 536 to 517 - recoverPose:', scaled_translation_pair3.ravel())
    print('Pair 2 - rotation from camera 517 to 536 - UBIQISENSE PIPELINE:', "( 150.39854946   13.33519403 -177.79742376) (deg.) ")
    print('Pair 2 - rotation from camera 536 to 517 - recoverPose:', rot_decomp_pair3)
    print('Pair 2 - scale factor:', scale_factor_pair3)
    print('Number of keypoints:', num_keypoints_pair3, "/ Threshold: ", threshold_approach2, "/ Reprojection error:", reproj_error_pair3)
    print('')

    #print('Pair 3 - translation from camera 517 to 520 - UBIQISENSE PIPELINE:', "[306.34257461 -94.6791296  380.83349388] (cm) ")
    print('Pair 3 - translation from camera 536 to 517 - recoverPose:', scaled_translation_pair2.ravel())
    #print('Pair 3 - rotation from camera 517 to 520 - UBIQISENSE PIPELINE:',"( 132.62900666  69.60734076 151.97386786) (deg.) ")
    print('Pair 3 - rotation from camera 517 to 536 - recoverPose:', rot_decomp_pair2)
    print('Pair 3 - scale factor:', scale_factor_pair3)
    print('Number of keypoints:', num_keypoints_pair3, "/ Threshold: ", threshold_pair3, "/ Reprojection error:",reproj_error_pair3)
    print('')





    # print("Pair 1 - translation from camera 517 to 518 - UBIQISENSE PIPELINE:",
    #       "[-483.67437729 -122.77347367  348.15576036] (cm) ")
    # print("Pair 2 - translation from camera 517 to 536 - UBIQISENSE PIPELINE:",
    #       "[-188.76428849 -211.59882113  731.79469696] (cm) ")
    # print("Pair 3 - translation from camera 517 to 520 - UBIQISENSE PIPELINE:",
    #       "[306.34257461 -94.6791296  380.83349388] (cm) ")
    # print("Pair 1 - rotation from camera 517 to 518 - UBIQISENSE PIPELINE:", "( 47.95943056 -67.13253629 -62.56659299) (deg.) ")
    # print("Pair 2 - rotation from camera 517 to 536 - UBIQISENSE PIPELINE:",
    #       "( 150.39854946   13.33519403 -177.79742376) (deg.) ")
    # print("Pair 3 - rotation from camera 517 to 520 - UBIQISENSE PIPELINE:",
    #       "( 132.62900666  69.60734076 151.97386786) (deg.) ")

    print('Approach 1-2, 2-3, 3-4 #############################################################################################################')
    # Load images
    images_common_517 = sorted(glob(f"{args.image_path_common}/517*.png"))
    images_common_518 = sorted(glob(f"{args.image_path_common}/518*.png"))
    images_common_536 = sorted(glob(f"{args.image_path_common}/536*.png"))
    images_common_520 = sorted(glob(f"{args.image_path_common}/520*.png"))

    # Extract keypoints for all cameras
    keypoints_common_517, keypoints_common_518, keypoints_common_536, keypoints_common_520, confidence_common_517, confidence_common_518, confidence_common_536, confidence_common_520 = extract_keypoints_common(images_common_517, images_common_518, images_common_536, images_common_520, opWrapper)

    # Filter keypoints
    threshold_common = 0.4
    filtered_common_517, filtered_common_518, filtered_common_536, filtered_common_520 = filter_keypoints_common(keypoints_common_517, keypoints_common_518, keypoints_common_536, keypoints_common_520, threshold_common)

    # Estimate relative pose
    R_recoverPose_pair1_common, t_recoverPose_pair1_common = estimate_recoverPose(cam517, cam518, filtered_common_517, filtered_common_518)

    R_recoverPose_pair2_common, t_recoverPose_pair2_common = estimate_recoverPose(cam517, cam536, filtered_common_517, filtered_common_536)

    # Triangulate points
    triangulated_points_pair1_common = triangulate_points(cam517, cam518, filtered_common_517, filtered_common_518, R_recoverPose_pair1_common, t_recoverPose_pair1_common)

    # solvePnp
    R_solvePnp_pair2_common, t_solvePnp_pair2_common = estimate_solvePnP(triangulated_points_pair1_common, filtered_common_536, cam536)
    R_solvePnp_pair3_common, t_solvePnp_pair3_common = estimate_solvePnP(triangulated_points_pair1_common, filtered_common_520, cam520)
    R_solvePnp_pair1_common, t_solvePnp_pair1_common = estimate_solvePnP(triangulated_points_pair1_common, filtered_common_517, cam517)

    # SolvePNP does not give the rot and trans between two cameras but from the object to the camera !!
    T1_obj = np.zeros((4, 4))
    T1_obj[0:3, 0:3] = R_solvePnp_pair1_common
    T1_obj[0:3, 3] = t_solvePnp_pair1_common.ravel()
    T1_obj[3, 3] = 1
    T1_obj = np.linalg.inv(T1_obj)

    R_solvePnp_pair1_common = np.linalg.norm(np.dot(R_solvePnp_pair1_common,np.array([20,20,20])))
    print("lin alg",R_solvePnp_pair1_common)
    T3_obj = np.zeros((4,4))
    T3_obj[0:3, 0:3] = R_solvePnp_pair2_common
    T3_obj[0:3, 3] = t_solvePnp_pair2_common.ravel()
    T3_obj[3, 3] = 1
    #T3_obj = np.linalg.pinv(T3_obj)

    T4_obj = np.zeros((4,4))
    T4_obj[0:3, 0:3] = R_solvePnp_pair3_common
    T4_obj[0:3, 3] = t_solvePnp_pair3_common.ravel()
    T4_obj[3, 3] = 1
    #T4_obj = np.linalg.inv(T4_obj)

    T12 = np.zeros((4,4))
    T12[0:3, 0:3] = R_recoverPose_pair1_common
    T12[0:3, 3] = t_recoverPose_pair1_common.ravel()
    T12[3, 3] = 1

    T13 = np.dot(T3_obj, T1_obj)
    #T13 = np.linalg.inv(T13)
    R_13 = T13[:3, :3]  # Rotation matrix is the top-left 3x3 submatrix
    t_13 = T13[:3, 3]  # Translation vector is the right-most column

    T14 = np.dot(T4_obj,T1_obj)
    R_14 = T14[:3, :3]  # Rotation matrix is the top-left 3x3 submatrix
    t_14 = T14[:3, 3]  # Translation vector is the right-most column

    # Calculate Reprojection error with calculate_reprojection_error()
    reproj_error_pair1_common = calculate_reprojection_error(cam518, triangulated_points_pair1_common, filtered_common_518, R_recoverPose_pair1_common, t_recoverPose_pair1_common)

    reproj_error_pair2_common = calculate_reprojection_error(cam536, triangulated_points_pair1_common, filtered_common_536, R_13, t_13)
    reproj_error_pair3_common = calculate_reprojection_error(cam520, triangulated_points_pair1_common,filtered_common_520, R_14, t_14)

    # Results
    # Scaling with the real distance between sensors
    real_distance_cm_pair1 = 592
    real_distance_cm_pair2 = 762
    real_distance_cm_pair3 = 481

    # Scale factor
    scale_factor_pair1_common, scaled_translation_pair1_common = scale_translation(t_recoverPose_pair1_common, real_distance_cm_pair1)
    scale_factor_pair2_common, scaled_translation_pair2_common = scale_translation(t_13, real_distance_cm_pair2)
    #scale_factor_pair22_common, scaled_translation_pair22_common = scale_translation(t_recoverPose_pair2_common, real_distance_cm_pair2)
    scale_factor_pair3_common, scaled_translation_pair3_common = scale_translation(t_14, real_distance_cm_pair3)

    # Decompose rotation matrix
    rot_decomp_pair1_common = decompose_rotation(R_recoverPose_pair1_common)
    rot_decomp_pair2_common = decompose_rotation(R_13)
    #rot_decomp_pair22_common = decompose_rotation(R_recoverPose_pair2_common)
    rot_decomp_pair3_common = decompose_rotation(R_14)

    # Number of keypoints
    num_keypoints_pair1 = len(filtered_common_517)
    num_keypoints_pair2 = len(filtered_common_518)
    num_keypoints_pair3 = len(filtered_common_536)

    # Print results
    print('Pair 1 - translation from camera 517 to 518 - UBIQISENSE PIPELINE:',
          "[-483.67437729 -122.77347367  348.15576036] (cm) ")
    print('Pair 1 - translation from camera 517 to 518 - recoverPose:', scaled_translation_pair1_common.ravel())
    print('Pair 1 - rotation from camera 517 to 518 - UBIQISENSE PIPELINE:',
          "( 47.95943056 -67.13253629 -62.56659299) (deg.) ")
    print('Pair 1 - rotation from camera 517 to 518 - recoverPose:', rot_decomp_pair1_common)
    print('Pair 1 - scale factor:', scale_factor_pair1_common)
    print('Number of keypoints:', num_keypoints_pair1, "/ Threshold: ", threshold_common, "/ Reprojection error:",reproj_error_pair1_common)
    print('')

    print('Pair 2 - translation from camera 517 to 536 - UBIQISENSE PIPELINE:',"[-188.76428849 -211.59882113  731.79469696] (cm) ")
    print('Pair 2 - translation from camera 517 to 536 - solvePnP:', scaled_translation_pair2_common.ravel())
    print('Pair 2 - rotation from camera 517 to 536 - UBIQISENSE PIPELINE:',"( 150.39854946   13.33519403 -177.79742376) (deg.) ")
    print('Pair 2 - rotation from camera 517 to 536 - solvePnP:', rot_decomp_pair2_common)
    print('Pair 2 - scale factor:', scale_factor_pair2_common)
    print('Number of keypoints:', num_keypoints_pair2, "/ Threshold: ", threshold_common, "/ Reprojection error:",reproj_error_pair2_common)
    print('')

    print('Pair 3 - translation from camera 517 to 520 - UBIQISENSE PIPELINE:',
          "[306.34257461 -94.6791296  380.83349388] (cm) ")
    print('Pair 3 - translation from camera 517 to 520 - solvePnP:', scaled_translation_pair3_common.ravel())
    print('Pair 3 - rotation from camera 517 to 520 - UBIQISENSE PIPELINE:',
          "( 132.62900666  69.60734076 151.97386786) (deg.) ")
    print('Pair 3 - rotation from camera 517 to 520 - solvePnP:', rot_decomp_pair3_common)
    print('Pair 3 - scale factor:', scale_factor_pair3_common)
    print('Number of keypoints:', num_keypoints_pair3, "/ Threshold: ", threshold_common, "/ Reprojection error:",reproj_error_pair3_common)
    print('')

    overlay_images(sorted(glob(f"{args.image_path_pair1_517_518}/517*.png")),filtered_keypoints_pair1_517)
    #draw_skeleton(images_common_517, filtered_common_517,  )

    #Visualize
    # visualize_3d_points(triangulated_points_pair1_517_518)
    # visualize_3d_points(triangulated_points_pair2_517_536)
    # visualize_3d_points(triangulated_points_pair3_517_520)

    #print(reprojected_points) remove a dim!!
    #visualize_reprojection_error(filtered_keypoints_pair1_518, reprojected_points)
    #visualize_reprojection_error_heatmap(filtered_keypoints_pair1_518, reprojected_points)
    #visualize_reprojection_error_scatter(filtered_keypoints_pair1_518, reprojected_points)


