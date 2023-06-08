import numpy as np
import cv2
import sys
import os
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
def extract_keypoints(images_cam517, images_cam518, opWrapper):
    """
    Extract keypoints and confidences for each image
    """
    images_cam1 = images_cam517
    images_cam2 = images_cam518
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

    print('Printing keypoints1 !!!!!!!!!!!!',keypoints1, keypoints1.shape)
    print('Printing keypoints2 !!!!!!!!!!!!',keypoints2, keypoints2.shape)

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
    confidence_scores1 = keypoints1[:, 3].astype(np.float32)
    confidence_scores2 = keypoints2[:, 3].astype(np.float32)

    # Filter by confidence score threshold
    valid_keypoints1 = confidence_scores1 >= threshold
    valid_keypoints2 = confidence_scores2 >= threshold

    # Ensure the same keypoints are valid in both images
    valid_keypoints = valid_keypoints1 & valid_keypoints2

    filtered_keypoints1 = keypoints1[valid_keypoints, :2].astype(np.float32)
    filtered_keypoints2 = keypoints2[valid_keypoints, :2].astype(np.float32)

    print("Filter:", filtered_keypoints1)
    print("Filter2:", filtered_keypoints2)
    return filtered_keypoints1, filtered_keypoints2

# Estimate functions
def estimate_relative_pose(cameras, filtered_keypoints1, filtered_keypoints2):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist1 = cameras[0]['distortion']
    dist2 = cameras[1]['distortion']
    # Split the keypoints array into two arrays, one for each camera
    keypoints1 = filtered_keypoints1
    keypoints2 = filtered_keypoints2

    camera_matrix = np.array([[1, 0, 60],
                              [0, 1, 35],
                              [0, 0, 1]])

    keypoints1 = keypoints1.reshape(-1, 2).astype(np.float32)
    keypoints2 = keypoints2.reshape(-1, 2).astype(np.float32)

    #keypoints1 = cv2.undistortPoints(np.expand_dims(keypoints1, axis=1), cameraMatrix=camera_matrix, distCoeffs=None)
    #keypoints2 = cv2.undistortPoints(np.expand_dims(keypoints2, axis=1), cameraMatrix=camera_matrix, distCoeffs=None)

    print("new element: ",keypoints1)
    print("new element2: ",keypoints2)

    # Estimate the essential matrix
    #E, mask = cv2.findEssentialMat(keypoints1, keypoints2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1)

    _ ,E , R_recoverPose, t_recoverPose, _= cv2.recoverPose(keypoints1, keypoints2, K1,dist1, K2, dist2)

    #recoverPose(points1, points2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, E, R, t, mask);
    print(f"recoverPose - Relative pose from camera cam1 to camera cam2:")
    print("Rotation recoverPose:")
    print(R_recoverPose)
    print("Translation recoverPose:")
    print(t_recoverPose)

    return K1, K2, R_recoverPose, t_recoverPose

def estimate_solvePnP(triangulated_points, filtered_keypoints2, cameras):
    # Using OpenCV's solvePnP function to estimate pose.

    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Processing data for solvePnP
    triangulated_points = triangulated_points[:, :3]
    triangulated_points = np.float32(triangulated_points)
    filtered_keypoints2 = np.float32(filtered_keypoints2)

    # Solving PnP
    ret, rvec, tvec_solvePnP = cv2.solvePnP(triangulated_points, filtered_keypoints2, K2, distCoeffs=dist)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnP, _ = cv2.Rodrigues(rvec)

    print(f"solvePnP - Relative pose from camera cam1 to camera cam2:")
    print("Rotation solvePnP: ")
    print(R_solvePnP)
    print("Translation solvePnP:")
    print(tvec_solvePnP)
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
    points_3d = np.float32(points_3d)
    filtered_keypoints2 = np.float32(filtered_keypoints2)

    # Solving PnPRansac
    ret, rvec, tvec_solvePnPRansac, inliers = cv2.solvePnPRansac(points_3d, filtered_keypoints2, K2, distCoeffs=dist)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRansac, _ = cv2.Rodrigues(rvec)

    print(f"solvePnPRansac - Relative pose from camera cam1 to camera cam2:")
    print("Rotation solvePnPRansac: ")
    print(R_solvePnPRansac)
    print("Translation solvePnPRansac:")
    print(tvec_solvePnPRansac)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRansac, tvec_solvePnPRansac
def estimate_solvePnPRefineLM(points_3d, filtered_keypoints2, cameras):
    # Using OpenCV's solvePnPRefineLM function to estimate pose.

    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Processing data for solvePnPRefineLM
    points_3d = points_3d[:, :3]
    points_3d = np.float32(points_3d)
    filtered_keypoints2 = np.float32(filtered_keypoints2)

    # Initial pose estimation with solvePnP
    _, rvec_init, tvec_init = cv2.solvePnP(points_3d, filtered_keypoints2, K2, distCoeffs=dist)


    # Refine pose estimation with solvePnPRefineLM
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(points_3d, filtered_keypoints2, K2, distCoeffs=dist, rvec=rvec_init, tvec=tvec_init)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRefineLM, _ = cv2.Rodrigues(rvec_refined)

    print(f"solvePnPRefineLM - Relative pose from camera cam1 to camera cam2:")
    print("Rotation solvePnPRefineLM: ")
    print(R_solvePnPRefineLM)
    print("Translation solvePnPRefineLM:")
    print(tvec_refined)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRefineLM, tvec_refined

# Reprojections
def calculate_reprojection_error_recoverPose(triangulated_points, filtered_keypoints2, R_recoverPose, t_recoverPose, cameras):
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

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

def triangulate_points(K1, K2, R_recoverPose, t_recoverPose, filtered_keypoints1, filtered_keypoints2):
    # Step 1: Construct the projection matrices for the two cameras
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1))))) # [I|0] for the first camera
    P2 = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose))) # [R|t] for the second camera

    # Step 2: Unpack the keypoint matches to separate arrays
    points1 = np.float32(filtered_keypoints1).reshape(-1, 2).T
    points2 = np.float32(filtered_keypoints2).reshape(-1, 2).T

    # Step 3: Call the triangulatePoints function
    homogeneous_3d_points = cv2.triangulatePoints(P1, P2, points1, points2)

    # Step 4: Convert the points from homogeneous to euclidean coordinates
    euclidean_3d_points = homogeneous_3d_points[:3, :] / homogeneous_3d_points[3, :]

    # Step 5: Reshape the points to a Nx3 array
    triangulated_points = euclidean_3d_points.T

    return triangulated_points

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


import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = ArgumentParser(description="Pairwise camera calibration using OpenPose")
    parser.add_argument("--image_path", type=str, default="data/Calibration_sets/517_518/", help="Path to the directory containing input images")
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

    # Load images and extract keypoints for each camera
    images_cam517 = sorted(glob(f"{args.image_path}/517*.png"))
    images_cam518 = sorted(glob(f"{args.image_path}/518*.png"))
    images_cam536 = sorted(glob(f"{args.image_path}/536*.png"))
    images_cam520 = sorted(glob(f"{args.image_path}/520*.png"))

    keypoints1, keypoints2, confidences1, confidences2 = extract_keypoints(images_cam517, images_cam518, opWrapper)

    # Filter keypoints based on confidence level and presence in both keypoint sets
    threshold = 0.8
    filtered_keypoints1, filtered_keypoints2 = filter_keypoints(keypoints1, keypoints2, threshold)

#     filtered_keypoints2 = np.array([[1170, 480],
# [1135, 489],
# [1115, 487],
# [1092, 494],
# [1092, 500],
# [1051, 510],
# [1114, 462],
# [1080, 469],
# [1079, 475],
# [1055, 480],
# [1036, 479],
# [995, 488],
# [1097, 459],
# [1075, 464],
# [1017, 476],
# [991, 482],
# [1064, 448],
# [1041, 453],
# [983, 464],
# [957, 469],
# [1058, 443],
# [1023, 451],
# [979, 459],
# [939, 466],
# [1012, 429],
# [977, 435],
# [932, 442],
# [892, 449]])
#
#     filtered_keypoints1 = np.array([[565, 432],
# [618, 413],
# [637, 413],
# [669, 402],
# [674, 395],
# [718, 380],
# [612, 451],
# [666, 430],
# [668, 423],
# [700, 412],
# [721, 411],
# [766, 393],
# [632, 451],
# [668, 438],
# [739, 411],
# [769, 399],
# [667, 465],
# [702, 450],
# [774, 422],
# [804, 410],
# [669, 473],
# [724, 450],
# [778, 429],
# [824, 410],
# [727, 497],
# [782, 472],
# [837, 448],
# [881, 428]])

#     filtered_keypoints1 = np.array([[783, 472],
# [695, 635],
# [966, 365],
# [578, 319],
# [756, 280],
# [848,550]])
#
#     filtered_keypoints2 = np.array([[976, 434],
# [1012, 368],
# [902, 266],
# [1278, 541],
# [1041, 639],
# [941,372]])

    # Estimate relative pose for each camera pair
    K1, K2, R_recoverPose, t_recoverPose = estimate_relative_pose(cameras, filtered_keypoints1, filtered_keypoints2)

    """ Triangulation """
    # Assuming K1, K2, R, t, filtered_keypoints1, and filtered_keypoints2 are defined
    triangulated_points = triangulate_points(K1, K2, R_recoverPose, t_recoverPose, filtered_keypoints1, filtered_keypoints2)
    print('TRIANGULATION:', triangulated_points)

    # Solve homo
    H = getHomography(
        points1=filtered_keypoints1,
        points2=filtered_keypoints2,
    )
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K1)

    for i in range(4):

        degrees, _, _, _, _, _ = cv2.RQDecomp3x3(Rs[i])
        print("_____________________________________________")
        print(Rs[i])
        print(i,degrees)

    '''Estimate functions''' ######################
    R_solvePnP, tvec_solvePnP = estimate_solvePnP(triangulated_points,filtered_keypoints2, cameras)
    R_solvePnPRansac, tvec_solvePnPRansac = estimate_solvePnPRansac(triangulated_points, filtered_keypoints2, cameras)
    R_solvePnPRefineLM, tvec_refined = estimate_solvePnPRefineLM(triangulated_points, filtered_keypoints2, cameras)

    rot517_518 = np.array([[-0.73405269, -0.67852982 ,-0.0276393 ],
 [ 0.22701519, -0.20682577 ,-0.951676  ],
 [-0.64002402 , 0.70485487, -0.30585759]])

    rot517 = np.array([[-0.7994412 ,  0.5993115,   0.04146691],
     [-0.18165657 ,- 0.17536862, - 0.96759844],
     [0.57262087 , 0.78107079 ,- 0.24906577]])
    rot517518, _, _, _, _, _ = cv2.RQDecomp3x3(rot517)
    degreesrefine, _, _, _, _, _ = cv2.RQDecomp3x3(R_solvePnPRefineLM)
    degreespnp, _, _, _, _, _ = cv2.RQDecomp3x3(R_solvePnP)
    degreesransac, _, _, _, _, _ = cv2.RQDecomp3x3(R_solvePnPRansac)
    degreesrecover, _, _, _, _, _ = cv2.RQDecomp3x3(R_recoverPose)
    print("rot517_518 Solution:", rot517518)
    print("recoverPose:", degreesrecover)
    print("PNP:", degreespnp)
    print("RansacPNP:", degreesransac)
    print("RefinedPNP:", degreesrefine)

    '''Reprojection error''' #######################
    reprojection_error_recov = calculate_reprojection_error_recoverPose(triangulated_points, filtered_keypoints2, R_recoverPose, t_recoverPose, cameras)
    print('Reprojection error recoverPose:', reprojection_error_recov)

    reprojection_error_pnp = calculate_reprojection_error_solvePnP(triangulated_points, filtered_keypoints2, R_solvePnP, tvec_solvePnP, cameras)
    print('Reprojection error solvepnp:', reprojection_error_pnp)

    reprojection_error_PnPRansac = calculate_reprojection_error_solvePnPRansac(triangulated_points, filtered_keypoints2, R_solvePnPRansac, tvec_solvePnPRansac, cameras)
    print('Reprojection error solvepnpRansac:', reprojection_error_PnPRansac)

    reprojection_error_PnPLM = calculate_reprojection_error_solvePnPLM(triangulated_points, filtered_keypoints2, R_solvePnPRefineLM, tvec_refined, cameras)
    print('Reprojection error solvepnpLM:', reprojection_error_PnPLM)

    # Scaling with the real distance between sensors
    real_distance_cm = 600

    current_distance = np.linalg.norm(t_recoverPose)
    current_distance_solvePnP = np.linalg.norm(tvec_solvePnP)
    current_distance_solvePnPRefineLM = np.linalg.norm(tvec_refined)

    scale_factor = real_distance_cm / current_distance
    scale_factor_solvePnP = real_distance_cm / current_distance_solvePnP
    scale_factor_solvePnPRefineLM = real_distance_cm / current_distance_solvePnPRefineLM

    print('Print scale factor: ', scale_factor, '/ scale factor SolvePnP:',scale_factor_solvePnP,'/ scale factor SolvePnP:',scale_factor_solvePnPRefineLM)

    T_scaled_cm = scale_factor * t_recoverPose
    T_scaled_cm_solvePnP = scale_factor_solvePnP * tvec_solvePnP
    T_scaled_cm_solvePnPRefineLM = scale_factor_solvePnPRefineLM * tvec_refined

    print('Print t_scaled_cm:', T_scaled_cm)
    print('Print t_scaled_cm_solvePnP:', T_scaled_cm_solvePnP)
    print('Print t_scaled_cm_solvePnPRefineLM:', T_scaled_cm_solvePnPRefineLM)

    num_keypoints = filtered_keypoints1.shape[0]
    print('Number of keypoints:',num_keypoints)
    # Visualize
    visualize_3d_points(triangulated_points)






