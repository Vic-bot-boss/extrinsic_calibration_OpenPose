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

def load_camera_params(calib_file_path, cam_num_1, cam_num_2):
    """
    Load camera calibration parameters for a pair of cameras from npz files
    """
    camera_files = [f"{calib_file_path}/Calib_Parameters{cam_num_1}.npz", f"{calib_file_path}/Calib_Parameters{cam_num_2}.npz"]
    cameras = []
    for camera_file in camera_files:
        with np.load(camera_file) as data:
            mtx, dist, frame_size, pxsize = [data[f] for f in ('mtx', 'dist', 'frame_size', 'pxsize')]
            cameras.append({'matrix': mtx, 'distortion': dist, 'frame_size': frame_size, 'pxsize': pxsize})
    print(f"Loaded intrinsics for cameras {cam_num_1} and {cam_num_2}")
    print(cameras)
    return cameras

def extract_keypoints(images, opWrapper):
    """
    Extract keypoints and confidences for each image
    """
    keypoints = []
    confidences = []
    confidence_scores = []

    for image in images:
        img = cv2.imread(image)
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        mapping = op.getPoseBodyPartMapping(op.PoseModel.BODY_25)
        strings = np.array([value for value in mapping.values()]).reshape(-1, 1)[:-1]

        keypoints.append(datum.poseKeypoints[0][:, :2]) # X, Y
        confidence_scores.append(datum.poseKeypoints[0][:, 2]) # Confidence score each keypoint
        confidences.append(datum.poseScores[0]) # Average confidence

    keypoints = np.array(keypoints)
    confidences = np.array(confidences)
    confidence_scores = np.array(confidence_scores)

    # Split the keypoints array into two arrays, one for each camera
    keypoints1 = keypoints[0][:, :2]
    keypoints1 = np.hstack((keypoints1, strings))
    keypoints2 = keypoints[1][:, :2]
    keypoints2 = np.hstack((keypoints2, strings))

    #print(f"Number of keypoints for first image {len(keypoints[0])} and shape {keypoints[0].shape}")
    #print(f"Number of keypoints for second image {len(keypoints[1])} and shape {keypoints[1].shape}")
    #print(f"Confidence levels for first image {confidence_scores[0]} and shape {confidence_scores[0].shape}")
    print(f"Confidence levels for second image {confidence_scores[1]} and shape {confidence_scores[0].shape}")

    #print(keypoints)
    #print("strings", strings)
    #print("k1", keypoints1)
    #print("k2", keypoints2)

    #poseModel = op.PoseModel.BODY_25
    #print(op.getPoseBodyPartMapping(op.PoseModel.BODY_25))
    #print(op.getPoseNumberBodyParts(poseModel))
    #print(op.getPosePartPairs(poseModel))
    #print(op.getPoseMapIndex(poseModel))

    return keypoints1, keypoints2, confidences, confidence_scores

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

    #np.set_printoptions(formatter={'float': lambda x: format(int(x), '')})

    print("Print charuco corners: ", charuco_corn1)
    print("Print charuco corners: ", charuco_corn2)
    print("Print ArUco IDs coordinates: ", new_elements)

    return charuco_corn1, charuco_corn2, aruco_ids_coords_all


def filter_keypoints(keypoints1,keypoints2, confidence_scores, threshold):
    """
    Filter keypoints based on the confidence score threshold and keep only the keypoints with matching IDs.

    Args:
        keypoints1 (np.ndarray): First keypoints array with shape (N, 3) where N is the number of keypoints.
        keypoints2 (np.ndarray): Second keypoints array with shape (N, 3) where N is the number of keypoints.
        confidence_scores (np.ndarray): Confidence scores array with shape (2N,) for each keypoint.
        threshold (float): Confidence score threshold, keypoints with scores below this value will be removed.

    Returns:
        np.ndarray, np.ndarray: Filtered keypoints arrays for keypoints1 and keypoints2 with matching IDs.
    """
    filtered_keypoints1 = []
    filtered_keypoints2 = []

    confidence_scores1 = confidence_scores[0]
    confidence_scores2 = confidence_scores[1]

    # Filter by confidence score threshold
    filtered_keypoints1 = keypoints1[confidence_scores1 >= threshold]
    filtered_keypoints2 = keypoints2[confidence_scores2 >= threshold]

    # Filter by common IDs
    common_ids = np.intersect1d(filtered_keypoints1[:, 2], filtered_keypoints2[:, 2])
    filtered_keypoints1 = filtered_keypoints1[np.isin(filtered_keypoints1[:, 2], common_ids)]
    filtered_keypoints2 = filtered_keypoints2[np.isin(filtered_keypoints2[:, 2], common_ids)]
    print(filtered_keypoints1)
    # Remove IDs
    filtered_keypoints1 = filtered_keypoints1[:, :2].astype(np.float32)
    filtered_keypoints2 = filtered_keypoints2[:, :2].astype(np.float32)
    print(filtered_keypoints1)
    return filtered_keypoints1, filtered_keypoints2

def estimate_relative_pose(cameras, charuco_corn1, charuco_corn2):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    # Split the keypoints array into two arrays, one for each camera
    #keypoints1 = filtered_keypoints1
    #keypoints2 = filtered_keypoints2
    keypoints1 = charuco_corn1
    keypoints2 = charuco_corn2

    print("new element: ",keypoints2)
    # Estimate the essential matrix
    E, mask = cv2.findEssentialMat(keypoints1, keypoints2, K1, method=cv2.RANSAC, prob=0.99, threshold=1.0)
    _, R_recoverPose, t_recoverPose, _ = cv2.recoverPose(E, keypoints1, keypoints2)

    # PnP
    #object_points = keypoints1.astype('float32') # 3D
    #keypoints2 = keypoints2.astype('float32')
    #retval, R_pnp, t_pnp = cv2.solvePnP(object_points, keypoints2, K2, None) # object_points must be 3D


    print(f"recoverPose - Relative pose from camera cam1 to camera cam2:")
    print("Rotation recoverPose:")
    print(R_recoverPose)
    print("Translation recoverPose:")
    print(t_recoverPose)

    print(f"solvePnP - Relative pose from camera cam1 to camera cam2:")
    print("Rotation solvePnP: coming soon")
    #print(R_pnp)
    print("Translation solvePnP:coming soon")
    #print(t_pnp)

    euler_angles = matrix_to_euler(R_recoverPose)
    print('Rotation matrix in euler angles:',euler_angles)

    rot = Rotation.from_matrix(R_recoverPose)
    angles = rot.as_euler("xyz", degrees=True)
    print(angles)
    angles[0] += 5
    rot = Rotation.from_euler("xyz", angles, degrees=True)
    new_rotation_matrix = rot.as_matrix()
    #print(new_rotation_matrix)

    return K1, K2, R_recoverPose, t_recoverPose

def estimate_solvePnPRefineLM(points_3d, charuco_corn2, cameras):
    # Using OpenCV's solvePnPRefineLM function to estimate pose.

    # Extract intrinsic parameters
    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    filtered_keypoints2 = charuco_corn2
    # Processing data for solvePnPRefineLM
    points_3d = points_3d[:, :3]
    points_3d = np.float32(points_3d)
    filtered_keypoints2 = np.float32(filtered_keypoints2)

    # Initial pose estimation with solvePnP
    _, rvec_init, tvec_init = cv2.solvePnP(points_3d, filtered_keypoints2, K2, distCoeffs=None)

    # Refine pose estimation with solvePnPRefineLM
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(points_3d, filtered_keypoints2, K2, distCoeffs=None,
                                                      rvec=rvec_init, tvec=tvec_init)

    # Converting the rotation vector to a rotation matrix.
    R_solvePnPRefineLM, _ = cv2.Rodrigues(rvec_refined)

    print(f"solvePnPRefineLM - Relative pose from camera cam1 to camera cam2:")
    print("Rotation solvePnPRefineLM: ")
    print(R_solvePnPRefineLM)
    print("Translation solvePnPRefineLM:")
    print(tvec_refined)

    # Returning the estimated rotation matrix and translation vector.
    return R_solvePnPRefineLM, tvec_refined

def triangulate_points(K1, K2, R_recoverPose, t_recoverPose, charuco_corn1, charuco_corn2):

    filtered_keypoints1 = charuco_corn1
    filtered_keypoints2 = charuco_corn2

    num_points = filtered_keypoints1.shape[0]
    points_3d = np.zeros((num_points, 4))

    # Compute the projection matrix for the first camera
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Compute the projection matrix for the second camera
    P2 = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose)))

    for i in range(num_points):
        kp1 = np.append(filtered_keypoints1[i], 1)
        kp2 = np.append(filtered_keypoints2[i], 1)

        A = np.zeros((4, 4))

        A[0:2, :] = np.outer(kp1, P1[2, :] - P1[0, :])[:2]
        A[2:4, :] = np.outer(kp2, P2[2, :] - P2[0, :])[:2]

        _, _, V = svd(A)
        X = V[-1, :4]
        X /= X[-1]
        points_3d[i, :] = X

    return points_3d


def reproject_points(points_3d, K2, R_recoverPose, t_recoverPose):
    num_points = points_3d.shape[0]
    points_homogeneous = np.hstack((points_3d[:, :3], np.ones((num_points, 1))))

    P = np.dot(K2, np.hstack((R_recoverPose, t_recoverPose)))
    points_2d_homogeneous = np.dot(P, points_homogeneous.T).T

    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2, np.newaxis]
    return points_2d

def matrix_to_euler(matrix):
    sy = np.sqrt(matrix[0, 0] * matrix[0, 0] +  matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(matrix[2, 1], matrix[2, 2])
        y = np.arctan2(-matrix[2, 0], sy)
        z = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        x = np.arctan2(-matrix[1, 2], matrix[1, 1])
        y = np.arctan2(-matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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

def applyHomography(
        img,
        homography
):
    """

    :param img:
    :param homography:

    :return:

    """
    height, width = img.shape
    return cv2.warpPerspective(
        src=img,
        M=homography,
        dsize=(width, height)
    )


if __name__ == '__main__':
    parser = ArgumentParser(description="Pairwise camera calibration using OpenPose")
    parser.add_argument("--image_path", type=str, default="data/charuco/", help="Path to the directory containing input images")
    parser.add_argument("--calib_path", type=str, default="data/intrinsic/", help="Path to the directory containing calibration files")
    args = parser.parse_args()

    # Load camera calibration parameters for each camera
    cameras = load_camera_params(args.calib_path, 518,520) # pick which camera pair you are calibrating
    print('Intrinsics loaded!')

    # Load OpenPose
    params = dict()
    params["model_folder"] = "openpose/models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Load images and extract keypoints for each camera
    images = sorted(glob(f"{args.image_path}/*.png"))
    keypoints1, keypoints2, confidences, confidence_scores = extract_keypoints(images, opWrapper)

    charuco_corn1,charuco_corn2, aruco_ids_coords_all = detect_charuco_diamonds(images)




    K1 = cameras[0]['matrix']
    K2 = cameras[1]['matrix']
    dist = cameras[1]['distortion']

    H = getHomography(
        points1=charuco_corn1,
        points2=charuco_corn2,
    )
    _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K1)

    for i in range(4):

        yaw, pitch, roll, _, _, _ = cv2.RQDecomp3x3(Rs[i])
        print("_____________________________________________")
        print(Rs[i])
        print(i,yaw)



    # Decompose homography to rotation, translation and plane normal.
    # _, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, CAM_MATRIX)


    # Filter keypoints based on confidence level and presence in both keypoint sets
    threshold = 0.7
    filtered_keypoints1, filtered_keypoints2 = filter_keypoints(keypoints1,keypoints2, confidence_scores, threshold)

    # Print confidence levels for each image
    for i, conf in enumerate(confidences):
        print(f"Average confidence score for image {i+1}: {conf}")

    # Match keypoints between pairs of cameras
    #matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) # different matching algos
    #matches = match_keypoints(keypoints, matcher)

    # Estimate relative pose for each camera pair
    K1, K2, R_recoverPose, t_recoverPose = estimate_relative_pose(cameras, charuco_corn1, charuco_corn2)

    points_3d = triangulate_points(K1, K2, R_recoverPose, t_recoverPose, charuco_corn1, charuco_corn2)
    print(points_3d)

    R_solvePnPRefineLM, tvec_refined = estimate_solvePnPRefineLM(points_3d, charuco_corn2, cameras)

    reprojected_keypoints2 = reproject_points(points_3d, K2, R_recoverPose, t_recoverPose)
    reprojection_error = np.mean(np.sqrt(np.sum((reprojected_keypoints2 - charuco_corn2) ** 2, axis=1)))
    print('Print reprojection error:',reprojection_error)

    num_keypoints = charuco_corn1.shape[0]
    print(num_keypoints)
    rms_error = np.sqrt(reprojection_error / num_keypoints)
    print('Print RMS error: ' ,rms_error)

    real_distance_cm = 766
    current_distance = np.linalg.norm(t_recoverPose)
    scale_factor = real_distance_cm / current_distance
    print('Print scale factor: ', scale_factor)
    T_scaled_cm = scale_factor * t_recoverPose
    print('Print t_scaled_cm:', T_scaled_cm)

