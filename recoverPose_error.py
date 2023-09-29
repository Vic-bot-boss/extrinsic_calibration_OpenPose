import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from argparse import ArgumentParser
from pathlib import Path



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

def estimate_recoverPose(cam1, cam2, keypoints1, keypoints2):
    """
    Estimate essential matrix and relative pose for a pair of cameras
    """
    # Extract intrinsic parameters
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
    method=cv2.RANSAC,
    prob=0.999,
    threshold=3.0,
    mask=mask)

    return R_recoverPose, t_recoverPose

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

    return rms_reprojection_error, projected_points.reshape(-1, 2)

# Additional utility to add noise to keypoints:
def add_controlled_noise(points, noise_increment, iteration, noise_type='linear'):
    if noise_type == 'linear':
        noise = np.linspace(0, noise_increment, len(points)) * iteration
        noisy_points = points + noise.reshape(-1, 1)
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, noise_increment * iteration, points.shape)
        noisy_points = points + noise
    return noisy_points


def introduce_outliers(keypoints, percentage_outliers, max_shift=10):
    num_outliers = int(len(keypoints) * percentage_outliers)

    # Select random keypoints to perturb
    selected_keypoints = keypoints[np.random.choice(keypoints.shape[0], num_outliers, replace=False)]

    # Generate random shifts for each selected keypoint
    shifts = (np.random.rand(*selected_keypoints.shape) - 0.5) * 2 * max_shift

    # Add shifts to selected keypoints
    perturbed_keypoints = selected_keypoints + shifts

    # Concatenate the perturbed keypoints with the original keypoints
    keypoints_with_perturbations = np.vstack([keypoints, perturbed_keypoints])

    return keypoints_with_perturbations


def introduce_controlled_outliers(keypoints1, keypoints2, num_outliers_added, offset_radius=5):
    assert keypoints1.shape == keypoints2.shape, "Both keypoints arrays should have the same shape"

    # Ensure num_outliers doesn't exceed the number of keypoints
    num_outliers = min(num_outliers_added, len(keypoints1))

    outliers1 = []
    outliers2 = []


    for _ in range(int(num_outliers_added)):

        # Randomly select an index
        idx = np.random.choice(keypoints1.shape[0])

        # Get the corresponding keypoint from both images
        selected_keypoint1 = keypoints1[idx]
        selected_keypoint2 = keypoints2[idx]

        # Generate a random offset within the given radius for both x and y coordinates for image 1
        offset1 = (np.random.rand(2) - 0.5) * 2 * offset_radius  # This produces values between -offset_radius and offset_radius

        # Generate a different random offset for image 2
        offset2 = (np.random.rand(2) - 0.5) * 2 * offset_radius  # This produces values between -offset_radius and offset_radius

        # Add this offset to the selected keypoints
        outlier1 = selected_keypoint1 + offset1
        outlier2 = selected_keypoint2 + offset2

        outliers1.append(outlier1)
        outliers2.append(outlier2)

    # Convert outliers lists to numpy arrays
    outliers1 = np.array(outliers1)
    outliers2 = np.array(outliers2)

    # Concatenate the outliers with the original keypoints
    if len(outliers1) > 0 and len(outliers2) > 0:
        keypoints_with_outliers1 = np.vstack([keypoints1, outliers1])
        keypoints_with_outliers2 = np.vstack([keypoints2, outliers2])
    else:
        keypoints_with_outliers1 = keypoints1.copy()
        keypoints_with_outliers2 = keypoints2.copy()

    return keypoints_with_outliers1, keypoints_with_outliers2



# Baseline experiment function
def baseline_experiment(cam1, cam2, keypoints1, keypoints2):
    R, t = estimate_recoverPose(cam1, cam2, keypoints1, keypoints2)
    print(f"R: {R}, t: {t}")
    triangulated_pts = triangulate_points(cam1, cam2, keypoints1, keypoints2, R, t)
    reprojection_err , proj_pts= calculate_reprojection_error(cam2, triangulated_pts, keypoints2, R, t)
    print(f"Reprojection Error: {reprojection_err}")
    #visualize_reprojected_points(keypoints2, proj_pts)
    return R, t, reprojection_err


def visualize_reprojection_error_scatter(initial_points, reprojected_points, num_outliers):
    # Compute the Euclidean distance between initial points and reprojected points
    errors = np.linalg.norm(initial_points - reprojected_points, axis=1)

    # Set up the figure and axes
    fig, ax = plt.subplots()

    # Plot the original points with blue color
    scatter_original = ax.scatter(initial_points[:, 0], initial_points[:, 1],
                                  c=errors, cmap='cool', s=100, label='Original Points')

    # If outliers exist, plot them with red color
    if num_outliers > 0:
        # Assuming the outliers are at the end of the list
        ax.scatter(reprojected_points[-num_outliers:, 0], reprojected_points[-num_outliers:, 1],
                   color='red', s=100, label='Outliers', marker='x')

    # Add a colorbar
    cbar = fig.colorbar(scatter_original)
    cbar.set_label('Reprojection Error')

    # Set axis labels, title, and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Reprojection Error Scatter Plot')
    ax.legend()

    # Optionally, annotate each point with its index
    for i, point in enumerate(initial_points):
        ax.annotate(str(i), (point[0], point[1]))

    # Show the plot
    plt.show()

def visualize_reprojected_points(original_points, reprojected_points):
    """
    Visualize the original and reprojected points on a 2D plane.
    """
    fig, ax = plt.subplots()

    # Plot the original points in blue
    ax.scatter(original_points[:, 0], original_points[:, 1], c='blue', label='Original Points', marker='o')

    # Plot the reprojected points in red
    ax.scatter(reprojected_points[:, 0], reprojected_points[:, 1], c='red', label='Reprojected Points', marker='x')

    # Draw lines between corresponding points
    for orig, reprj in zip(original_points, reprojected_points):
        ax.plot([orig[0], reprj[0]], [orig[1], reprj[1]], 'k-', lw=0.5)

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Original vs Reprojected Points')
    ax.legend()

    # Show the plot
    plt.show()

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

def outlier_experiment(cam1, cam2, keypoints1, keypoints2, outlier_increment=1 , num_iterations=7):
    outliers_count = []  # This will now track the absolute number of outliers added
    num_keypoints = []
    rmse_rotations = []
    rmse_translations = []
    reproj_errors = []

    # Ground truth for comparison
    R_charuco, t_charuco = estimate_recoverPose(cam1, cam2, keypoints1, keypoints2)
    print(f"Ground Truth R: {R_charuco}, t: {t_charuco}")
    triangulated_pts = triangulate_points(cam1, cam2, keypoints1, keypoints2, R_charuco, t_charuco)
    reproj_error, proj_pts = calculate_reprojection_error(cam2, triangulated_pts, keypoints2, R_charuco, t_charuco)
    print(f"Ground Truth Reprojection Error: {reproj_error}")

    num_outliers_added = 0  # Start at negative of increment so the first iteration adds 0


    for i in range(1, num_iterations + 1):
        # num_outliers_added += outlier_increment
        # Increment only if it's not the first iteration
        if i > 1:
            num_outliers_added += outlier_increment
        print(f"Iteration {i}: Adding {num_outliers_added} outliers")  # Debugging print
        # Round the number of outliers to add

        keypoints1_outliers, keypoints2_outliers = introduce_controlled_outliers(keypoints1, keypoints2,
                                                                                 num_outliers_added)

        # # Print the number of keypoints to verify
        # print(
        #     f"KeyPoints1: Original = {keypoints1}, With Outliers = {keypoints1_outliers}")  # Debugging print
        # print(
        #     f"KeyPoints2: Original = {keypoints2}, With Outliers = {keypoints2_outliers}")  # Debugging print

        R_noisy, t_noisy = estimate_recoverPose(cam1, cam2, keypoints1_outliers, keypoints2_outliers)
        triangulated_pts = triangulate_points(cam1, cam2, keypoints1_outliers, keypoints2_outliers, R_noisy, t_noisy)
        reproj_error, proj_pts = calculate_reprojection_error(cam2, triangulated_pts, keypoints2_outliers, R_noisy, t_noisy)

        # Convert rotation matrices to rotation vectors
        rvec_noisy, _ = cv2.Rodrigues(R_noisy)
        rvec_charuco, _ = cv2.Rodrigues(R_charuco)

        # Compute the RMSE between the two rotation vectors
        rmse_rotation = np.sqrt(np.mean((rvec_noisy - rvec_charuco) ** 2))
        rmse_translation = np.sqrt(np.mean((t_noisy - t_charuco) ** 2))

        num_keypoints.append(len(keypoints1_outliers))  # Tracking the total number of keypoints including outliers
        rmse_rotations.append(rmse_rotation)
        rmse_translations.append(rmse_translation)
        reproj_errors.append(reproj_error)
        outliers_count.append(num_outliers_added)

        # num_outliers_added += outlier_increment

    # outliers_count[0] = 0
    return outliers_count, rmse_rotations, rmse_translations, reproj_errors, num_keypoints


# Experiment with added noise
def noise_experiment(cam1, cam2, keypoints1, keypoints2, noise_type):
    noise_levels = []
    rmse_rotations = []
    rmse_translations = []
    reproj_errors = []

    # Ground truth for comparison
    R_charuco, t_charuco, _ = baseline_experiment(cam1, cam2, keypoints1, keypoints2)


    # Loop through different levels of noise
    num_iterations = 100
    noise_increment = 0.01

    for i in range(0, num_iterations + 1):
        noise_level = noise_increment * i
        noisy_keypoints1 = add_controlled_noise(keypoints1, noise_increment, i, noise_type)
        noisy_keypoints2 = add_controlled_noise(keypoints2, noise_increment, i, noise_type)

        R_noisy, t_noisy = estimate_recoverPose(cam1, cam2, noisy_keypoints1, noisy_keypoints2)
        triangulated_pts = triangulate_points(cam1, cam2, noisy_keypoints1, noisy_keypoints2, R_noisy, t_noisy)
        reproj_error, proj_pts = calculate_reprojection_error(cam2, triangulated_pts, noisy_keypoints2, R_noisy, t_noisy)

        #visualize_reprojected_points(noisy_keypoints2, proj_pts)

        # Convert rotation matrices to rotation vectors
        rvec_noisy, _ = cv2.Rodrigues(R_noisy)
        rvec_charuco, _ = cv2.Rodrigues(R_charuco)

        # Compute the RMSE between the two rotation vectors
        rmse_rotation = np.sqrt(np.mean((rvec_noisy - rvec_charuco) ** 2))
        rmse_translation = np.sqrt(np.mean((t_noisy - t_charuco) ** 2))

        noise_levels.append(noise_level)
        rmse_rotations.append(rmse_rotation)
        rmse_translations.append(rmse_translation)
        reproj_errors.append(reproj_error)

    return noise_levels, rmse_rotations, rmse_translations, reproj_errors


# Driver function to run experiments and plot
def run_experiments(cam1, cam2, keypoints1, keypoints2):
    # Run baseline
    R_base, t_base, reproj_err_base = baseline_experiment(cam1, cam2, keypoints1, keypoints2)
    print(f"Baseline RMSE Rotation: {R_base}, RMSE Translation: {t_base}, Reprojection Error: {reproj_err_base}")

    # Run linear noise experiment
    noise_levels, rmse_rotations_linear, rmse_translations_linear, reproj_errors_linear = noise_experiment(cam1, cam2,
                                                                                                           keypoints1,
                                                                                                           keypoints2,
                                                                                                           noise_type='linear')

    # Run Gaussian noise experiment
    _, rmse_rotations_gaussian, rmse_translations_gaussian, reproj_errors_gaussian = noise_experiment(cam1, cam2,
                                                                                                      keypoints1,
                                                                                                      keypoints2,
                                                                                                      noise_type='gaussian')
    print(len(noise_levels))
    print(len(reproj_errors_linear))

    plt.figure(figsize=(20, 15))

    # Plotting RMSE Rotation
    plt.subplot(3, 2, 1)
    plt.plot(noise_levels, rmse_rotations_linear, 'ro-', markersize=5, label='Linear Noise')
    plt.title('RMSE Rotation Error (Linear Noise)')
    #plt.xlabel('Noise Level')
    plt.ylabel('RMSE Rotation')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(noise_levels, rmse_rotations_gaussian, 'go-', markersize=5, label='Gaussian Noise')
    plt.title('RMSE Rotation Error (Gaussian Noise)')
    #plt.xlabel('Noise Level')
    plt.ylabel('RMSE Rotation')
    plt.legend()
    plt.grid(True)

    # Plotting RMSE Translation
    plt.subplot(3, 2, 3)
    plt.plot(noise_levels, rmse_translations_linear, 'ro-', markersize=5, label='Linear Noise')
    plt.title('RMSE Translation Error (Linear Noise)')
    #plt.xlabel('Noise Level')
    plt.ylabel('RMSE Translation')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(noise_levels, rmse_translations_gaussian, 'go-', markersize=5, label='Gaussian Noise')
    plt.title('RMSE Translation Error (Gaussian Noise)')
    #plt.xlabel('Noise Level')
    plt.ylabel('RMSE Translation')
    plt.legend()
    plt.grid(True)

    # Plotting Reprojection Error
    plt.subplot(3, 2, 5)
    plt.plot([0] + noise_levels, [reproj_err_base] + reproj_errors_linear, 'ro-', markersize=5, label='Linear Noise')
    plt.title('Reprojection Error (Linear Noise)')
    plt.xlabel('Noise Level')
    plt.ylabel('Reprojection Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot([0] + noise_levels, [reproj_err_base] + reproj_errors_gaussian, 'go-', markersize=5,
             label='Gaussian Noise')
    plt.title('Reprojection Error (Gaussian Noise)')
    plt.xlabel('Noise Level')
    plt.ylabel('Reprojection Error')
    plt.legend()
    plt.grid(True)

    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.8)
    plt.tight_layout()
    plt.show()


# Running and plotting the outlier experiment
def plot_outlier_experiment(cam1, cam2, keypoints1, keypoints2):
    iterations, rmse_rotations, rmse_translations, reproj_errors, num_keypoints = outlier_experiment(cam1, cam2, keypoints1, keypoints2)

    plt.figure(figsize=(18, 12))

    # Plotting Number of Keypoints
    plt.subplot(4, 1, 1)
    plt.plot(iterations, num_keypoints, 'go-', markersize=5, label='Number of Keypoints', color='orange')
    plt.xlabel('Number of Outliers')
    plt.ylabel('Number of Keypoints')
    plt.legend()
    plt.grid(True)

    # Plotting RMSE Rotation
    plt.subplot(4, 1, 2)
    plt.plot(iterations, rmse_rotations,'go-', markersize=5, label='RMSE Rotation', color='purple')
    plt.xlabel('Number of Outliers')
    plt.ylabel('RMSE Rotation')
    plt.legend()
    plt.grid(True)

    # Plotting RMSE Translation
    plt.subplot(4, 1, 3)
    plt.plot(iterations, rmse_translations,'go-', markersize=5, label='RMSE Translation', color='purple')
    plt.xlabel('Number of Outliers')
    plt.ylabel('RMSE Translation')
    plt.legend()
    plt.grid(True)

    # Plotting Reprojection Error
    plt.subplot(4, 1, 4)
    plt.plot(iterations, reproj_errors,'go-', markersize=5, label='Reprojection Error', color='purple')
    plt.xlabel('Number of Outliers')
    plt.ylabel('Reprojection Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



# EXPERIMENTS WITH NOISE #############################################################


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


filtered_keypoints1_charuco = np.array([[793,104], [793,508], [521,139], [521,473], [992,140], [992,472]])
filtered_keypoints2_charuco = np.array([[398,175], [398,578], [199,211], [199,542], [667,211], [667,541]])


# Run the driver function
# run_experiments(cam517, cam520, filtered_keypoints1_charuco, filtered_keypoints2_charuco)

# Run the outlier experiment
plot_outlier_experiment(cam517, cam520, filtered_keypoints1_charuco, filtered_keypoints2_charuco)

