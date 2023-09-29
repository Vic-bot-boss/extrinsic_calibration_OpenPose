import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib.image as mpimg
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from math import sqrt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.optimize import linear_sum_assignment
from math import sqrt
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

# # Load OpenPose
# params = dict()
# params["model_folder"] = "openpose/models/"
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()
# print('OpenPose loaded!')
#
# # Load COCO validation annotations
# with open("data/COCO/person_keypoints_val2017.json", "r") as f:
#     coco_data = json.load(f)
#
# # Initialize a dictionary to store the detected poses
# detected_poses = {}
#
# # Initialize zero array for BODY_25 model (25 keypoints, each with x, y coordinates and confidence score)
# zero_array = np.zeros((25, 3), dtype=np.float32).tolist()  # Updated to include confidence score as float
#
# # Process each image in the COCO validation set
# for image_info in coco_data['images']:
#     image_id = image_info['id']
#     image_path = os.path.join("C:/Users/botis/Downloads/RMSE_OP/val2017/val2017", image_info['file_name'])  # Replace with the path to your COCO val2017 folder
#     # Read the image
#     image = cv2.imread(image_path)
#
#     # Skip if the image couldn't be read
#     if image is None:
#         print(f"Warning: Could not read image {image_path}")
#         continue
#
#     # Create OpenPose datum and populate
#     datum = op.Datum()
#     datum.cvInputData = image
#
#     # Process the image to get keypoints
#     opWrapper.emplaceAndPop(op.VectorDatum([datum]))
#
#     # Initialize keypoints with zeros and confidence scores
#     detected_keypoints = zero_array.copy()
#
#     # If keypoints are detected, update the zero array with confidence scores
#     if datum.poseKeypoints is not None:
#         detected_keypoints_multiple = []
#         for person_idx in range(datum.poseKeypoints.shape[0]):
#             detected_keypoints_single = zero_array.copy()
#             for i in range(datum.poseKeypoints.shape[1]):
#                 x, y, confidence = datum.poseKeypoints[person_idx, i, :]
#                 # Explicitly convert to Python native types
#                 detected_keypoints_single[i] = [float(x), float(y), float(confidence)]
#             detected_keypoints_multiple.append(detected_keypoints_single)
#
#     # Store in dictionary
#     detected_poses[image_id] = {
#         'keypoints': detected_keypoints_multiple
#     }
#
# # Save the detected poses to a new JSON file
# with open("data/COCO/detected_poses_newer.json", "w") as f:
#     json.dump(detected_poses, f)
#
# print("Pose detection complete. Results saved to 'detected_poses_newer.json'.")


############################################################
# Define the mapping between COCO keypoints and OpenPose keypoints
# Note: OpenPose keypoints are 0-indexed, while COCO keypoints are in the form of [x, y, visibility]
#       so we take every 3rd element starting from 0 for 'x' and 1 for 'y' to map to OpenPose.
# COCO keypoints: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L304
# OpenPose keypoints: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

# Load the JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Define the mapping between COCO keypoints and OpenPose keypoints
coco_to_openpose_mapping = {
    0: 0,  # Nose
    1: 16,  # Left Eye
    2: 15,  # Right Eye
    3: 18,  # Left Ear
    4: 17,  # Right Ear
    5: 5,  # Left Shoulder
    6: 2,  # Right Shoulder
    7: 6,  # Left Elbow
    8: 3,  # Right Elbow
    9: 7,  # Left Wrist
    10: 4,  # Right Wrist
    11: 12,  # Left Hip
    12: 9,  # Right Hip
    13: 13,  # Left Knee
    14: 10,  # Right Knee
    15: 14,  # Left Ankle
    16: 11  # Right Ankle
}

# Create labels for the keypoints
keypoint_labels = {
    0: 'Nose',
    1: 'Left Eye',
    2: 'Right Eye',
    3: 'Left Ear',
    4: 'Right Ear',
    5: 'Left Shoulder',
    6: 'Right Shoulder',
    7: 'Left Elbow',
    8: 'Right Elbow',
    9: 'Left Wrist',
    10: 'Right Wrist',
    11: 'Left Hip',
    12: 'Right Hip',
    13: 'Left Knee',
    14: 'Right Knee',
    15: 'Left Ankle',
    16: 'Right Ankle'
}

def plot_keypoints_grid(image_path, matched_coco, matched_openpose, all_openpose, labels, mapping):
    # Load the image
    img = plt.imread(image_path)

    # Create a legend for markers
    legend_elements = [
        mlines.Line2D([], [], color='red', marker='o', linestyle='', label='COCO Matched', markersize=8),
        mlines.Line2D([], [], color='blue', marker='x', linestyle='', label='OpenPose Matched', markersize=8),
        mlines.Line2D([], [], color='green', marker='s', linestyle='', label='OpenPose All', markersize=8)
    ]

    # Iterate through keypoint types and plot each type individually
    for idx, keypoint_type in enumerate(labels.values()):
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f'Hungarian matching: {keypoint_type}')

        # Get the keypoints for the current type
        coco_points = matched_coco.get(keypoint_type, [])
        openpose_points = matched_openpose.get(keypoint_type, [])
        all_openpose_points = all_openpose.get(keypoint_type, [])  # All OpenPose keypoints

        # # Plot all OpenPose keypoints for the current type
        # for x, y in all_openpose_points:
        #     plt.scatter(x, y, c='green', marker='s', s=20)

        # Plot matched COCO keypoints for the current type
        for x, y in coco_points:
            plt.scatter(x, y, c='red', marker='o', s=10)

        # Plot matched OpenPose keypoints for the current type
        for x, y in openpose_points:
            plt.scatter(x, y, c='blue', marker='x', s=10)

        # Display the simplified legend with marker descriptions
        plt.legend(handles=legend_elements, loc='upper right', fontsize='small')

        # Show the current plot
        plt.show()

# Define function to extract keypoints for a given image from COCO data
def extract_coco_keypoints(coco_data, image_id):
    keypoints = None
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image_id:
            keypoints = annotation['keypoints']
            break
    return keypoints if keypoints else [0] * 51

# Define function to extract keypoints for a given image from OpenPose data
def extract_openpose_keypoints(openpose_data, image_id):
    return openpose_data.get(str(image_id), {}).get('keypoints', [[0, 0]] * 25)


# Function to check if a keypoint lies within a bounding box
def keypoints_in_bbox(keypoints, bbox):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    keypoints_in_box = []
    for kpt in keypoints:
        x, y = kpt[:2]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            keypoints_in_box.append(kpt)

    return np.array(keypoints_in_box)

# Initialize variables to store squared errors for each common keypoint

rmse_by_keypoint = {key: [] for key in coco_to_openpose_mapping.keys()}
wrmse_by_keypoint = {key: [] for key in coco_to_openpose_mapping.keys()}
confidence_by_keypoint = {key: [] for key in coco_to_openpose_mapping.keys()}

squared_errors = {key: [] for key in coco_to_openpose_mapping.keys()}
weighted_squared_errors = {key: [] for key in coco_to_openpose_mapping.keys()}
total_confidences = {key: 0 for key in coco_to_openpose_mapping.keys()}
count_per_keypoint = {key: 0 for key in coco_to_openpose_mapping.keys()}

total_squared_error = 0
total_weighted_squared_error = 0
total_weight = 0
total_count = 0

rmse_values = []
confidence_values = []

coco_points_2299 = []
openpose_points_2299 = []
coco_points_2299_hungarian = []
openpose_points_2299_hungarian = []

matched_coco_keypoints = defaultdict(list)  # Store matched COCO keypoints by type
matched_openpose_keypoints = defaultdict(list)  # Store matched OpenPose keypoints by type
unmatched_openpose_keypoints = defaultdict(list)  # Store unmatched OpenPose keypoints by type
all_openpose = defaultdict(list)

# Define a threshold for the maximum allowable distance between matching keypoints
max_distance_threshold = 50 # Adjust this value based on your observations

# Load the COCO and OpenPose data
coco_data = load_json('data/COCO/modified_person_keypoints_val2017.json')
openpose_data = load_json('data/COCO/detected_poses_newer.json')

common_image_ids = set(coco_data.keys()).intersection(set(openpose_data.keys()))

# Loop through all image IDs available in both datasets
for image_id in common_image_ids:
    coco_keypoints_all = np.array(coco_data[str(image_id)]['keypoints'])
    openpose_keypoints_all = np.array(openpose_data[str(image_id)]['keypoints'])

    # For single-person images, directly compare keypoints
    if len(coco_keypoints_all) == 1 or len(openpose_keypoints_all) == 1:
        coco_keypoints = coco_keypoints_all[0]
        openpose_keypoints = openpose_keypoints_all[0]

        for coco_idx, openpose_idx in coco_to_openpose_mapping.items():
            coco_x, coco_y, coco_visibility = coco_keypoints[coco_idx]
            openpose_x, openpose_y, openpose_confidence = openpose_keypoints[openpose_idx]  # Include confidence

            if coco_x == 0 and coco_y == 0:
                continue

            if coco_visibility > 0:
                squared_error = (coco_x - openpose_x) ** 2 + (coco_y - openpose_y) ** 2
                squared_errors[coco_idx].append(squared_error)
                # Populate the dictionaries with the matched keypoints

                # Weighted squared error
                weighted_squared_error = squared_error * openpose_confidence
                weighted_squared_errors[coco_idx].append(weighted_squared_error)
                total_confidences[coco_idx] += openpose_confidence

                # Calculate wRMSE (New line)
                wrmse = sqrt(weighted_squared_error)

                # Add wRMSE and confidence values to dictionaries (New lines)
                wrmse_by_keypoint[coco_idx].append(wrmse)

                # Update counts and errors (only once)
                total_squared_error += squared_error
                total_weighted_squared_error += weighted_squared_error
                total_weight += openpose_confidence
                total_count += 1
                count_per_keypoint[coco_idx] += 1

                # Collect data for graphs
                rmse_values.append(sqrt(squared_error))
                confidence_values.append(openpose_confidence)

                # Calculate RMSE and confidence
                rmse = sqrt(squared_error)
                confidence = openpose_confidence

                # Add RMSE and confidence values to dictionaries
                rmse_by_keypoint[coco_idx].append(rmse)
                confidence_by_keypoint[coco_idx].append(confidence)

    # For multiple-person images, apply the Hungarian algorithm
    else:
        num_coco_people = len(coco_keypoints_all)
        num_openpose_people = len(openpose_keypoints_all)

        # Create a cost matrix
        cost_matrix = np.zeros((num_coco_people, num_openpose_people))

        for i, coco_keypoints in enumerate(coco_keypoints_all):
            for j, openpose_keypoints in enumerate(openpose_keypoints_all):
                total_cost = 0
                for coco_idx, openpose_idx in coco_to_openpose_mapping.items():
                    coco_x, coco_y, coco_visibility = coco_keypoints[coco_idx]
                    openpose_x, openpose_y, openpose_confidence = openpose_keypoints[openpose_idx]

                    if coco_visibility > 0:
                        cost = euclidean((coco_x, coco_y), (openpose_x, openpose_y))
                        total_cost += cost

                cost_matrix[i, j] = total_cost

        # Perform Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Loop over the optimal assignments and update errors
        for i, j in zip(row_ind, col_ind):
            coco_keypoints = coco_keypoints_all[i]
            openpose_keypoints = openpose_keypoints_all[j]

            for coco_idx, openpose_idx in coco_to_openpose_mapping.items():
                coco_x, coco_y, coco_visibility = coco_keypoints[coco_idx]
                openpose_x, openpose_y, openpose_confidence = openpose_keypoints[openpose_idx]

                if coco_visibility > 0:
                    squared_error = (coco_x - openpose_x) ** 2 + (coco_y - openpose_y) ** 2
                    squared_errors[coco_idx].append(squared_error)

                    # Weighted squared error
                    weighted_squared_error = squared_error * openpose_confidence
                    weighted_squared_errors[coco_idx].append(weighted_squared_error)
                    total_confidences[coco_idx] += openpose_confidence

                    # Calculate RMSE, wRMSE, and confidence
                    rmse = sqrt(squared_error)
                    wrmse = sqrt(weighted_squared_error)  # wRMSE
                    confidence = openpose_confidence

                    # Add RMSE, wRMSE, and confidence values to dictionaries
                    rmse_by_keypoint[coco_idx].append(rmse)
                    wrmse_by_keypoint[coco_idx].append(wrmse)  # Added this line
                    confidence_by_keypoint[coco_idx].append(confidence)

                    # Update counts and errors
                    total_squared_error += squared_error
                    total_weighted_squared_error += weighted_squared_error
                    total_weight += openpose_confidence
                    total_count += 1
                    count_per_keypoint[coco_idx] += 1


    # Only populate for image 2299
    if image_id == '2299':
        # Reverse the mapping for easier lookup
        openpose_to_coco_mapping = {v: k for k, v in coco_to_openpose_mapping.items()}

        for person_openpose_kps in openpose_keypoints_all:
            for openpose_idx, (openpose_x, openpose_y, openpose_confidence) in enumerate(person_openpose_kps):
                if openpose_confidence > 0:
                    coco_idx = openpose_to_coco_mapping.get(openpose_idx)
                    if coco_idx is not None:
                        label = keypoint_labels[coco_idx]
                        all_openpose[label].append((openpose_x, openpose_y))

        for person_coco_kps, person_openpose_kps in zip(coco_keypoints_all, openpose_keypoints_all):
            for coco_idx, openpose_idx in coco_to_openpose_mapping.items():
                coco_x, coco_y, coco_visibility = person_coco_kps[coco_idx]
                openpose_x, openpose_y, openpose_confidence = person_openpose_kps[openpose_idx]

                if coco_visibility > 0 and openpose_confidence > 0:
                    matched_coco_keypoints[keypoint_labels[coco_idx]].append((coco_x, coco_y))
                    matched_openpose_keypoints[keypoint_labels[coco_idx]].append((openpose_x, openpose_y))

    # # Loop through each keypoint type
    # for label in keypoint_labels.values():
    #     if label in matched_coco_keypoints:
    #         coco_x, coco_y = zip(*matched_coco_keypoints[label])
    #         plot_keypoint('C:/Users/botis/Downloads/RMSE_OP/val2017/val2017/000000002299.jpg', coco_x, coco_y,
    #                       title=f"Matched COCO {label} Keypoints")
    #
    #     if label in matched_openpose_keypoints:
    #         openpose_x, openpose_y = zip(*matched_openpose_keypoints[label])
    #         plot_keypoint('C:/Users/botis/Downloads/RMSE_OP/val2017/val2017/000000002299.jpg', openpose_x, openpose_y,
    #                       title=f"Matched OpenPose {label} Keypoints")
    #
    #     if label in unmatched_openpose_keypoints:
    #         unmatched_x, unmatched_y = zip(*unmatched_openpose_keypoints[label])
    #         plot_keypoint('C:/Users/botis/Downloads/RMSE_OP/val2017/val2017/000000002299.jpg', unmatched_x, unmatched_y,
    #                       title=f"Unmatched OpenPose {label} Keypoints")



# plot_keypoints_grid('C:/Users/botis/Downloads/RMSE_OP/val2017/val2017/000000002299.jpg', matched_coco_keypoints, matched_openpose_keypoints, all_openpose, keypoint_labels, coco_to_openpose_mapping)


# # Plotting points for image 2299 with Hungarian algorithm
# plt.figure(figsize=(12, 12))
#
# # Plot COCO keypoints
# coco_x = [x for x, y in coco_points_2299_hungarian]
# coco_y = [y for x, y in coco_points_2299_hungarian]
# plt.scatter(coco_x, coco_y, c='red', label='COCO')
#
# # Plot OpenPose keypoints
# openpose_x = [x for x, y in openpose_points_2299_hungarian]
# openpose_y = [y for x, y in openpose_points_2299_hungarian]
# plt.scatter(openpose_x, openpose_y, c='blue', label='OpenPose')
#
# # Add legend and show plot
# plt.legend()
# plt.title('Keypoint Matching for Image 2299 (Hungarian Algorithm)')
# plt.xlabel('x-coordinate')
# plt.ylabel('y-coordinate')
# plt.show()

# Calculate RMSE for each keypoint
rmse_per_keypoint = {key: sqrt(np.mean(errors)) if errors else 0 for key, errors in squared_errors.items()}
wRMSE_per_keypoint = {key: sqrt(sum(errors) / total_confidences[key]) if errors else 0 for key, errors in weighted_squared_errors.items()}

print("RMSE per keypoint:")
print(rmse_per_keypoint)
print("Weighted RMSE per keypoint:")
print(wRMSE_per_keypoint)
print("Count per keypoint:")
print(count_per_keypoint)

# Add these lines here
overall_RMSE = sqrt(total_squared_error / total_count)
overall_wRMSE = sqrt(total_weighted_squared_error / total_weight)

print("Overall RMSE:")
print(overall_RMSE)
print("Overall Weighted RMSE:")
print(overall_wRMSE)

# Calculate the actual total number of unique images in the COCO dataset based on annotations
total_unique_coco_images = len(coco_data.keys())

# Calculate the total number of images in OpenPose dataset
total_openpose_images = len(openpose_data.keys())

# Number of valid images used in the calculation (common between both datasets)
num_valid_images = len(common_image_ids)

# Number of ignored images in OpenPose dataset
ignored_openpose_images = total_openpose_images - num_valid_images

# Number of ignored images in the COCO dataset
ignored_coco_images = total_unique_coco_images - num_valid_images

# Calculate the total number of unique 'image_id's in the COCO dataset
# This is the same as the total number of unique images based on the new structure
total_unique_coco_image_ids = total_unique_coco_images

# Find the number of these that actually have 0 keypoints annotated
# For the new structure, each 'image_id' has at least one person annotated, so this is 0
num_images_with_zero_keypoints = 0

total_unique_coco_images, total_openpose_images, num_valid_images, ignored_coco_images, ignored_openpose_images, total_unique_coco_image_ids, num_images_with_zero_keypoints
# Detailed output
print("\n=== Number of Images in COCO Dataset ===")
print(f"Total number of unique 'image_id's in the COCO dataset: {total_unique_coco_image_ids}")
print(f"Number of these 'image_id's that actually have 0 keypoints annotated: {num_images_with_zero_keypoints}")

print("\n=== Number of Images ===")
print(f"Total number of unique images in the COCO dataset based on annotations: {total_unique_coco_images}")
print(f"Total number of images in the OpenPose dataset: {total_openpose_images}")
print(f"Number of valid images used in the RMSE calculation (common between both datasets): {num_valid_images}")
print(f"Number of images ignored from the COCO dataset: {ignored_coco_images}")
print(f"Number of images ignored from the OpenPose dataset: {ignored_openpose_images}")

print("\n=== Count of Keypoints ===")
print("The count of keypoints represents the number of times each keypoint appears in the common set of images.")
print("It is used for each RMSE calculation.")
total_keypoints_count = 0  # Initialize the total keypoints count
for keypoint, count in count_per_keypoint.items():
    print(f"For the keypoint '{keypoint}', it was used {count} times in the RMSE calculation.")
    total_keypoints_count += count  # Add the count to the total keypoints count

print(f"\nTotal number of keypoints used in the RMSE calculation: {total_keypoints_count}")

# Initialize a counter for the number of people
num_people_in_coco = 0

# Loop through all entries in the COCO JSON data
for entry in coco_data.values():
    # Check if the entry contains keypoints arrays
    if 'keypoints' in entry:
        # Increment the counter for each keypoints array
        num_people_in_coco += len(entry['keypoints'])

# Display the number of people
print(f"Number of people in the COCO dataset: {num_people_in_coco}")


# RMSE
# For the Boxplot
plt.figure(figsize=(15, 7))
plt.boxplot([rmse_by_keypoint[k] for k in sorted(rmse_by_keypoint.keys())])
plt.title('Boxplot of RMSE for Each Keypoint')
plt.xlabel('Keypoint')
plt.ylabel('RMSE')
plt.xticks(ticks=np.arange(1, len(coco_to_openpose_mapping) + 1), labels=sorted(coco_to_openpose_mapping.keys()), rotation=45)
plt.tight_layout()
plt.show()

# For the Scatter Plot
plt.figure(figsize=(15, 7))

# Create a colormap
colors = plt.cm.jet(np.linspace(0, 1, len(coco_to_openpose_mapping)))

for idx, key in enumerate(sorted(confidence_by_keypoint.keys())):
    num_points = min(len(confidence_by_keypoint[key]), len(rmse_by_keypoint[key]))

    # Use the minimum number of points to ensure that the dimensions match
    plt.scatter(confidence_by_keypoint[key][:num_points],
                rmse_by_keypoint[key][:num_points],
                color=colors[idx],
                alpha=0.6,
                label=f"Keypoint {key}")

plt.title('Scatter Plot of RMSE vs Confidence')
plt.xlabel('Confidence')
plt.ylabel('RMSE')
plt.colorbar(label='Keypoint Index')
plt.legend()
plt.tight_layout()
plt.show()



# Improved Boxplot
plt.figure(figsize=(15, 7))
bp = plt.boxplot([wrmse_by_keypoint[k] for k in sorted(wrmse_by_keypoint.keys())],
                 patch_artist=True,
                 medianprops={'color': 'black'})

# Add colors to the boxes
colors = plt.cm.jet(np.linspace(0, 1, len(wrmse_by_keypoint)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Boxplot of wRMSE for Each Keypoint')
plt.xlabel('Keypoint')
plt.ylabel('wRMSE')
plt.xticks(ticks=np.arange(1, len(wrmse_by_keypoint) + 1),
           labels=[keypoint_labels[k] for k in sorted(wrmse_by_keypoint.keys())],
           rotation=45)

plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Sample code to improve the aesthetics of the Scatter Plot
plt.figure(figsize=(15, 7))

# Create a colormap
colors = plt.cm.jet(np.linspace(0, 1, len(keypoint_labels)))

# Add grid lines and background color
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_facecolor('white')

# Loop through each keypoint to plot
for idx, key in enumerate(sorted(confidence_by_keypoint.keys())):
    num_points = min(len(confidence_by_keypoint[key]), len(wrmse_by_keypoint[key]))

    plt.scatter(confidence_by_keypoint[key][:num_points],
                wrmse_by_keypoint[key][:num_points],
                color=colors[idx],
                alpha=0.6,
                label=f"{keypoint_labels[key]}")

# Add titles and labels
plt.title('Scatter Plot of wRMSE vs Confidence', fontsize=16)
plt.xlabel('Confidence', fontsize=14)
plt.ylabel('wRMSE', fontsize=14)

# Add the legend, making it the same height as the plot
legend = plt.legend(title='Keypoint', bbox_to_anchor=(1.05, 0), loc='center left')
frame = plt.gca()
legend.set_bbox_to_anchor((1.05, 0, 0.2, 1.), frame.transAxes)
# Make layout fit better
plt.tight_layout()

# Show the plot
plt.show()






# latest
# # For multiple-person images, apply minimum distance matching
    # else:
    #     for coco_keypoints in coco_keypoints_all:
    #         for coco_idx, openpose_idx in coco_to_openpose_mapping.items():
    #             coco_x, coco_y, coco_visibility = coco_keypoints[coco_idx]
    #
    #             if coco_x == 0 and coco_y == 0:
    #                 continue
    #
    #             min_distance = float('inf')
    #             closest_openpose_keypoints = None
    #
    #             for openpose_keypoints in openpose_keypoints_all:
    #                 openpose_x, openpose_y, openpose_confidence = openpose_keypoints[openpose_idx]
    #                 distance = euclidean((coco_x, coco_y), (openpose_x, openpose_y))
    #
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     closest_openpose_keypoints = openpose_keypoints
    #
    #             if closest_openpose_keypoints is not None:
    #                 openpose_x, openpose_y, openpose_confidence = closest_openpose_keypoints[openpose_idx]
    #
    #                 if coco_visibility > 0:
    #                     squared_error = (coco_x - openpose_x) ** 2 + (coco_y - openpose_y) ** 2
    #                     squared_errors[coco_idx].append(squared_error)
    #
    #                     # Weighted squared error
    #                     weighted_squared_error = squared_error * openpose_confidence
    #                     weighted_squared_errors[coco_idx].append(weighted_squared_error)
    #                     total_confidences[coco_idx] += openpose_confidence
    #
    #                     # Calculate wRMSE (New line)
    #                     wrmse = sqrt(weighted_squared_error)
    #
    #                     # Add wRMSE and confidence values to dictionaries (New lines)
    #                     wrmse_by_keypoint[coco_idx].append(wrmse)
    #
    #                     # Update counts and errors (only once)
    #                     total_squared_error += squared_error
    #                     total_weighted_squared_error += weighted_squared_error
    #                     total_weight += openpose_confidence
    #                     total_count += 1
    #                     count_per_keypoint[coco_idx] += 1
    #
    #                     # Collect data for graphs
    #                     rmse_values.append(sqrt(squared_error))
    #                     confidence_values.append(openpose_confidence)
    #
    #                     # Calculate RMSE and confidence
    #                     rmse = sqrt(squared_error)
    #                     confidence = openpose_confidence
    #
    #                     # Add RMSE and confidence values to dictionaries
    #                     rmse_by_keypoint[coco_idx].append(rmse)
    #                     confidence_by_keypoint[coco_idx].append(confidence)



#    # For multiple-person images, apply minimum distance matching
#     else:
#         for coco_keypoints in coco_keypoints_all:
#             for coco_idx, openpose_idx in coco_to_openpose_mapping.items():
#                 coco_x, coco_y, coco_visibility = coco_keypoints[coco_idx]
#
#                 if coco_x == 0 and coco_y == 0:
#                     continue
#
#                 min_distance = float('inf')
#                 closest_openpose_keypoints = None
#
#                 for openpose_keypoints in openpose_keypoints_all:
#                     openpose_x, openpose_y, openpose_confidence = openpose_keypoints[openpose_idx]  # Include confidence
#                     distance = euclidean((coco_x, coco_y), (openpose_x, openpose_y))
#
#                     if distance < min_distance:
#                         min_distance = distance
#                         closest_openpose_keypoints = openpose_keypoints
#
#                 if closest_openpose_keypoints is not None:
#                     openpose_x, openpose_y, openpose_confidence = closest_openpose_keypoints[openpose_idx]  # Include confidence
#
#                     if coco_visibility > 0:
#                         squared_error = (coco_x - openpose_x) ** 2 + (coco_y - openpose_y) ** 2
#                         squared_errors[coco_idx].append(squared_error)
#
#                         # Weighted squared error
#                         weighted_squared_error = squared_error * openpose_confidence
#                         weighted_squared_errors[coco_idx].append(weighted_squared_error)
#                         total_confidences[coco_idx] += openpose_confidence
#
#                         # Calculate wRMSE (New line)
#                         wrmse = sqrt(weighted_squared_error)
#
#                         # Add wRMSE and confidence values to dictionaries (New lines)
#                         wrmse_by_keypoint[coco_idx].append(wrmse)
#
#                         # Update counts and errors (only once)
#                         total_squared_error += squared_error
#                         total_weighted_squared_error += weighted_squared_error
#                         total_weight += openpose_confidence
#                         total_count += 1
#                         count_per_keypoint[coco_idx] += 1
#
#                         # Collect data for graphs
#                         rmse_values.append(sqrt(squared_error))
#                         confidence_values.append(openpose_confidence)
#
#                         # Calculate RMSE and confidence
#                         rmse = sqrt(squared_error)
#                         confidence = openpose_confidence
#
#                         # Add RMSE and confidence values to dictionaries
#                         rmse_by_keypoint[coco_idx].append(rmse)
#                         confidence_by_keypoint[coco_idx].append(confidence)
# #
# Collect matched points for image 2299
#                         if image_id == '2299':
#                             coco_points_2299.append((coco_x, coco_y))
#                             openpose_points_2299.append((openpose_x, openpose_y))
#
# # Plotting points for image 2299
# plt.figure(figsize=(12, 12))
#
# # Plot COCO keypoints
# coco_x = [x for x, y in coco_points_2299]
# coco_y = [y for x, y in coco_points_2299]
# plt.scatter(coco_x, coco_y, c='red', label='COCO')
#
# # Plot OpenPose keypoints
# openpose_x = [x for x, y in openpose_points_2299]
# openpose_y = [y for x, y in openpose_points_2299]
# plt.scatter(openpose_x, openpose_y, c='blue', label='OpenPose')
#
# # Add legend and show plot
# plt.legend()
# plt.title('Keypoint Matching for Image 2299 (Minimum Distance)')
# plt.xlabel('x-coordinate')
# plt.ylabel('y-coordinate')
# plt.show()