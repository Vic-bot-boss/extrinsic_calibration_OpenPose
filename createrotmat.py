import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert degrees to radians
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    # Calculate individual rotation matrices
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(roll_rad), -np.sin(roll_rad)],
                           [0, np.sin(roll_rad), np.cos(roll_rad)]])

    rotation_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                           [0, 1, 0],
                           [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    rotation_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                           [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                           [0, 0, 1]])

    # Combine rotation matrices in XYZ order (roll -> pitch -> yaw)
    rotation_matrix = rotation_z.dot(rotation_y).dot(rotation_x)

    return rotation_matrix

roll = 45  # degrees
pitch = 8  # degrees
yaw = 0  # degrees

rotation_matrix = euler_to_rotation_matrix(roll, pitch, yaw)
print(rotation_matrix)