import numpy as np
import zipfile

# Load the NPZ file
archive = np.load('visualize_3d/Sensor_Parameters_12_517.npz')
filedic = dict(np.load(archive))

# Access and modify the values
new_values = np.array([[ 0.99026807,  0.09841024 , 0.09841024],
 [ 0.        ,  0.70710678, -0.70710678],
 [-0.1391731  , 0.70022527 , 0.70022527]])

filedic['R'] = new_values

# Save the modified NPZ file
np.savez('your_modified_file.npz', **filedic)

# Close the NPZ file
filedic.close()
