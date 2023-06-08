from numpy import load

data = load('D:/Ubiqisense/thesis/visualize_3d/Sensor_Parameters_12_517.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

data = load('D:/Ubiqisense/thesis/visualize_3d/Sensor_Parameters_12_518.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

data = load('D:/Ubiqisense/thesis/visualize_3d/Sensor_Parameters_13_536.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

data = load('D:/Ubiqisense/thesis/visualize_3d/Sensor_Parameters_14_520.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])