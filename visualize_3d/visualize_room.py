import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path


def toggle_plots(event: MouseEvent):
    """

    Args:
        event:

    Returns:

    """
    # Check if the event is a mouse button press
    if event.dblclick:
        # Clear the current axis
        plt.gca().clear()
        # Toggle the mode and draw
        current_mode = MODES.pop(0)
        MODES.insert(1, current_mode)
        draw_routines(ax=AX, current_mode=current_mode)
        # Update the canvas to show the toggled plot
        plt.gcf().canvas.draw()


def set_axes(ax, title):
    """

    Args:
        ax:
        title:

    Returns:

    """
    # Add labels and legend
    ax.set_xlabel('X (cm)', fontsize=8)
    ax.set_ylabel('Y (cm)', fontsize=8)
    ax.set_zlabel('Z (cm)', fontsize=8)
    ax.set_title(title, fontsize=10)

    # Normalizing and limits the axes
    ax.set_box_aspect([1.0, 1.0, 1.0])
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.6 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Add ticks and legends
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10)



def draw_cuboid(ax, dims, rotation, translation, plot3d_kargs):
    """

    Args:
        ax:
        dims:
        rotation:
        translation:
        plot3d_kargs:

    Returns:

    """
    x_min, y_min, z_min, x_max, y_max, z_max = dims
    cuboid_vertices = np.array(
        [[x_min, y_max, z_min],
         [x_max, y_max, z_min],
         [x_max, y_min, z_min],
         [x_min, y_min, z_min],
         [x_min, y_max, z_max],
         [x_max, y_max, z_max],
         [x_max, y_min, z_max],
         [x_min, y_min, z_max]]
    )
    cuboid_vertices = np.dot(cuboid_vertices, rotation) + translation

    cuboid_faces= np.array(
        [[cuboid_vertices[0], cuboid_vertices[1], cuboid_vertices[2], cuboid_vertices[3]],
         [cuboid_vertices[4], cuboid_vertices[5], cuboid_vertices[6], cuboid_vertices[7]],
         [cuboid_vertices[0], cuboid_vertices[1], cuboid_vertices[5], cuboid_vertices[4]],
         [cuboid_vertices[2], cuboid_vertices[3], cuboid_vertices[7], cuboid_vertices[6]],
         [cuboid_vertices[1], cuboid_vertices[2], cuboid_vertices[6], cuboid_vertices[5]],
         [cuboid_vertices[4], cuboid_vertices[7], cuboid_vertices[3], cuboid_vertices[0]]]
    )
    ax.add_collection3d(
        Poly3DCollection(
            cuboid_faces,
            **plot3d_kargs,
        ),
    )

def draw_image_plane(ax, rotation, translation, image_plane_scaled_size, plot3d_kargs):
    """

    Args:
        ax:
        rotation:
        translation:
        image_plane_scaled_size:
        plot3d_kargs:

    Returns:

    """
    # Define camera image planes
    image_plane_flat_corners = np.array(
        [[-image_plane_scaled_size[0], -image_plane_scaled_size[1], 0],
         [ image_plane_scaled_size[0], -image_plane_scaled_size[1], 0],
         [ image_plane_scaled_size[0],  image_plane_scaled_size[1], 0],
         [-image_plane_scaled_size[0],  image_plane_scaled_size[1], 0]]
    )
    image_plane_vertices = np.dot(image_plane_flat_corners, rotation) + translation
    image_plane_face = np.array(
        [[image_plane_vertices[0], image_plane_vertices[1], image_plane_vertices[2], image_plane_vertices[3]]]
    )
    ax.add_collection3d(
        Poly3DCollection(
            image_plane_face,
            **plot3d_kargs,
        ),
    )

def draw_fov(ax, rotation, translation, camera_fov, depth, plot3d_kargs):
    """

    Args:
        ax:
        rotation:
        translation:
        camera_fov:
        depth:
        plot3d_kargs:

    Returns:

    """
    # Define FoV boundary planes
    fov_corners = np.array(
        [[0, 0, 0],
         [depth*np.tan(camera_fov[0]/2), -depth*np.tan(camera_fov[1]/2), depth],
         [-depth*np.tan(camera_fov[0]/2), -depth*np.tan(camera_fov[1]/2), depth],
         [0, 0, 0],
         [0, 0, 0],
         [depth * np.tan(camera_fov[0] / 2), depth * np.tan(camera_fov[1] / 2), depth],
         [-depth * np.tan(camera_fov[0] / 2), depth * np.tan(camera_fov[1] / 2), depth],
         [0, 0, 0],
         [0, 0, 0],
         [depth * np.tan(camera_fov[0] / 2), -depth * np.tan(camera_fov[1] / 2), depth],
         [depth * np.tan(camera_fov[0] / 2), depth * np.tan(camera_fov[1] / 2), depth],
         [0, 0, 0],
         [0, 0, 0],
         [-depth*np.tan(camera_fov[0]/2), -depth*np.tan(camera_fov[1]/2), depth],
         [-depth*np.tan(camera_fov[0]/2), depth*np.tan(camera_fov[1]/2), depth],
         [0, 0, 0],
         [-depth * np.tan(camera_fov[0] / 2), -depth * np.tan(camera_fov[1] / 2), depth],
         [-depth * np.tan(camera_fov[0] / 2), depth * np.tan(camera_fov[1] / 2), depth],
         [depth * np.tan(camera_fov[0] / 2), depth * np.tan(camera_fov[1] / 2), depth],
         [depth * np.tan(camera_fov[0] / 2), -depth * np.tan(camera_fov[1] / 2), depth]]
    )
    fov_boundary_vertices = np.dot(fov_corners, rotation) + translation
    fov_faces = np.array(
        [[fov_boundary_vertices[0], fov_boundary_vertices[1], fov_boundary_vertices[2], fov_boundary_vertices[3]],
         [fov_boundary_vertices[4], fov_boundary_vertices[5], fov_boundary_vertices[6], fov_boundary_vertices[7]],
         [fov_boundary_vertices[8], fov_boundary_vertices[9], fov_boundary_vertices[10], fov_boundary_vertices[11]],
         [fov_boundary_vertices[12], fov_boundary_vertices[13], fov_boundary_vertices[14], fov_boundary_vertices[15]],
         [fov_boundary_vertices[16], fov_boundary_vertices[17], fov_boundary_vertices[18], fov_boundary_vertices[19]]]
    )
    ax.add_collection3d(
        Poly3DCollection(
            fov_faces,
            **plot3d_kargs,
        ),
    )

def add_item2legend(ax, translation, label, color):
    """

    Args:
        ax:
        translation:
        label:
        color:

    Returns:

    """
    # Plot camera positions and attach a legend to it
    ax.plot(
        [translation[0]], [translation[1]], [translation[2]],
        "o",
        color=color,
        markersize=5,
        label=label,
    )

def draw_coordination_system(ax, rotation, translation, arrow_scale):
    """

    Args:
        ax:
        rotation:
        translation:
        arrow_scale:

    Returns:

    """
    # Draw x, y, and z arrows for the sensor's coordination system
    unit_vectors = np.array(
        [[arrow_scale, 0, 0],
         [0, arrow_scale, 0],
         [0, 0, arrow_scale]]
    )
    color = ("red", "green", "blue")
    for i in range(3):
        image_points_3d = np.matmul(rotation.T, unit_vectors[i])
        ax.quiver(
            translation[0],
            translation[1],
            translation[2],
            image_points_3d[0],
            image_points_3d[1],
            image_points_3d[2],
            color=color[i],
            arrow_length_ratio=0.3,
            linewidth=2
        )

def draw_routines(ax, current_mode):
    """

    Args:
        ax:
        current_mode:

    Returns:

    """
    # Set a few graphic parameters
    image_plane_scaled_size = (64, 36)  # a ratio of image size (1280, 720)
    arrow_scale = 32     # preferably smaller than image_plane_scaled_size
    plot3d_common_kargs = {"linestyle": "solid", "linewidth": 0.5}
    colors = ["#67A0E2", "#D474FF", "#407F4A", "#E25C30", "#58A3F9"]

    # Draw room
    plot3d_local_kargs = {"color": "black", "alpha": 0.05}
    room_x_min = 0
    room_y_min = 0
    room_z_min = 0
    room_x_max = 610
    room_y_max = 500
    room_z_max = 300
    room_dims = (room_x_min, room_y_min, room_z_min, room_x_max, room_y_max, room_z_max)
    draw_cuboid(
        ax=ax,
        dims=room_dims,
        rotation=np.eye(3),
        translation=np.array([0, 0, 0]),
        plot3d_kargs={**plot3d_common_kargs, **plot3d_local_kargs},
    )
    draw_coordination_system(
        ax=ax,
        rotation=np.eye(3),
        translation=np.array([0, 0, 0]),
        arrow_scale=2*arrow_scale,
    )
    add_item2legend(
        ax=ax,
        label=f"Origo!",
        translation=np.array([0, 0, 0]),
        color="black",
    )

    # Draw dedicated zone
    # plot3d_local_kargs = {"color": "red", "alpha": 0.25}
    # zone_x_min = -75
    # zone_y_min = -67.5
    # zone_z_min = 0
    # zone_x_max = 75
    # zone_y_max = 67.5
    # zone_z_max = 200
    # zone_dims = (zone_x_min, zone_y_min, zone_z_min, zone_x_max, zone_y_max, zone_z_max)
    # draw_cuboid(
    #     ax=ax,
    #     dims=zone_dims,
    #     rotation=np.eye(3),
    #     translation=np.array([447, 154, 0]),
    #     plot3d_kargs={**plot3d_common_kargs, **plot3d_local_kargs},
    # )
    # add_item2legend(
    #     ax=ax,
    #     label=f"Dedicated Zone",
    #     translation=np.array([447, 154, 100]),
    #     color="red",
    # )

    # Loop for the available Sensor-Parameter files and draw each (image-plane and sensor-coordination-system)
    for sensor_parameters_file in sorted(Path.glob(Path(), "Sensor_Parameters_*.npz")):
        with np.load(str(sensor_parameters_file)) as param:
            # Print core parameters
            print(f"{sensor_parameters_file} \n ------------------")
            print(param['R'])
            print(param['P'])
            print(param['dirVect'])

            # For a few sensors, use preselected colors, else create random color
            if not len(colors):
                color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            else:
                color = colors.pop()

            plot3d_local_kargs = {"color": color, "alpha": 0.2}
            if current_mode == "axes":
                draw_image_plane(
                    ax=ax,
                    rotation=param['R'],
                    translation=param['P'][0],
                    image_plane_scaled_size=image_plane_scaled_size,
                    plot3d_kargs={**plot3d_common_kargs, **plot3d_local_kargs},
                )
                draw_coordination_system(
                    ax=ax,
                    rotation=param['R'],
                    translation=param['P'][0],
                    arrow_scale=arrow_scale,
                )
                add_item2legend(
                    ax=ax,
                    label=f"Sensor {str(sensor_parameters_file).split('_')[3].split('.')[0]}",
                    translation=param['P'][0],
                    color=color,
                )

            if current_mode == "fov":
                camera_fov = (np.pi / 180 * 58, np.pi / 180 * 38)
                depth = 300
                draw_fov(
                    ax=ax,
                    rotation=param['R'],
                    translation=param['P'][0],
                    camera_fov=camera_fov,
                    depth=depth,
                    plot3d_kargs={**plot3d_common_kargs, **plot3d_local_kargs},
                )
                add_item2legend(
                    ax=ax,
                    label=f"Sensor {str(sensor_parameters_file).split('_')[3].split('.')[0]}",
                    translation=param['P'][0],
                    color=color,
                )

    if current_mode == "axes":
        title = "Image-planes"
    else:
        title = "FoVs"

    set_axes(ax, title)
    plt.savefig("fig1.png")


if __name__ == "__main__":
    # Initialize the figure with one 3D-projection subplot
    FIG = plt.figure(figsize=(9, 9))
    AX = FIG.add_subplot(111, projection='3d')

    # Connect the mouse button press event to the toggle_plots function with double clicks
    FIG.canvas.mpl_connect('button_press_event', toggle_plots)
    MODES = ["axes", "fov"]
    toggle_plots.MODES = MODES
    INITIAL_MODE = MODES.pop(0)
    MODES.insert(1, INITIAL_MODE)
    draw_routines(ax=AX, current_mode=INITIAL_MODE)

    # Show the initial plot
    plt.show()
