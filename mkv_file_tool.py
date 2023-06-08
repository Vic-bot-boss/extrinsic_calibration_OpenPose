import os
import imageio.v2 as imageio
import av

# Specify the directory containing the PNG files
directory = "data/520_1"

# Specify the output video file name
output_file = "data/videos/520.mkv"

# Specify the FPS and resolution of the images
fps = 10
width = 1280
height = 720

# Get a list of all the PNG files in the directory
png_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

# Sort the PNG files in alphanumeric order
png_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Create an output video writer using the Lagarith codec
output = av.open(output_file, 'w')
video_stream = output.add_stream('ffv1', rate=fps)
video_stream.width = width
video_stream.height = height

# Iterate over the PNG files and write each frame to the output video file
for png_file in png_files:
    # Load the PNG image using imageio
    image = imageio.imread(png_file)

    # Convert the image to RGB if necessary
    if image.ndim == 2:
        image = image[..., None].repeat(3, axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]

    # Resize the image if necessary
    if image.shape[0] != height or image.shape[1] != width:
        image = imageio.imresize(image, (height, width))

    # Create a video frame from the image and add it to the output video stream
    frame = av.VideoFrame.from_ndarray(image, format='rgb24')
    packet = video_stream.encode(frame)
    output.mux(packet)


# Close the output video file
output.close()
print('Video created')