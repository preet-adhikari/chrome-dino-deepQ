import sys
import io
from io import BytesIO
import skimage as skimage
from skimage import transform, color, exposure, io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crop_and_resize(screenshot, canvas_details = 0, actual_width = 0, dino_width = 0, xPos = 0):

    image = skimage.io.imread(BytesIO(screenshot))

    image = image [80:400 , 250:1200]

    # Calculate the scaling between screen coordinates and canvas pixels

    # See image

    # Convert the image
    image = skimage.color.rgb2gray(image)
    # image = skimage.transform.resize(image, (75, 75))
    # Trying 84x84
    image = skimage.transform.resize(image, (84, 84))

    # plt.figure()
    # plt.imshow(image)
    # plt.title("Cropped Dino and Obstacle View")
    # plt.axis('off')
    # plt.show()
    # transform and resize the shape
    # image = skimage.transform.resize(image, (75, 75))

    # rescale pixel values to be between 0 to 255
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))

    # Normalize it
    image = image / 255.0
    return image
