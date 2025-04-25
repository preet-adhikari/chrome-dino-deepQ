import sys
import io
from io import BytesIO
import skimage as skimage
from skimage import transform, color, exposure, io
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np


def crop_and_resize(screenshot, canvas_details, actual_width, dino_width, xPos):
    image = skimage.io.imread(BytesIO(screenshot))

    # Crop the image so that only the screen before the dino is visible
    image = image[int(canvas_details['y']):int(canvas_details['height']+canvas_details['y']), int(canvas_details['x']+(int(dino_width)*(int(
        canvas_details['width'])/int(actual_width)))+(int(xPos)*(int(canvas_details['width'])/int(actual_width)))):int((canvas_details['width']/2)+canvas_details['x'])]
    # print("Image shape:", image.shape)
    # print("Image dtype:", image.dtype)
    # print("Image : ", image)

    # Assuming `image` is your (150, 247, 3) array
    # plt.figure(figsize=(6, 4))  # Adjust size as needed
    # plt.imshow(image.astype(np.uint8))  # Convert to uint8 if it's float
    # plt.axis('off')  # Hide the axis ticks
    # plt.title("Image with shape (150, 247, 3)")
    # plt.show()
    # sys.exit()
    # Convert the image
    image = skimage.color.rgb2gray(image)

    # transform and resize the shape
    image = skimage.transform.resize(image, (75,75))


    # rescale pixel values to be between 0 to 255
    image = skimage.exposure.rescale_intensity(
        image, out_range=(0, 255))
    
    # Normalize it
    image = image / 255.0
    return image
    