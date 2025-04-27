import sys
import io
from io import BytesIO
import skimage as skimage
from skimage import transform, color, exposure, io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crop_and_resize(screenshot, canvas_details, actual_width, dino_width, xPos):

    image = skimage.io.imread(BytesIO(screenshot))
    # image = image[400:700, 300:900]

    # Get the width and height of the image
    # print(canvas_details)

    # image = image[canvas_details["top"]:canvas_details["bottom"], canvas_details["left"]:canvas_details["right"]]
    image = image [80:400 , 250:1200]
    # Crop the image so that only the screen before the dino is visible
    # image = image[
    #     int(canvas_details["y"]) : int(canvas_details["height"] + canvas_details["y"]),
    #     int(
    #         canvas_details["x"]
    #         + (int(dino_width) * (int(canvas_details["width"]) / int(actual_width)))
    #         + (int(xPos) * (int(canvas_details["width"]) / int(actual_width)))
    #     ) : int((canvas_details["width"] / 2) + canvas_details["x"]),
    # ]

    # ========== 🔥 DEBUG DRAWING ENDS HERE 🔥 ==========

    # Calculate the scaling between screen coordinates and canvas pixels

    # See image
    # plt.figure()
    # plt.imshow(image)
    # plt.title("Cropped Dino and Obstacle View")
    # plt.axis('off')
    # plt.show()
    # Convert the image
    image = skimage.color.rgb2gray(image)

    # transform and resize the shape
    image = skimage.transform.resize(image, (75, 75))

    # rescale pixel values to be between 0 to 255
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))

    # Normalize it
    image = image / 255.0
    return image
