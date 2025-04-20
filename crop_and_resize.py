def crop_and_resize(image):
    # Crop the image
    image = image.crop((10, 25, 1200, 600))
    # Change the image into grayscale
    image = image.convert("L")
    # Resize the image for DQN
    image = image.resize((84,84))
    return image
    