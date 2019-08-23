def get_hw(image):
    if len(image.shape) is 4:
        return image.shape[1], image.shape[2]
    else:
        return image.shape[0], image.shape[1]
