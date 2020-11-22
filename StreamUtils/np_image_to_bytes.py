import numpy as np
import cv2


def convert(image: np.array, image_format, is_RGB=False):
    if not is_RGB:
        image = image[:, :, ::-1]
    success, encoded_image = cv2.imencode(image_format, image)
    content = encoded_image.tobytes()
    return content
