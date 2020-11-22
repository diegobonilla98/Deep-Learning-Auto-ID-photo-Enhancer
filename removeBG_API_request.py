import requests
from StreamUtils import np_image_to_bytes, bytestream_to_np
import numpy as np
import cv2
import matplotlib.pyplot as plt


def remove_background(image: np.array, image_format, is_RGB):
    image_bytes = np_image_to_bytes.convert(image, image_format, is_RGB)
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': image_bytes},
        data={'size': 'auto', 'type': 'person'},
        headers={'X-Api-Key': 'APIKEY'},
    )
    if response.status_code == requests.codes.ok:
        image_alpha = bytestream_to_np.convert(response.content)
        return image_alpha
    else:
        print("Error:", response.status_code, response.text)
        raise ValueError
