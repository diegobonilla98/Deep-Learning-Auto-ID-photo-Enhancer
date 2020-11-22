from PIL import Image
import base64
import io
import numpy as np


def from_file(filename, clean_chars=True, forbidden_chars=None):
    if clean_chars and forbidden_chars is None:
        forbidden_chars = ['\n', ' ']
    with open(filename, 'r') as file:
        bytes_stream = file.read()
    if clean_chars:
        for char in forbidden_chars:
            bytes_stream = bytes_stream.replace(char, '')
    bytes_stream = bytes_stream.encode()
    return convert(bytes_stream)


def convert(bytes_stream, dtype='uint8'):
    if not isinstance(bytes_stream, bytes):
        bytes_stream = bytes_stream.encode()
    base64_decoded = base64.b64decode(bytes_stream)
    image = Image.open(io.BytesIO(base64_decoded))
    return np.array(image, dtype=dtype)
