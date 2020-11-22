import numpy as np
import io
from PIL import Image


def convert(buffer_bytes, dtype='uint8'):
    if not isinstance(buffer_bytes, bytes):
        buffer_bytes = buffer_bytes.encode()
    image = Image.open(io.BytesIO(buffer_bytes))
    return np.array(image, dtype=dtype)
