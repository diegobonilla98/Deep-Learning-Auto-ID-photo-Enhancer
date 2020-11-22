from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K


tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
session = tf.Session(config=config)
K.set_session(session)

smile_model = load_model('lenet.hdf5')


def detect_smile(gray, loc):
    x1, y1, x2, y2 = loc
    gray_prep = cv2.resize(gray[y1: y2, x1: x2], (28, 28))
    gray_prep = gray_prep / 255.
    gray_prep = gray_prep.reshape((1, 28, 28, 1)).astype('float32')
    smiler = smile_model.predict(gray_prep)[0][1]
    if smiler > 0.4:
        return 2
    elif smiler > 0.002:
        return 1
    else:
        return 0
