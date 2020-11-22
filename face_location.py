import dlib
import cv2


detector = dlib.get_frontal_face_detector()


def get_face_location(gray):
    rects = detector(gray, 0)
    if len(rects):
        rect = rects[0]
        return rect.left(), rect.top(), rect.right(), rect.bottom()
    else:
        return None
