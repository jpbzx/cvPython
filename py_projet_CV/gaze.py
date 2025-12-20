import cv2
import numpy as np

def get_gaze_direction(eye_region):
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        gray, 70, 255, cv2.THRESH_BINARY_INV
    )

    h, w = thresh.shape
    left = thresh[:, 0:w//3]
    center = thresh[:, w//3:2*w//3]
    right = thresh[:, 2*w//3:w]

    left_count = cv2.countNonZero(right)
    center_count = cv2.countNonZero(center)
    right_count = cv2.countNonZero(left)

    if left_count > center_count and left_count > right_count:
        return "LEFT"
    elif right_count > left_count and right_count > center_count:
        return "RIGHT"
    else:
        return "CENTER"
