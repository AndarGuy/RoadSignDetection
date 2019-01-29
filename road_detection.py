import cv2
import numpy as np
import color_utills as cu

color = cu.get_color('black')
frame = cv2.imread('./img/test/Picture110.jpg')


def get_deviation(frame):
    h, w, _ = np.shape(frame)
    frame = frame[int(h - h * 0.2):h, 0:w]
    frame = cv2.bilateralFilter(frame, 9, 120, 120)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(color[0]), np.array(color[1]))
    M = cv2.moments(mask)

    try:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        half = w / 2
        center_diff = w / half - x
        diff_in_percents = (center_diff / half) + 1
    except ZeroDivisionError:
        return '0'

    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
    cv2.imshow('road', frame)

    return diff_in_percents
