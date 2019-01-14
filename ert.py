import cv2
import numpy as np

cap = cv2.VideoCapture(0)

window_name = "Window"


def callback(x):
    pass


cv2.namedWindow(window_name)
cv2.createTrackbar("lowH", window_name, 0, 255, callback)
cv2.createTrackbar("lowS", window_name, 0, 255, callback)
cv2.createTrackbar("lowV", window_name, 0, 255, callback)
cv2.createTrackbar("upH", window_name, 0, 255, callback)
cv2.createTrackbar("upS", window_name, 0, 255, callback)
cv2.createTrackbar("upV", window_name, 0, 255, callback)

while True:
    upH = cv2.getTrackbarPos("upH", window_name)
    upV = cv2.getTrackbarPos("upV", window_name)
    upS = cv2.getTrackbarPos("upS", window_name)
    lowH = cv2.getTrackbarPos("lowH", window_name)
    lowV = cv2.getTrackbarPos("lowV", window_name)
    lowS = cv2.getTrackbarPos("lowS", window_name)

    ret, frame = cap.read()

    frame = cv2.resize(frame, (240, 180))
    frame = cv2.bilateralFilter(frame, 9, 120, 120)

    lower = np.array([lowH, lowS, lowV])
    upper = np.array([upH, upS, upV])

    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hvs, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
