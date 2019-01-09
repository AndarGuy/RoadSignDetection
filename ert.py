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

    cv2.putText(frame, "CERE", (10, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 5, cv2.LINE_8)

    lower = np.array([lowH, lowV, lowS])
    upper = np.array([upH, upV, upS])

    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hvs, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    M = cv2.moments(mask)
    if (M["m00"] != 0):
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        print(x, y)
    else:
        print("Not found")

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask, contours, -1, (0, 255, 0), 3)

    cv2.imshow("frame", frame)
    cv2.imshow("res", res)
    cv2.imshow("mask", mask)
    cv2.imshow("grey", hvs)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
