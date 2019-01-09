import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def callback(x):
    pass


upH = 126
upV = 255
upS = 189
lowH = 96
lowV = 147
lowS = 23

while True:

    ret, frame = cap.read()

    frame = cv2.blur(frame, (15, 15))

    lower = np.array([lowH, lowV, lowS])
    upper = np.array([upH, upV, upS])

    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hvs, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    ret, thresh = cv2.threshold(mask, 127, 255, 0)

    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    needContours = []

    for contour in contours:
        if 200 < len(contour) < 1000:
            needContours.append(contour)

    cv2.drawContours(frame, needContours, -1, (0, 255, 0), 3)

    x, y = 0, 0

    for contour in needContours:
        t1, t2 = contour[0][0]
        print(contour[0][0])
        x += t1
        y += t2

    if x != 0 and y != 0:

        print("end")

        cv2.circle(frame, (x, y), int(100), (255, 255, 255), 10)

        cv2.putText(frame, str(x) + " " + str(y), (x, y), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5, cv2.LINE_8)

    else:
        print("not found")

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
