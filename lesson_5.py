import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def callback(x):
    pass


def getShape(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    shape = 'nan'

    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
    elif 5 == len(approx):
        shape = "pentagon"
    else:
        shape = "circle"
    return shape


upH = 126
upV = 255
upS = 189
lowH = 96
lowV = 147
lowS = 23

while True:

    ret, frame = cap.read()

    frame = cv2.blur(frame, (30, 30))

    lower = np.array([lowH, lowV, lowS])
    upper = np.array([upH, upV, upS])

    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hvs, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    _, thresh = cv2.threshold(mask, 127, 255, 0)

    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    needContours = []

    for contour in contours:
        if 1000 < cv2.contourArea(contour):
            needContours.append(contour)
        else:
            print(len(contour))

    cv2.drawContours(frame, needContours, -1, (0, 255, 0), 3)

    x, y = 0, 0

    for contour in needContours:
        t1, t2 = contour[0][0]
        x += t1
        y += t2

    if len(needContours) > 0:
        for c in needContours:
            maxY = (0, 0)
            minY = (0, 99999)
            maxX = (0, 0)
            minX = (99999, 0)
            for d in c:
                x_cur, y_cur = d[0]
                # print(x_cur, y_cur)
                # print(minX, minY, maxX, maxY)
                if x_cur < minX[0]:
                    minX = (x_cur, y_cur)
                if x_cur > maxX[0]:
                    maxX = (x_cur, y_cur)
                if y_cur < minY[1]:
                    minY = (x_cur, y_cur)
                if y_cur > maxY[1]:
                    maxY = (x_cur, y_cur)

            (cx, cy), radius = cv2.minEnclosingCircle(c)
            center = (int(cx), int(cy))
            radius = int(radius)
            cv2.putText(frame, getShape(c), (int(cx + (radius * 2)), int(cy)), cv2.QT_FONT_NORMAL, 2, (0, 0, 0), 2, cv2.LINE_8)
            cv2.circle(frame, center, int(radius * 1.5), (0, 255, 0), 2)
            cv2.circle(frame, minY, int(5), (255, 255, 255), -1)
            cv2.circle(frame, minX, int(5), (255, 255, 255), -1)
            cv2.circle(frame, maxX, int(5), (255, 255, 255), -1)
            cv2.circle(frame, maxY, int(5), (255, 255, 255), -1)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
