import cv2
import color_utills as cu
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

GREEN = 'green'
RED = 'red'

colors = {GREEN: cu.get_color(GREEN), RED: cu.get_color(RED)}


def get_traffic_light(frame):
    colors_area = {
    color: len(np.where(cv2.inRange(frame, np.array(colors[color][0]), np.array(colors[color][1])) == 255)[0]) for color
    in colors.keys()}
    print(colors_area)


get_traffic_light(frame)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
