import sign_detection
import road_detection
import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (480, 360))
    print(road_detection.get_deviation(frame))
    frame = frame[0:240, 180:360]
    print(sign_detection.get_sign(frame))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
