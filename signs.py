import cv2
import numpy as np
from scipy.stats import itemfreq

cap = cv2.VideoCapture(0)


# Генератор пути до файла
def get_path_to_img(name):
    return PATH_TO_IMG + name + IMG_EXTENSION


def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]


class Template:
    image, name, color = [None] * 3

    def __init__(self, image, name):
        self.name = name
        self.image = image
        self.color = get_dominant_color(image, 1)


FILE_SIZE = '3x'  # размер изображения(можно оставить путым)
PATH_TO_IMG = './img/' + FILE_SIZE + '/'  # путь до папки с изображением
IMG_EXTENSION = '.png'  # расширение файла

# Указание названия файлов изображений
STOP_FILE_NAME = 'stop'
LEFT_FILE_NAME = 'left'
RIGHT_FILE_NAME = 'right'
FORWARD_FILE_NAME = 'forward'
BAD_ROAD_FILE_NAME = 'bad_road'
ONCOMING_PRIORITY_FILE_NAME = 'oncoming_priority'
ACCOMPANYING_PRIORITY_FILE_NAME = 'accompanying_priority'

# Инициализация cv темплейтов для файлов
stopTemplate = cv2.imread(get_path_to_img(STOP_FILE_NAME))
leftTemplate = cv2.imread(get_path_to_img(LEFT_FILE_NAME))
rightTemplate = cv2.imread(get_path_to_img(RIGHT_FILE_NAME))
forwardTemplate = cv2.imread(get_path_to_img(FORWARD_FILE_NAME))
oncomingPriorityTemplate = cv2.imread(get_path_to_img(ONCOMING_PRIORITY_FILE_NAME))
accompanyingPriorityTemplate = cv2.imread(get_path_to_img(ACCOMPANYING_PRIORITY_FILE_NAME))
badRoadTemplate = cv2.imread(get_path_to_img(BAD_ROAD_FILE_NAME))

signTemplates = [Template(stopTemplate, STOP_FILE_NAME), Template(leftTemplate, LEFT_FILE_NAME),
                 Template(rightTemplate, RIGHT_FILE_NAME),
                 Template(forwardTemplate, FORWARD_FILE_NAME),
                 Template(oncomingPriorityTemplate, ONCOMING_PRIORITY_FILE_NAME),
                 Template(accompanyingPriorityTemplate, ACCOMPANYING_PRIORITY_FILE_NAME),
                 Template(badRoadTemplate, BAD_ROAD_FILE_NAME)]

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 37)
    hvs = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hvs, np.array([int(i) - 70 for i in signTemplates[3].color]),
                       np.array([int(i) + 70 for i in signTemplates[3].color]))
    img, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    Z = frame.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((frame.shape))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
