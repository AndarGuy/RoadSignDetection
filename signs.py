import operator
import uuid

import cv2
import numpy as np
from colorthief import ColorThief

cap = cv2.VideoCapture(0)


# Генератор пути до файла
def getPath(name):
    return PATH_TO_IMG + name + IMG_EXTENSION


# Переводим цвет из RGB в HSV
def toHSV(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)


# Берем основной цвет изображения, короме черного и белого
def getDominantColor(path):
    color_thief = ColorThief(path)
    for color in color_thief.get_palette():
        if len(set([element > 20 for element in color])) > 1 or [element > 20 for element in color][0]:
            return color


def getShape(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

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


STR_COLOR_BLACK = 'black'
STR_COLOR_GREEN = 'green'
STR_COLOR_RED = 'red'
STR_COLOR_WHITE = 'white'
STR_COLOR_BLUE = 'blue'
STR_COLOR_BACKGROUND = 'background'

COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)

FILE_SIZE = '1x'  # размер изображения(можно оставить путым)
PATH_TO_IMG = './img/' + FILE_SIZE + '/'  # путь до папки с изображением
IMG_EXTENSION = '.png'  # расширение файла

# Указание названий
STOP = 'stop'
LEFT = 'left'
RIGHT = 'right'
FORWARD = 'forward'
BAD_ROAD = 'bad_road'
ONCOMING_PRIORITY = 'oncoming_priority'
ACCOMPANYING_PRIORITY = 'accompanying_priority'

TAGS = {STOP}

names = [STOP, LEFT, RIGHT, FORWARD, BAD_ROAD, ONCOMING_PRIORITY, ACCOMPANYING_PRIORITY]

# Инициализация cv темплейтов из файлов
templates = {name: cv2.cvtColor(cv2.imread(getPath(name)), cv2.COLOR_BGR2GRAY) for name in names}

# Указываем цветовые диапазоны для знаков
MANUAL_COLORS = True
BACKGROUND_ENABLE = True
colors = {}

if BACKGROUND_ENABLE:
    upHSV = [255, 28, 255]
    lowHSV = [49, 4, 132]
    colors[STR_COLOR_BACKGROUND] = (lowHSV, upHSV)
# Background color


if MANUAL_COLORS and not BACKGROUND_ENABLE:
    # Красный
    upHSV = [197, 255, 255]
    lowHSV = [0, 97, 91]
    colors[STR_COLOR_RED] = (lowHSV, upHSV)
    # Синий
    upHSV = [119, 209, 125]
    lowHSV = [103, 119, 60]
    colors[STR_COLOR_BLUE] = (lowHSV, upHSV)
    # Черный
    upHSV = [157, 120, 53]
    lowHSV = [0, 10, 12]
    colors[STR_COLOR_BLACK] = (lowHSV, upHSV)
    # Белый
    upHSV = [255, 28, 255]
    lowHSV = [49, 4, 132]
    colors[STR_COLOR_WHITE] = (lowHSV, upHSV)

# Создаем массив цветов дорожных знаков

FAULT = 50  # Указываем сдвиг погрешности
MIN_ACCURACY = 0.4

if not MANUAL_COLORS:
    for name in templates.keys():
        color = list(toHSV(getDominantColor(getPath(name))))
        colors[uuid.uuid4()] = ([c - FAULT for c in color], [c + FAULT for c in color])

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (np.array(frame).shape[1] // 3, np.array(frame).shape[0] // 3))
    log = np.array(frame)
    cont = np.array(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(frame, 9, 120, 120)
    hvs = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    potentialSigns = []

    for color in colors.keys():
        mask = cv2.inRange(hvs, np.array(colors[color][0]), np.array(colors[color][1]))

        ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_TOZERO)
        if BACKGROUND_ENABLE:
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [contour for contour in contours if cv2.contourArea(contour) > 200]
        hull = [cv2.convexHull(contours[i], False) for i in range(len(contours))]

        for h in hull:
            cv2.drawContours(log, h, -1, tuple(colors[color][0]), -1)

        for i in range(len(contours)):
            crop = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise
            cv2.drawContours(crop, hull, i, 255, -1)  # Draw filled contour in mask
            out = np.zeros_like(frame)  # Extract out the object and place into output image
            out[crop == 255] = frame[crop == 255]

            # Now crop
            (x, y, _) = np.where(crop == 255)
            (topX, topY) = (np.min(x), np.min(y))
            (bottomX, bottomY) = (np.max(x), np.max(y))
            out = frame[topX:bottomX + 1, topY:bottomY + 1]
            out = cv2.resize(out, (82, 82))
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out = cv2.bilateralFilter(out, 9, 75, 75)
            potentialSigns.append(out)

    matches = {}
    signs = {}
    for sign in potentialSigns:
        id = str(uuid.uuid4())[:4]
        matches[id] = {'id': id}
        signs[id] = sign
        for name in templates.keys():
            temp = cv2.resize(templates[name], (sign.shape[1], sign.shape[0]))

            res = cv2.matchTemplate(sign, templates[name], cv2.TM_CCOEFF_NORMED)
            matches[id][name] = np.max(res)

    color = COLOR_BLACK

    if matches:
        winner = max([max(list(matches[match].items())[1:], key=operator.itemgetter(1)) for match in matches],
                     key=lambda x: x[1])
        if winner[1] > MIN_ACCURACY:
            color = COLOR_GREEN
        cv2.putText(frame, str(winner), (5, 30), cv2.QT_FONT_NORMAL, 1, color, 1, cv2.LINE_8)
        cv2.imshow('temp', templates[winner[0]])
    cv2.imshow('frame', frame)
    cv2.imshow('log', log)
    cv2.imshow('contour', cont)
    cv2.imshow('blur', blur)
    # cv2.imshow('blur', blur)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
