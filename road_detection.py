import cv2
import numpy as np
import color_utills as cu

# Цвет по умолчанию
color = cu.get_color('black')


# frame = cv2.imread('./img/test/Picture110.jpg')

# Функция изменения цвета
def change_road_color(color_str):
    global color
    color = cu.get_color(color_str)


# Функция возвращает в процентах отклонение от центра изображения
def get_deviation(frame):
    h, w, _ = np.shape(frame)
    frame = frame[int(h - h * 0.2):h, 0:w]
    frame = cv2.bilateralFilter(frame, 9, 120, 120)  # сглаживающий фильтр
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(color[0]), np.array(color[1]))
    M = cv2.moments(mask)

    try:
        x, y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])  # координаты центра дороги
        half = w / 2  # узнаем координату центра изображения
        center_diff = w / half - x  # узнаем отклонение в пикселях между центром и дорогой
        diff_in_percents = (center_diff / half) + 1  # узнаем отклонение в процентах
    except ZeroDivisionError:
        return '0'  # если что-то пошло не так и программа поделилв на нуль, возвращаем нуль

    # cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
    # cv2.imshow('road', frame)

    return diff_in_percents
