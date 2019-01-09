import cv2
from colorthief import ColorThief

color_thief = ColorThief('./img/3x/forward.png')
dominant_color = color_thief.get_color(quality=1)
print(cv2.cvtColor(img_rgb,img_hsv,cv2.COLOR_RGB2HSV))
