import cv2
import numpy as np
import random

def augment(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (5,5), 0)
    if random.random() < 0.3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * (0.5 + np.random.rand())
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image
