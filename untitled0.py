# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:48:46 2019

@author: LG-PC
"""

import cv2, sys
import matplotlib.pyplot as plt
import pytesseract
import numpy as np

plt.style.use('dark_background')

img_jpg = 'test_sample03.png'
image = cv2.imread(img_jpg, cv2.IMREAD_COLOR)
image_gray = cv2.imread(img_jpg, cv2.IMREAD_GRAYSCALE)
'''
image = cv2.resize(image, (480,640))
image_gray = cv2.resize(image_gray, (480,640))
'''

blur = cv2.GaussianBlur(image_gray, ksize=(5,5), sigmaX=0)
ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(blur, 10, 250)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

#contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)

contours_xy = np.array(contours)
contours_xy.shape

# x의 min과 max 찾기
x_min, x_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        x_min = min(value)
        x_max = max(value)
 
# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        y_min = min(value)
        y_max = max(value)

# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min

VAR = -30

img_ori = image[y-VAR:y+h+VAR, x-VAR:x+w+VAR]



'''
img_ori = cv2.imread('test_img08.jpg',cv2.IMREAD_COLOR)
'''

height, width, channel = img_ori.shape

img = cv2.resize(img_ori, (480,640))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

img_blurred = cv2.GaussianBlur(gray, ksize = (5,5), sigmaX = 0)
# 노이즈를 제거 하기위해서
img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue = 255.0,
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType = cv2.THRESH_BINARY_INV,
        blockSize = 19,
        C = 9
        )

# 이미지에 threshold 지정

img_closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, structuringElement)

plt.figure(figsize = (12,10))
plt.imshow(img_ori, cmap = 'gray')

plt.figure(figsize = (12,10))
plt.imshow(img_closing, cmap = 'gray')

text1 = pytesseract.image_to_string(image, lang='kor',config='--oem 0 --psm 6')
text2 = pytesseract.image_to_string(img_closing, lang='kor',config='--oem 0 --psm 6')
print(text1)
print(text2)