import cv2
import sys
import time
import numpy as np

faceDetected = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
img = cv2.imread('dataset/image2.2.jpg', 0)

faces = faceDetected.detectMultiScale(img, 1.3, 5)
print faces
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 1)

cv2.imshow('Faces',img)

cv2.waitKey()