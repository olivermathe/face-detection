import cv2
import sys
import time
import numpy as np

faceDetected = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
img = cv2.imread('images/cat_1.jpg', 0)

faces = faceDetected.detectMultiScale(img, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 1)

cv2.imshow('Faces',img)

if(cv2.waitKey(1) == ord('q')):
    cv2.destroyAllWindows()
