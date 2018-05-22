import cv2
import sys
import time
import numpy as np

faceDetected = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetected.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (10,255,234), 2)

    cv2.imshow('Faces',faces[0])

    if(cv2.waitKey(1) == ord('q')):
        break;

cap.release()
cv2.destroyAllWindows()
