import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "haarcascade/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

im = cv2.imread('images/image_5.jpg',1)
font = cv2.FONT_HERSHEY_SIMPLEX

gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray, 1.2,5)

for(x,y,w,h) in faces:
    
    Id, conf = recognizer.predict(gray[y:y+h,x:x+w])

    conf = int(np.round(conf))
    
    print(conf)
    if(conf > 50):
        if(Id==1):
            Id="Matheus "
        elif(Id==2):
            Id="Flavia "
        elif(Id==3):
            Id="Mano "
        elif(Id==4):
            Id="Tank "
        elif(Id==5):
            Id="Wagner " 
    else:
        Id="Unknown"

    if Id != "Unknown":
        #cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        cv2.putText(im,Id,(x,y+h+30),font,1,(0,231,255),2)
        cv2.putText(im,str(conf)+"%",(x,y+h+45),font,.5,(252,1,252),1)
    cv2.imshow('im',im) 
if cv2.waitKey(10) and 0xFF==ord('q'):
    cv2.destroyAllWindows()
