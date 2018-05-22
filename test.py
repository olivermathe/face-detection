import cv2
import numpy as np

img = cv2.imread('images/image_1.jpg')

def imgAvg(imgArr):

    pixAvg = []

    for row in imgArr:

        for pix in row:

            pixAvg.append(np.average(pix))

    return np.average(pixAvg)

def blackAndWhite(imgArr):

    newRow = []
    newImg = []
    
    average = imgAvg(imgArr)
    
    for r in range(0,len(imgArr)):

        for p in range(0,len(imgArr[r])):
            
            if(np.average(imgArr[r][p]) > average):
                imgArr[r][p] = [0,0,0]
            else:
                imgArr[r][p] = [255,255,255]

    return imgArr

img = blackAndWhite(img)
#print(img)
cv2.imshow("img",img)

#print(imgArray)
