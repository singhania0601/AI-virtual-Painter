import cv2
import HandTrackingModule as htm
import time
import os
import numpy as np
import mediapipe as mp
brushThickness = 15
eThickness = 50
xp,yp = 0,0

folderPath = "photos"
myList = os.listdir(folderPath)
# print(myList)
overLay = []
for imgPath in myList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overLay.append(image)
# print(len(overLay))
header = overLay[0]
drawColor = (222,46,235)
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
detector = htm.handDetector()
imageCanvas = np.zeros((480,640,3),np.uint8)

while True:
    success, img=cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img,draw=False)
    lmList = detector.findPosition(img,draw = False)
    if len(lmList) != 0:

        # print(lmList)
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        fingers = detector.fingerUp()
        # print(fingers)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            if y1 < 90:
                if 142<x1<184:
                    header = overLay[0]
                    drawColor = (222, 46, 235)
                elif 226<x1<267:
                    header = overLay[1]
                    drawColor = (46, 46, 235)
                elif 303<x1<354:
                    header = overLay[2]
                    drawColor = (235, 58, 46)
                elif 393<x1<438:
                    header = overLay[3]
                    drawColor = (46, 146, 235)
                elif 476<x1<562:
                    header = overLay[4]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)



            print("Selection Mode")
        elif fingers[1] and fingers[2] == False:

            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            if xp == 0 and yp == 0:
                xp = x1
                yp = y1
            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, eThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp,yp = x1,y1
            print("Drawing Mode")

    imgGray = cv2.cvtColor(imageCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imageCanvas)
    img[0:90,0:640] = header
    img = cv2.addWeighted(img,0.5,imageCanvas,0.5,0)
    # cv2.imshow("video", imageCanvas)
    cv2.imshow("vid", img)
    cv2.waitKey(1)
