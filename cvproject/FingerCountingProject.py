import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "Fingerimages"
myList = os.listdir(folderPath)
print(myList)

overlayList = []

for imPath in myList:
    # importing images
    image = cv2.imread(f"{folderPath}/{imPath}")
    # print(f"{folderPath}/{imPath}")
    # we have imported, now we have to save the images
    overlayList.append(image)

print(len(overlayList))
detector = htm.handDetector(detectionCon=0.8)
pTime = 0

tipIds = [4, 8, 12, 16, 20] # these are the tips of all 5 fingers
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) !=0:
        fingers = []
        # the below if command will work correctly for the right hand
        # thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]: # here we are checking only the x-axis so we entered[1]
            fingers.append(1)
            # print("Index finger open")
        else:
            fingers.append(0)
        # remaining 4 fingers
        for id in range(1, 5):
            #  if the top value is less than bottom value the finger is opened and vice versa.
            # if lmList[8][2] < lmList[6][2]:
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:# here we are checking only the y-axis so we entered[2]
                fingers.append(1)
                # print("Index finger open")
            else:
                fingers.append(0)
                # print("finger closed")


        # print(fingers)
        #to count the number of values present of 1
        totalFingers = fingers.count(1)
        print(totalFingers)
        # to automate the different size image, then we can use the below line
        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1] # img[0:400, 0:200] -  image size, overlayList[0] - displaying the first image
        # If I want to shift the location of that image then follow the below code
        # img[100:500, 100:300] = overlayList[4]
        # overlayList[-1] - it will take the last image 6 when totalFinger is 0. [0 - 1] becomes [-1]
        cv2.rectangle(img, (20, 420), (170, 600), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 560), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 0, 0), 25)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (1100, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
