import cv2
import numpy as np
import time

classifierFace  = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

videoCam = cv2.VideoCapture(0)

if not videoCam.isOpened():
    print("The camera is not accessible")
    exit()

buttonispressed = False
while (buttonispressed == False):
    ret, framework = videoCam.read()

    if ret == True:
        gray = cv2.cvtColor(framework, cv2.COLOR_BGR2GRAY)
        dafFace = classifierFace.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 2)

        for (x, y, w, h) in dafFace:
            cv2.rectangle(framework, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        #print("Number of Faces detected: ", len(dafFace))
        teks = "Number of Faces Detected = " + str(len(dafFace))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(framework, teks, (0, 30), font, 1, (255, 0, 0), 1)

        cv2.imshow("Results", framework)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            buttonispressed = True
            break


videoCam.release()
cv2.destroyAllWindows()