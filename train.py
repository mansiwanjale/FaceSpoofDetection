import math
import time
import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
model = YOLO("../models/l_version_1_300.pt")
classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()    #current time to calculate fps
    success, img = cap.read()       #capture a frame from the webcam
    results = model(img, stream=True, verbose=False)    #Run YOLO model on the captured frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # loop through the detected objects and bounding box
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]    #coordinates of the bounding box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))  #draw rectangle
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 #calculate the confidence score
            # Class Name
            cls = int(box.cls[0])   #class index of the object
            if conf > confidence:
                #check if the confidence is above threshhold

                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
