import math
import time
import cv2  #opencv used for computer vision tasks
import cvzone #simplify drawing operations like rectangles etc
from ultralytics import YOLO
#to detect objects or faces in the video feed
#fast and accurate face detection

confidence = 0.6
#the frames with confidence score higher than 0.6 are valid and processed further

cap = cv2.VideoCapture(0)  # For default Webcam
cap.set(3, 640)     #setting video frame
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For testing prerecorded videos

model = YOLO("../models/l_version_1_300.pt")
#loading pretrained yolo model
# for
classNames = ["fake", "real"]
#classnames defined

prev_frame_time = 0 #timestamp of previous frame
new_frame_time = 0  #''     ''      new ""
#to measure frame timings
#for FPS calculations


while True:
    new_frame_time = time.time()
    #capture current timefor time difference between frames

    success, img = cap.read()
    #read frame from cap object
    #success- bool to check if frame was read successfully or not


    results = model(img, stream=True, verbose=False)
    #running YOLO model

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
           # iterates through detected obj and extracts the bounding box co ordinate


            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # confidence score of the detectction

            # Class Name
            cls = int(box.cls[0])
            # class id of the detected obj

            if conf > confidence:
# if exceeds the predefined threshhold, real and green if not , then red
                if classNames[cls] == 'real':
                    color = (0, 255, 0) #green
                else:
                    color = (0, 0, 255) #red

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)  #bounding box with specified color
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)
                #displays class name and confidence score of the image

    fps = 1 / (new_frame_time - prev_frame_time)
    # Calculate FPS
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    #shows image with title Image

    cv2.waitKey(1)
#for the display to be refreshed