import cv2  # image processing using OpenCV.
import cvzone
from cvzone.FaceDetectionModule import FaceDetector     #for face detection
from time import time       # timestamping saved files.

#############################
#Parameters for configuration
classID = 0     # 0 is fake and 1 is real #Helps in differentiating between genuine and spoof images
outputFolderPath = 'Dataset/all'    #Ensures that all data is systematically saved
confidence = 0.8   # high-confidence detections are processed and saved. This helps improve the quality of the collected dataset.

save = True #Provides flexibility during development to either save data or just run the detection pipeline without creating files, which can be useful for testing.

blurThreshold = 35 # larger is more focus   # Ensures that only clear, high-quality images are saved, which is important for training an accurate model.
debug = True    #Allows developers to see the intermediate steps and results, making it easier to identify and fix issues in the detection pipeline.
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480  #camera dimensions, setting dimensions for camera capture
floatingPoint = 6
#############################

cap = cv2.VideoCapture(0) #cap oj created....capture vid from default camera ...0 index means first default camera
cap.set(3, camWidth)    # camera resolution
cap.set(4, camHeight)   #3 and 4 is the height and width of the frame , according to camw and camh values ,set above
detector = FaceDetector()   #obj of facedetector from cvzone lib
# this obj ^detector^ is used to find and
# annote faces in images i.e. frames that are captured

while True:
    success, img = cap.read()   # success is a boolean value that tells if the frame was captured successfully and stored in img variable
    #cap.read() reads the next frame from the webcam.
    imgOut = img.copy() #copy of the captured frame which is stored in img

    img, bboxs = detector.findFaces(img, draw = False)
# detector.findFaces(img, draw = False) processes the img to find faces
#bbox - boundbox around detected face
    #draw = False means dont draw now , we have to draw while debugging

    listBlur = [] #true/false values indicating if the faces are blur or not
    # to store boolean vals indicating if detected face is blurry or not

    listInfo = [] #the normalized values and the class name for the label txt file
#stores normalized coordinates and class labels

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]
            #print(x, y, w, h)
            #iterates through each detected face

            # ------ check the score ------
            if score > confidence:
                # only the faces with detection score higher than the set confidence score are processed further
    #maintain quality of data

                # ------ adding an offset to the face detected ------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                #adds margin around detected face

                # ------ to avoid values below 0 i.e. negative values------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # ------ find the blurriness ------

                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # extracts the face region from the image
                # calculate laplacian variance of face region to measure blurriness
                #compares that to a blurTHreshhold and appends result to listBlur list

                # ------ normalize values ------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)

                #Converts boundbox co ordinates to a normalized scale

                # ------ to avoid values above 1 ------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                #should be less than 1 , range(0,1)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
                #append normalized values to to listInfo

                # ------ drawing ------
                cv2. rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20), scale = 2, thickness = 3)

                #drawing bounding box
                #visual feedback for debugging

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20), scale=2,
                                       thickness=3)
        # ------ to save ------
        if save:
            if all(listBlur) and listBlur != []:
                # ------ save image ------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                print(timeNow)
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                # ------ save label text file ------
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()
                    #save images to specified folder
                    #generate timestamp to use as a unique file name for saving the images

                    ##label information to corresponding file



    cv2.imshow("images", imgOut)
    #display processed image with annotations
    #realtime feedback for users
    cv2.waitKey(1)

