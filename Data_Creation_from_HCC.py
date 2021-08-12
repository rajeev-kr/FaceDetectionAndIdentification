from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

# def detectAndDisplay(frame):
    # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    # faces = face_cascade.detectMultiScale(frame_gray)
    # for (x,y,w,h) in faces:

        # center = (x + w//2, y + h//2)
        # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 1)
        # faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (0, 255, 0 ), 1)
    # cv.imshow('Capture - Face detection', frame)
# ---------------------------------------------------------------
framesskip=0
face_data = []
dataset_path = 'HCCDATA/'
#---------------------------------------------------------------
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade/haarcascade_frontalface_alt.xml')
# parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade/haarcascade_eye.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
# eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
# eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades-----------------------------------
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading eyes cascade')
#     exit(0)
camera_device = args.camera

#-- 2. Read the video stream-------------------------------------------
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
#-------------------------------------------Start of Program------------------------------
file_name = input("Enter your Name : ")
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    # print(faces)
    faces = sorted(faces, key = lambda f:f[2]*f[3])# sorting faces
    for (x,y,w,h) in faces[-1:]:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)# putting rectangle
        
        offset = 10 #cropping padding
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv.resize(face_section,(100,100))
        framesskip+=1
        if(framesskip%10==0):
            face_data.append(face_section)
            print(len(face_data))
    cv.imshow('Capture - Face detection', frame)
    # cv.imshow('Capture', face_section)

    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data)
print("Data sccesfully saved at "+dataset_path+file_name+'.npy')
cap.release()
cv.destroyAllWindows()