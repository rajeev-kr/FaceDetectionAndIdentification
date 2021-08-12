from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import os
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())
def knn(train,test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i, -1]
        d = distance(test,ix)
        dist.append([d,iy])
    dk = sorted(dist, key = lambda x:x[0])[:k]
    labels = np.array(dk)[:,-1]

    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]
# ---------------------------------------------------------------

#---------------------------------------------------------------
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade/haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades-----------------------------------
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
camera_device = args.camera

#-- 2. Read the video stream-------------------------------------------
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
#-------------------------------------------Start of Program------------------------------
framesskip=0
face_data = []
dataset_path = 'HCCDATA/'
labels = []
class_id = 0
names = {}

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        print("Loaded "+ fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        target = class_id*np.ones((data_item.shape[0],))
        class_id +=1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# testing

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
        offset = 10 #cropping padding
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv.resize(face_section,(100,100))
        
        out = knn(trainset,face_section.flatten())
        pred_name = names[int(out)]
        cv.putText(frame,pred_name,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)# putting rectangle

    cv.imshow('Capture - Face detection', frame)
    # cv.imshow('Capture', face_section)

    key_pressed = cv.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
