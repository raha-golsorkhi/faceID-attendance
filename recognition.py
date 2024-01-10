import cv2
import numpy as np
import face_recognition
import os

path = 'Documents/projects/photos'
images = []
imgNames = []
myList = os.listdir(path)

for cls in myList:
    curimg = cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    imgNames.append(os.path.splitext(cls[0]))

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor('img', cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList 

encodeListKnown = findEncodings(images)
print(encodeListKnown)
print("Encoding Completed.")
    
webcam = cv2.VideoCapture(0)
while True:
    success, img = webcam.read()
    imgSize = cv2.resize(img, (0, 0), None, 0.2, 0.2)
    imgSize = cv2.cv2Color(imgSize, cv2.COLOR_BGR2RGB)

    faceFrame = face_recognition.face_locations(imgSize)[0] 
    encodeFrame = face_recognition.face_encodings(imgSize, faceFrame)

    for encode_face, faceloc in zip(encodeFrame, faceFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        faceDis = face_recognition.face_distance(encodeListKnown, encode_face)
        matching = np.argmin(faceDis)

        if matches[matching]:
            name = imgNames[matching].upper()
            print(name)

