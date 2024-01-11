import cv2
import numpy as np
import face_recognition
import os

path = '/Users/rahagolsorkhi/Documents/projects/photos'
images = []
imgNames = []
myList = os.listdir(path)

for cls in myList: #looping through images 
    curimg = cv2.imread(f'{path}/{cls}') #reading each image with openCV
    images.append(curimg) #adding images
    imgNames.append(os.path.splitext(cls[0])) #adding image names

def findEncodings(images): #converts each image to RGB, and does face encoding for more precise result
    encodeList = []
    for img in images:
        img = cv2.cvtColor('img', cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        for encode in encodings:
            encodeList.append(encode)
    return encodeList 

encodeListKnown = findEncodings(images) #calls findEncodings function, and prints the resulting list of face encodings
print(encodeListKnown)
print("Encoding Completed.")
    
cap = cv2.VideoCapture(0)
while True: #setting up video capturing
    success, img = cap.read() 
    #cap.read get frames from webcam, returns tuple 
    #which the first element is 'True' or 'False' meanning successful or unsuccessful process.
    imgSize = cv2.resize(img, (0, 0), None, 0.2, 0.2)
    imgSize = cv2.cvtColor(imgSize, cv2.COLOR_BGR2RGB)

    faceFrames = face_recognition.face_locations(imgSize)
    encodeFrames = face_recognition.face_encodings(imgSize, faceFrames)

    for encode_face, faceloc in zip(encodeFrames, faceFrames):
        #Iterating over each face in the frame, comparing it with the known encodings, and identifying potential matches.
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        faceDis = face_recognition.face_distance(encodeListKnown, encode_face)
        matching = np.argmin(faceDis)

        if matches[matching]:
            name = imgNames[matching].upper()
            print(name)
            
            # Draw a rectangle around the face
            top, right, bottom, left = faceloc
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            # Display the name near the face
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


