import cv2
from face_recognition.api import face_encodings
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
classNames = []

# grabbing the list of images in the  Images Attendance
Mylist = os.listdir(path)
print(Mylist)

for cl in Mylist:
    curImg = cv2.imread(f"{path}/{cl} ")
    images.append(curImg)
    # appending the class names and removing the jpg at the end and saving it to classNames
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# creating function for the encoding


def findEncodings(images):
    encodeList = []
    for img in images:
        # convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # FIND THE ENCODING
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# WRITING TO THE CSV FILE
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList) 
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            yearString = now.strftime('%d:%b:%Y')
            f.writelines(f"\n{name}, {dtString},{yearString}")



# markAttendance('Elon')




encodeListKnown = findEncodings(images)
print("Encoding complete")
# print(len(encodeListKnown))

# initializing the webcam

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # imgS = cv2.resize, (img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgS = cv2.cvtColor(np.float32(imgS), cv2.COLOR_BGR2RGB)
    # imgS = cv2.cvtColor(cv2.UMat(imgS), cv2.COLOR_RGB2GRAY)

    # print("hello")
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # print("hi")
    # finding the matches of the faces with the current webcame and image in our directory
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        #print("entered")
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2 )
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,
            (255,255,255),2)
            markAttendance(name)

    
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
