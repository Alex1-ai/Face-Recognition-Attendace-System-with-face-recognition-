import numpy as np
import cv2
import face_recognition

# loading the images to face recognition

imgElon = face_recognition.load_image_file("imagesBasic/ElonMusk.jpg")
# convert it to rgb style
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# loading the images to face recognition

imgTest = face_recognition.load_image_file("imagesBasic/Billgate.jpg")
# convert it to rgb style
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# finding the face  for imgElon
faceLoc = face_recognition.face_locations(imgElon)[0]
# encoding the face
encodeElon = face_recognition.face_encodings(imgElon)[0]

# to check the face location on the pics
cv2.rectangle(imgElon,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255),2)


# finding the face  for imgElon
faceLocTest = face_recognition.face_locations(imgTest)[0]
# encoding the face
encodeTest = face_recognition.face_encodings(imgTest)[0]

# to check the face location on the pics
cv2.rectangle(imgTest,(faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255),2)

#print(faceLoc) # face location has 4 sides


# comparing the faces and find if it the same
results = face_recognition.compare_faces([encodeElon], encodeTest)
# finding the distance of the distance(the lower the distance the best the match)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)


print(results, faceDis)
cv2.putText(imgTest, f"{results}, {round(faceDis[0],2)}", (50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(225,0,0), 2)




# showing the face
cv2.imshow('ElonMusk', imgElon)
cv2.imshow('ElonTest', imgTest)
cv2.waitKey(0)
