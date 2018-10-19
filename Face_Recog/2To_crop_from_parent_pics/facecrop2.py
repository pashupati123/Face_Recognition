import cv2
import sys
import glob
import os
import shutil
cascPath = "C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
files=glob.glob("*.jpg")
id = 1
d = 1
src = "C:\Users\hp\PycharmProjects\untitled"
for file in files:

    # Read the image
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
       # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Crop Padding
    left = 10
    right = 10
    top = 10
    bottom = 10
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Faces found", image)
    # id keeps track of cropped images of a selected image
    id = 1
    # create new path to store all cropped images of same image
    path = "C:\Users\hp\PycharmProjects\untitled\image"+str(d)
    d = d+1
    os.mkdir(path, 0777);
    for (x, y, w, h) in faces:
        cropped = image[y:y + h, x:x + w]
        cv2.imwrite("cropped_image" + str(id) + ".jpg", cropped)
        # move all cropped images of a image in to the same folder
        shutil.move("cropped_image" + str(id) + ".jpg",path)
        id = id + 1


    cv2.waitKey(0)  # wait between two images
