import cv2
import sys
import glob

#cascPath = "C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml"
cascPath = "./haarcascade_frontalface_default.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
files=glob.glob("*.jpg")
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
    left = 15
    right = 15
    top = 15
    bottom = 15
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
   #displaying the faces and number of faces are found
    cv2.imshow("Faces found", image)
cv2.waitKey(0)
