''' Face Detection '''

import cv2
import os

# pip uninstall opencv-python
# python is >= 3.13+
# python version 3.8 - 3.12
# python install opencv-contrib-python
# add opencv and python in environment variable 

# create new environment
# environment setup: conda create (conda commands)
# install python 3.8 - 3.12
# install opencv

detector = cv2.CascadeClassifier(r"F:\python\haarcascade_frontalface_alt (1).xml") # loading haarcascade_frontalface_default.xml file 
cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL) # window title & window size normal
cam = cv2.VideoCapture(1) # 1 is used for external camera


while True:

    rect, frame = cam.read() # rect = Is the camera starts & frame = converting the object in the array format
    faces = detector.detectMultiScale(frame, 1.2) # detectMultiScale is used to detect the face in the image
    print(len(faces)) # Optional 

    # The area it is capturing is in the form of rectangle, so we need to draw rectangle on the face detected area
    for (x, y, w, h) in faces: # x = x coordinate, y = y coordinate, w = width of rectangle, h = height of rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 100), 2) # rectangle on the face detected area

    cv2.imshow("Face Detection", frame) # First one for the title and 2nd for showing the image
    if cv2.waitKey(5) == ord('q'): # ord('q') = q ka ascii value return karega, agar q press kiya to loop break ho jaayega
        break

cam.release() # For releasing the camera
cv2.destroyAllWindows() # Destroy all the windows opened by OpenCV


'''  detect face '''

def face_detect(frame):
    detector = cv2.CascadeClassifier(r"F:\python\haarcascade_frontalface_alt (1).xml") # loading haarcascade_frontalface_default.xml file
    faces = detector.detectMultiScale(frame, 1.2) # detectMultiScale is used to detect the face in the image
    return faces # return the face detected area in the form of rectangle

''' gray scale '''

def gray_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting the image into gray scale
    # return gray # return the gray scale image

''' cut face '''

def cut_face(image, faces_coord):
    cut_faces = []
    for (x, y, w, h) in faces_coord: 
        face = image[y:y+h, x:x+w]
        cut_faces.append(face) # append the cut face into the list
    return cut_faces # return the list of cut faces

''' 

Defalut array (Like in this way the data will come)
[[21,78,65,54],
[21,78,65,54],
[21,78,65,54],
[21,78,65,54]]

'''

''' resize '''

def resize(images, size=(80, 100)):
    resized_images = []
    for image in images:
        img = cv2.resize(image, size) # resizing the image into 80x100 (Can change the image size as per requirement)
        resized_images.append(img) # append the resized image into the list
    return resized_images # return the list of resized images

''' normalise intensity '''

def normalize_intensity(images):
    normalized_faces = []
    for image in images:
        normalized_faces.append(cv2.equalizeHist(image)) # MA'AM WRITE IMG INSTEAD OF IMAGE # normalising the intensity of the image & append the normalised image into the list
    return normalized_faces # return the list of normalised imagesq

''' image plot '''

import matplotlib.pyplot as plt # for showing the image matplotlib is used

def plot(image, title = " "):
    plt.figure(figsize=(12, 12)) # size of the image
    if image.shape == 3:
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting the image into RGB format (Can covert into gray scale also (if the shape is in 3D))
        plt.imshow(image, c = "gray") 
        plt.title(title)
        plt.axes('off')
        plt.show()

''' Pipeline '''

def pipeline(image, face_coord):
    faces = cut_face(image, face_coord) # cut the face from the image
    faces = resize(faces) # resize the cut face into 80x100 (Can change the image size as per requirement)
    faces = normalize_intensity(faces) # normalise the intensity of the resized face
    return faces # return the list of faces

''' Draw Rectangle '''
def draw_rectangle(frame, coords):
    for (x, y, w, h) in coords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 2) # draw a rectangle around the face
    # return frame # return the image with rectangle drawn around the face (When needed we can call)

''' DATASET CREATION - Let's create our own images dataset! '''

name = input("Enter your name: ") 
no_samples = int(input("Enter the number of samples you want to capture: ")) # number of samples we want to capture

folder = "dataset/" + name.lower() # folder name where we want to save the images
if os.path.exists(folder): # if the folder already exists then delete it and create a new one
    print("Folder with this name is already exists")
else:
    os.mkdir("f:\\python\\folder")
    start_cap = False # for starting the capturing of images
    sample = 1 # Using the same sample name

    cam = cv2.VideoCapture(0) 

    while True:
        rect, frame = cam.read()
        gray = gray_scale(frame) # converting the image into gray scale
        coords = face_detect(gray) # detect the face in the image
        if len(coords) == 0: 
            faces = pipeline(gray, coords) 
            image_name = folder + "/" + str(sample) + ".jpg" # image name will be 1.jpg, 2.jpg, 3.jpg, 4.jpg, 5.jpg, 6.jpg, 7.jpg, 8.jpg, 9.jpg, 10.jpg
            
            ''' cv2.imwrite(img, image_name, 0) # save the image # DO THIS USING FOR LOOP! '''
            
        else:
            print("No face found!")