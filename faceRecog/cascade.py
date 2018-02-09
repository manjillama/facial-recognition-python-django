import numpy as np
import cv2
import os
from PIL import Image
from settings import BASE_DIR


'''
    Take image as an input
    Detect face
    Crops the face part
    returns cropped face image
'''
def facecrop(image):
    facedata = BASE_DIR+'/ml/haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]

        #converts img array into grayscale
        gray_image = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
        # Converts np array back into image
        img = Image.fromarray(gray_image)
        # re-sizing to common dimension
        img = img.resize((150,150), Image.ANTIALIAS)
        #img.save('cropped.jpg')
    return img

#facecrop("ashesh.jpeg").show()
