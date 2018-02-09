import os
from PIL import Image
import numpy as np

def getImagesWithID(path):
    #counterTest = 0
    h = 150
    w = 150
    # create a list for the path for all the images that is available in the folder
    # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
    # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
    #print imagePaths

    # Now, we loop all the images and store that userid and the face with different image list
    faces = []
    Ids = []
    for imagePath in imagePaths:
        #counterTest = counterTest + 1
        # First we have to open the image then we have to convert it into numpy array
        faceToResize = Image.open(imagePath).convert('L') #convert it to grayscale
        faceImg = faceToResize.resize((h,w), Image.ANTIALIAS)
        #faceImg.save('test/'+str(counterTest)+'.jpg')

        # converting the PIL image to numpy array
        # @params takes image and convertion format
        faceNp = np.array(faceImg, 'uint8')
        #Converting 2D array into 1D
        faceNp = faceNp.flatten()

        # Now we need to get the user id, which we can get from the name of the picture
        # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
        # Then we split the second part with . splitter
        # Initially in string format so hance have to convert into int format
        ID = int(os.path.split(imagePath)[-1].split('.')[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
        # Images
        faces.append(faceNp)
        # Label
        Ids.append(ID)
        #print ID
        # cv2.imshow("training", faceNp)
        # cv2.waitKey(10)
    return np.array(Ids), np.array(faces), h, w
