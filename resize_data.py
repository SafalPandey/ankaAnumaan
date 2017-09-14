import numpy
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


trainingData = [f for f in os.listdir(os.path.join(BASE_DIR,'training_data'))    if os.path.isfile(os.path.join(BASE_DIR+'/training_data',f))]

# print(trainingData)

for f in trainingData:
    img = cv2.imread(os.path.join(BASE_DIR,'training_data/'+f))
    h,w , channels= img.shape
    if h < w:
        padded_img = cv2.copyMakeBorder(img,int((w-h)/2),int((w-h)/2),0,0,cv2.BORDER_CONSTANT,value = [255,255,255])
        # cv2.imshow('img',padded_img)
    if w < h:

        padded_img = cv2.copyMakeBorder(img,0,0,int((h-w)/2),int((h-w)/2),cv2.BORDER_CONSTANT,value = [255,255,255])

    resized_img = cv2.resize(padded_img,(30,30),interpolation=cv2.INTER_AREA)
    rotated_img = numpy.rot90(resized_img,-1)
    cv2.imwrite(os.path.join(BASE_DIR,'30x30_data/'+f),rotated_img)