import cv2
import numpy as np
import os
import sys
import MODEL

def get_numbers(file_name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR,"input/"+file_name)
    img = cv2.imread(path)
    img_gray= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    ret, thresh1 = cv2.threshold(img_gray,245,255,cv2.THRESH_BINARY)
    ret, thresh = cv2.threshold(thresh1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    imga ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    print("Number of contours:",len(contours))
    for i in range(len(contours)):


        if hierarchy[0][i][3] != -1:
            continue
        x,y,w,h = cv2.boundingRect(contours[i])
        if w * h < 900:
            continue
        else:
            result = MODEL.classify(img[y:y+h, x:x+w],True)
            # cv2.imshow('img',img[y:y+h,x:x+w])
            # cv2.waitKey(0)

        cv2.drawContours(img, contours[i], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
        cv2.putText(img,str(result), (x-5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, 0)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite(os.path.join(BASE_DIR,"input/Contoured "+file_name),img)

if __name__ == "__main__":
    if (len(sys.argv) == 2):
        get_numbers(sys.argv[1])

    else:
        print("Please specify the filename of image to be classified.")
