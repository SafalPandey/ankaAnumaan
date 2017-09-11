import cv2
import numpy as np

img = cv2.imread('D:/_nepali char/data/safal 001.jpg')
img_gray= cv2.imread('D:/_nepali char/data/safal 001.jpg',cv2.IMREAD_GRAYSCALE)

ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
imga ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours[37], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)

print(len(contours))
for i in range(0, len(contours)):
    area = cv2.contourArea(contours[i]);
    if i == 0:
        maxArea = area
    if area > maxArea:
        maxArea = area
        index = i
print(index)


cnt = contours[37]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imwrite('convehull.jpg',img)

#cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
