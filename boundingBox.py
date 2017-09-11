import cv2
import numpy as np

img = cv2.imread('E:/_nepali char/test1.jpg')
img_gray= cv2.imread('E:/_nepali char/test1.jpg',cv2.IMREAD_GRAYSCALE)

ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
imga ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

print(len(contours))
for i in range(0, len(contours)):
    area = cv2.contourArea(contours[i]);
    if i == 0:
        maxArea = area
        index = i
    if area > maxArea:
        maxArea = area
        index = i

print(index)
# cv2.drawContours(img, contours[index], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
print(hierarchy[0])
# maxAreaContour = hierarchy[index]
# print(maxAreaContour)
firstChildIndex = hierarchy[0][2][2]
# firtChild = contours[firstChildIndex]
print(firstChildIndex)

cv2.drawContours(img, contours[firstChildIndex], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
print(contours[2])

cnt = contours[index]
x,y,w,h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cropped_image = img[y:y+h, x:x+w]
cv2.imwrite('convehull.jpg',img)

#cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
