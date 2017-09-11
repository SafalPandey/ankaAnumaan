import cv2
import numpy as np

img = cv2.imread('E:/_nepali char/test1.jpg')
img_gray= cv2.imread('E:/_nepali char/test1.jpg',cv2.IMREAD_GRAYSCALE)

ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
imga ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
for i in range(0, len(contours)):
    area = cv2.contourArea(contours[i])
    if i == 0:
        maxArea = area
        bigIndex = i
    if area > maxArea:
        maxArea = area
        bigIndex = i

print(bigIndex)
# cv2.drawContours(img, contours[index], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
print(hierarchy)
# maxAreaContour = hierarchy[index]
# print(maxAreaContour)
i = hierarchy[0][bigIndex][2]
count = 0
while True:
	count+=1
	boxArea = cv2.contourArea(contours[i])
	if boxArea < img.size * 1/100 :
		i = hierarchy[0][i][0]
		continue
	childIndex = hierarchy[0][i][2]
	while True:
		contentArea = cv2.contourArea(contours[childIndex])
		if contentArea >= img.size * 0.25/100:
			cv2.drawContours(img, contours[childIndex], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
			cv2.imwrite('convehull.jpg',img)
		if(hierarchy[0][childIndex][0] == -1):
			break
		childIndex = hierarchy[0][childIndex][0]
	print(count)
	i = hierarchy[0][i][0]
	if(hierarchy[0][i][0]== -1):
		break
	# pass


nextBoxIndex = hierarchy[0][0][2]
# firtChild = contours[firstChildIndex]
firstChildIndex = hierarchy[0][nextBoxIndex][2]

print(firstChildIndex)



print(contours[2])

cnt = contours[index]
x,y,w,h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cropped_image = img[y:y+h, x:x+w]
cv2.imwrite('convehull.jpg',img)

#cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
