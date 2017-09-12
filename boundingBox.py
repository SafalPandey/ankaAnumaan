import cv2
import numpy as np
from cropper import crop

path = "E:/_nepali char/data/safal 001.jpg"

img = cv2.imread(path)
img = crop(img,True)
img_gray= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
img_gray = crop(img_gray,False)
ret, thresh1 = cv2.threshold(img_gray,245,255,cv2.THRESH_BINARY)
cv2.imwrite("thresholdcheck.jpg",thresh1)
ret, thresh = cv2.threshold(thresh1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
imga ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

print(len(contours))
for i in range(0, len(contours)):
    area = cv2.contourArea(contours[i])
    if i == 0:
        maxArea = area
        # print(maxArea)
        # print(img_gray.size)
        bigIndex = i
    if area > maxArea:
        maxArea = area
        bigIndex = i
# print(cv2.contourArea(contours[2]))
print(bigIndex)
# cv2.drawContours(img, contours[index], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
print(hierarchy)
# maxAreaContour = hierarchy[index]
# print(maxAreaContour)
i = hierarchy[0][bigIndex][2]
print(i)
count = 0
while True:
	# print("here")
	count+=1
	x,y,w,h = cv2.boundingRect(contours[i])
	

	boxArea = w * h
	print("BA:"+str(boxArea))
	if boxArea < img_gray.size/100 : 
		# print("herein")
		i = hierarchy[0][i][0]
		# print(i)
		if(i == -1):
			break
		continue

	childIndex = hierarchy[0][i][2]
	# cv2.drawContours(img, contours[childIndex], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)

	while True:
		x,y,w,h = cv2.boundingRect(contours[childIndex])

		contentArea = w * h
		print(contentArea)
		

		if (contentArea > 900  ):
			cropped_image = img[y:y+h, x:x+w]
			print(childIndex)
			cv2.imwrite("E:/_nepali char/croppedchildren/"+str(childIndex)+"AB.jpg",cropped_image)

			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

			cv2.drawContours(img, contours[childIndex], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)
			break
	
		if(hierarchy[0][childIndex][0] == -1):
			break

		else:
			childIndex = hierarchy[0][childIndex][0]

			# cv2.imwrite('convehull.jpg',img)
		
	
	i = hierarchy[0][i][0]
	if(hierarchy[0][i][0]== -1):
		break
	# pass
# cv2.drawContours(img, contours[52], -1, (255,0,255),2, cv2.LINE_AA, maxLevel=2)


nextBoxIndex = hierarchy[0][0][2]
# firtChild = contours[firstChildIndex]
firstChildIndex = hierarchy[0][nextBoxIndex][2]

print(firstChildIndex)



print(contours[2])

cnt = contours[bigIndex]
x,y,w,h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# cropped_image = img[y:y+h, x:x+w]
cv2.imwrite('convehull.jpg',img)

#cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
