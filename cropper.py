import numpy 
import cv2

def crop(img,isColor):
	if isColor:
		img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		img_gray = img
	ret, thresh1 = cv2.threshold(img_gray,245,255,cv2.THRESH_BINARY)
	# cv2.imwrite("thresholdcheck2.jpg",thresh1)
	ret, thresh = cv2.threshold(thresh1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	imga ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	# print(len(contours))
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

	cnt = contours[bigIndex]
	x,y,w,h = cv2.boundingRect(cnt)
	# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	cropped_image = img[y:y+h, x:x+w]
	return cropped_image


