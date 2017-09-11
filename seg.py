import numpy as np
import cv2

img = cv2.imread('D:/_nepali char/data/safal 001.jpg')

gray = cv2.imread('D:/_nepali char/data/safal 001.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('gray.jpg', gray)

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imwrite('thresh.jpg',thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imwrite('opening.jpg', opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imwrite('sure_bg.jpg', sure_bg)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
ret, sure_fg = cv2.threshold(dist_transform,0*dist_transform.max(),255,0)
cv2.imwrite('sure_fg.jpg', sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,0,255]
cv2.imwrite('sasfal.jpg', img)
