import numpy
import cv2

img = cv2.imread('D:/_nepali char/30x30_data/1.suvash1 011. of 001295.jpg')
rot = numpy.rot90(img,-1)

cv2.imshow('ree',rot)
cv2.waitKey(0)
