import cv2 as cv
import numpy as np

im0 = cv.imread("./_FUNDA/bs37_0.jpg")
im1 = cv.imread("./_FUNDA/bs37_1.jpg")

im = np.zeros(im0.shape)

im[:, 0:580, :] = im0[:, 0:580, :]
im[:, 580:640, :] = im1[:, 580:640, :]

cv.imshow("res", im)
cv.waitKey(0)

cv.imwrite("./_FUNDA/bs37.jpg", im)
