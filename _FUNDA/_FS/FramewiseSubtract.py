import cv2 as cv
import numpy as np


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    if max_val != min_val:
        out = (im.astype('float') - min_val) / (max_val - min_val)
    else:
        out = im.astype('float') / 255
    return out


DATA_PATH = "D:/Proj/UAV/dataset/drones/"
DATA_POSTFIX = ".avi"
DATA_NUM = 37

MAX_TARGET_NUM = 3

cap = cv.VideoCapture(DATA_PATH + "Video_%s"%DATA_NUM + DATA_POSTFIX)
prevFrame = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # now is uint
    frame = im2double(frame)
    rows, cols = frame.shape
    
    frame = cv.GaussianBlur(frame,(5, 5), 1, 1)
    if len(prevFrame) != 0:
        imBS = frame - prevFrame
        idx = np.argmax(imBS)
        v1 = (int(int(idx) % int(cols) - 40), int(int(idx) / int(cols) - 40))
        v2 = (int(v1[0] + 80), int(v1[1] + 80))

        imBS[imBS > 0.2 * np.max(imBS)] = 1
        cv.rectangle(frame, v1, v2, 1, 2)
        cv.rectangle(imBS, v1, v2, 1, 2)
        cv.imshow("frame", frame)
        cv.imshow("Subtraction", imBS)
        cv.waitKey(0)
    prevFrame = frame
    
    # cv.destroyAllWindows()
    