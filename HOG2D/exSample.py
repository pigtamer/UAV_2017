# !/usr/bin/python
# this is for extracting 2-d hog feature from video seq
import os
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import annot_parser # this is not wrong as we've added

data_path = "D:/Proj/UAV/dataset/drones/"
data_postfix = ".mp4"

if not os.path.exists("./pos"): os.makedirs("./pos")
if not os.path.exists("./nega"): os.makedirs("./nega")

# parse annot into array
drones_nums = [1, 11, 12, 18, 19, 29, 37, 46, 47, 48, 49, 53, 55, 56]
TRAIN_SET_RANGE = [233] # select some videos by slicing

IF_SHOW_PATCH = True # warning: it can critically slow down extraction process
IF_PLOT_HOG_FEATURE = False

NEGA_SPF = 3
POS_SPF = 1
RAND_SAMPLE = 100
# VID_NUM = 1
PSIZE = 64

# parse videos in training set
t = time.time()
idx = 0
nidx = 0

for VID_NUM in TRAIN_SET_RANGE:

    locations, labels = annot_parser.parse("X:/UAV/annot/drones/", VID_NUM)
    data_num = VID_NUM

    cap = cv.VideoCapture(data_path + "Video_%s"%data_num + data_postfix)
    # cap = cv.VideoCapture(0)

    # parse each video    
    time_stamp = 0
    while(True):
        ret, frame = cap.read()
        if not ret: break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        x_0 = locations[time_stamp][0] # 1
        x_1 = locations[time_stamp][2] # 3
        y_0 = locations[time_stamp][1] # 2
        y_1 = locations[time_stamp][3] # 4

        sigma = 1 + 0.5*np.random.rand()

        if not x_0 == -1 :
            patch = frame[x_0:x_1, y_0:y_1]
            px, py = patch.shape
            # patch = cv.resize(patch, (PSIZE, PSIZE)) # size of target area varies in time so we resize each patch to a certain size, fitting HoG Descriptor.
            idx = idx + 1
            cv.imwrite("./pos/v_%d_%d.jpg"%(VID_NUM, idx),patch)

        npatch = frame.copy()
        if not x_0 == -1:
            npatch[x_0 - 20:x_1 + 20, y_0 - 20:y_1 + 20] = 0
        nidx = nidx + 1
        cv.imwrite("./nega/v_%d_%d.jpg"%(VID_NUM, nidx), npatch)

        if IF_SHOW_PATCH:
            cv.imshow("npatch", npatch) # if no target in annotated, then a negative sample is added from a random area on each frame. image window can flicker when negative samples are displayed.
            # cv.waitKey(24)
            if not x_0 == -1:
                cv.imshow("patch", patch) # if no target in annotated, then a negative sample is added from a random area on each frame. image window can flicker when negative samples are displayed.
                cv.waitKey(24)

        time_stamp = time_stamp + 1
        if time_stamp == len(labels) : break


elapsed_time = time.time() - t
print("Dataset generated in %s sec"%elapsed_time)