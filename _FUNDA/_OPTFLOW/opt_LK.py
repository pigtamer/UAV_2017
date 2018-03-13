# Adapted from opencv 3.4.0 samples for python.
import numpy as np
import cv2 as cv

def draw_str(dst, target, s): 
    # show string on current frame, indicating frame info
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

lk_params = dict( winSize  = (10, 10),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 5,
                       blockSize = 5 )

DATA_PATH = "D:/Proj/UAV/dataset/drones/"
DATA_POSTFIX = ".avi"
DATA_NUM = 37

# cap = cv.VideoCapture(DATA_PATH + "Video_%s"%DATA_NUM + DATA_POSTFIX)
cap = cv.VideoCapture("X:/UAV/_FUNDA/_OPTFLOW/vid.mp4")
track_len = 10 # length of each track
detect_interval = 5
tracks = []
prev_gray = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        
        p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        MX_TAR = 0       

        
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
        
        tracks = new_tracks # update track of each frame

        cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0)) 
        # draw track in current frame
        # cv.rectangle(vis, (int(x - 40), int(y - 40)), (int(x + 40), int(y + 40)), (0, 0, 255), 2)


    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)
        p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray
    cv.imshow('Tracking', vis)
    ch = cv.waitKey(24)
    if ch == 27: # ESC
        break
