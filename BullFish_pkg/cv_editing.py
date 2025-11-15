import numpy as np
import cv2 as cv
from math import log10

def get_rm(x, y, angle):
    if angle > -180 and angle < -135:
        return cv.getRotationMatrix2D((x, y), -(angle + 180), 1.0)
    elif angle >= -135 and angle < -45 and angle != -90:
        return cv.getRotationMatrix2D((y, x), -(angle + 90), 1.0)
    elif angle >= -45 and angle < 45 and angle != 0:
        return cv.getRotationMatrix2D((x, y), -angle, 1.0)
    elif angle >= 45 and angle < 135 and angle != 90:
        return cv.getRotationMatrix2D((y, x), -(angle - 90), 1.0)
    elif angle >= 135 and angle < 180:
        return cv.getRotationMatrix2D((x, y), -(angle - 180), 1.0)
    else:
        return None

def frame_rotate(frame, x, y, angle, rm=None):
    if angle > -180 and angle < -135:
        frame_t = cv.rotate(frame, cv.ROTATE_180)
        return cv.warpAffine(frame_t, rm, (x, y))
    elif angle >= -135 and angle < -45 and angle != -90:
        frame_t = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        return cv.warpAffine(frame_t, rm, (y, x))
    elif angle == -90:
        return cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif angle >= -45 and angle < 45 and angle != 0:
        return cv.warpAffine(frame, rm, (x, y))
    elif angle >= 45 and angle < 135 and angle != 90:
        frame_t = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        return cv.warpAffine(frame_t, rm, (y, x))
    elif angle == 90:
        return cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    elif angle >= 135 and angle < 180:
        frame_t = cv.rotate(frame, cv.ROTATE_180)
        return cv.warpAffine(frame_t, rm, (x, y))
    elif angle == 180:
        return cv.rotate(frame, cv.ROTATE_180)
    else:
        return frame.copy()

def frame_crop(frame, crop_tlx, crop_tly, crop_x, crop_y):
    crop_brx = crop_tlx + crop_x
    crop_bry = crop_tly + crop_y
    if crop_x != 0 and crop_y != 0:
        return frame[crop_tly:crop_bry, crop_tlx:crop_brx]
    else:
        return frame

def frame_grc(frame, x, y, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y):
    frame_t = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_t = frame_rotate(frame_t, x, y, rotate, rm)
    return frame_crop(frame_t, crop_tlx, crop_tly, crop_x, crop_y)

def frame_blur(frame, ksize):
    if ksize > 0:
        return cv.GaussianBlur(frame, (ksize, ksize), 0)
    else:
        return frame.copy()

def max_entropy_threshold(image, threshold_reduction):
    hist, _ = np.histogram(image.ravel(), bins = 256, range=(0, 256))
    nhist = hist / hist.sum()
    pT = np.zeros(256)
    ii = 1
    while ii <= 255:
        pT[ii] = pT[ii - 1] + nhist[ii]
        ii += 1
    hB = np.zeros(256)
    hW = np.zeros(256)
    t = 0
    while t <= 255:
        if pT[t] > 0:
            hhB = 0
            ii = 0
            while ii <= t:
                if nhist[ii] > 0:
                    temp = nhist[ii] / pT[t]
                    hhB -= temp * log10(temp)
                ii += 1
            hB[t] = hhB
        pTW = 1 - pT[t]
        if pTW > 0:
            hhW = 0
            ii = t + 1
            while ii <= 255:
                if nhist[ii] > 0:
                    temp = nhist[ii] / pTW
                    hhW -= temp * log10(temp)
                ii += 1
            hW[t] = hhW
        t += 1
    hmax = hB[0] + hW[0]
    tmax = 0
    t = 1
    while t <= 255:
        h = hB[t] + hW[t]
        if h > hmax:
            hmax = h
            tmax = t
        t += 1
    return round(tmax * (1 - threshold_reduction / 100))

def sq_area(image, point, r):
    sq = image[(point[1] - r):(point[1] + r + 1), (point[0] - r):(point[0] + r + 1)]
    try:
        return np.sum(sq == 255)
    except:
        print('\nThe edge of the video is reached. Tracking might not be accurate.')
        return 99999
