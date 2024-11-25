import cv2 as cv

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
        return frame
