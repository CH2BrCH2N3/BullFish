import os
import traceback
import csv
import cv2 as cv
import numpy as np
import math
from time import time

class settings:
    def __init__(self, default):
        self.default = default
        self.user_set = default
        self.current = default
skip_opening = settings(0)
ask_skip = settings(1)
video_start = settings(-1)
video_end = settings(-1)
rotate = settings(181)
crop_tlx = settings(-1)
crop_tly = settings(-1)
crop_x = settings(-1)
crop_y = settings(-1)
swimarea_tlx = settings(-1)
swimarea_tly = settings(-1)
swimarea_x = settings(-1)
swimarea_y = settings(-1)
check = settings(1)
ksize = settings(5)
sampling_time = settings(0.2)
threshold1_reduction = settings(10)
save_binaryvideo = settings(1)
spine_analysis = settings(1)
contour_points_dist = settings(10)
auto_bg = settings(1)
fish_cover_size = settings(1.5)
threshold2_reduction = settings(0)
tail2_range = settings(90)
head_r = settings(15)
show_errors = settings(1)
turn_max = settings(6000)
save_annotatedvideo = settings(0)
if os.path.exists('bullfish_tracker_settings.csv'):
    with open('bullfish_tracker_settings.csv', 'r') as f:
        settings_dict = {row[0]: row[1] for row in csv.reader(f)}
        print('bullfish_tracker_settings.csv is opened for the settings.')
    try:
        skip_opening.user_set = int(settings_dict['skip_opening'])
        ask_skip.user_set = int(settings_dict['ask_skip'])
        video_start.user_set = int(settings_dict['video_start'])
        video_end.user_set = int(settings_dict['video_end'])
        rotate.user_set = int(settings_dict['rotate'])
        crop_tlx.user_set = int(settings_dict['crop_tlx'])
        crop_tly.user_set = int(settings_dict['crop_tly'])
        crop_x.user_set = int(settings_dict['crop_x'])
        crop_y.user_set = int(settings_dict['crop_y'])
        swimarea_tlx.user_set = int(settings_dict['swimarea_tlx'])
        swimarea_tly.user_set = int(settings_dict['swimarea_tly'])
        swimarea_x.user_set = int(settings_dict['swimarea_x'])
        swimarea_y.user_set = int(settings_dict['swimarea_y'])
        check.user_set = int(settings_dict['check'])
        ksize.user_set = int(settings_dict['k_size'])
        sampling_time.user_set = float(settings_dict['sampling_time'])
        threshold1_reduction.user_set = float(settings_dict['threshold1_reduction'])
        save_binaryvideo.user_set = int(settings_dict['save_binaryvideo'])
        spine_analysis.user_set = int(settings_dict['spine_analysis'])
        contour_points_dist.user_set = int(settings_dict['contour_points_dist'])
        auto_bg.user_set = int(settings_dict['auto_bg'])
        fish_cover_size.user_set = float(settings_dict['fish_cover_size'])
        threshold2_reduction.user_set = float(settings_dict['threshold2_reduction'])
        tail2_range.user_set = float(settings_dict['tail2_range'])
        head_r.user_set = float(settings_dict['head_r'])
        show_errors.user_set = int(settings_dict['show_errors'])
        turn_max.user_set = float(settings_dict['turn_max'])
        save_annotatedvideo.user_set = int(settings_dict['save_annotatedvideo'])
    except Exception:
        traceback.print_exc()
        print('Some of the settings cannot be found.\nDefault settings are used.')
else:
    print('bullfish_tracker_settings.csv cannot be found.')
    settings_dict = {'skip_opening': skip_opening.user_set,
                     'ask_skip': ask_skip.user_set,
                     'video_start': video_start.user_set,
                     'video_end': video_end.user_set,
                     'rotate': rotate.user_set,
                     'crop_tlx': crop_tlx.user_set,
                     'crop_tly': crop_tly.user_set,
                     'crop_x': crop_x.user_set,
                     'crop_y': crop_y.user_set,
                     'swimarea_tlx': swimarea_tlx.user_set,
                     'swimarea_tly': swimarea_tly.user_set,
                     'swimarea_x': swimarea_x.user_set,
                     'swimarea_y': swimarea_y.user_set,
                     'check': check.user_set,
                     'k_size': ksize.user_set,
                     'sampling_time': sampling_time.user_set,
                     'threshold1_reduction': threshold1_reduction.user_set,
                     'save_binaryvideo': save_binaryvideo.user_set,
                     'spine_analysis': spine_analysis.user_set,
                     'contour_points_dist': contour_points_dist.user_set,
                     'auto_bg': auto_bg.user_set,
                     'fish_cover_size': fish_cover_size.user_set,
                     'threshold2_reduction': threshold2_reduction.user_set,
                     'tail2_range': tail2_range.user_set,
                     'head_r': head_r.user_set,
                     'show_errors': show_errors.user_set,
                     'turn_max': turn_max.user_set,
                     'save_annotatedvideo': save_annotatedvideo.user_set}
    with open('bullfish_tracker_settings.csv', 'w') as f:
        for key in settings_dict:
            f.write(key + ',' + str(settings_dict[key]) + '\n')
    print('Default settings are created.')
if not skip_opening.user_set:
    if input('Enter e to exit the program to edit settings, others to continue:') == 'e':
        from sys import exit
        exit()
turn_max.user_set = turn_max.user_set * math.pi / 180
tail2_range.user_set = tail2_range.user_set * math.pi / 180

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_input(t, prompt):
    while True:
        inputs = input(prompt)
        try:
            return t(inputs)
        except Exception as error:
            print(error)
            print('...Try again')     

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

def frame_rotate(frame, x, y, angle, rm):
    if angle > -180 and angle < -135:
        frame_t = cv.rotate(frame, cv.ROTATE_180)
        frame_t = cv.warpAffine(frame_t, rm, (x, y))
    elif angle >= -135 and angle < -45 and angle != -90:
        frame_t = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        frame_t = cv.warpAffine(frame_t, rm, (y, x))
    elif angle == -90:
        frame_t = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif angle >= -45 and angle < 45 and angle != 0:
        frame_t = cv.warpAffine(frame_t, rm, (x, y))
    elif angle >= 45 and angle < 135 and angle != 90:
        frame_t = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame_t = cv.warpAffine(frame_t, rm, (y, x))
    elif angle == 90:
        frame_t = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    elif angle >= 135 and angle < 180:
        frame_t = cv.rotate(frame, cv.ROTATE_180)
        frame_t = cv.warpAffine(frame_t, rm, (x, y))
    elif angle == 180:
        frame_t = cv.rotate(frame, cv.ROTATE_180)
    else:
        frame_t = frame
    return frame_t

videofilenames = []
metadata = {}
for file in os.listdir('.'):
     
    filename = os.fsdecode(file)
    filename_split = os.path.splitext(filename)
    supported_formats = ['.avi', '.mp4']
    if filename_split[1] not in supported_formats:
        continue
    video = cv.VideoCapture(filename)
    if not video.isOpened():
        print(filename + ' cannot be opened.')
        continue
    print('\nOpening ' + filename + ' ...')
    if ask_skip.user_set and input('Enter s to skip inspecting this video, any other things to continue:') == 's':
        print(filename + ' skipped.')
        continue
    videofilenames.append(filename)
    
    try:
        
        videoname = filename_split[0]
        path = './' + videoname
        create_path(path)
        
        x_original = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        y_original = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        framenumber_original = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv.CAP_PROP_FPS)
        
        create_metadata = True
        if os.path.exists(path + '/' + videoname + '_metadata.csv'):
            while True:
                response = input(videoname + '_metadata.csv' + ' already exists. Enter y to use the existing metadata. Enter o to overwrite the metadata.')
                if response == 'y':        
                    create_metadata = False
                elif response != 'o':
                    print('Try again')
                    continue
                break
        
        if create_metadata:
            
            print('Creating metadata for ' + videoname + '...')
            
            if video_start.user_set == -1:
                video_start.current = get_input(int, 'What is the starting frame?')
            else:
                video_start.current = video_start.user_set
            if video_end.user_set == -1:
                video_end.current = get_input(int, 'What is the ending frame?')
            else:
                video_end.current = video_end.user_set
            while True:
                if video_end.current < framenumber_original and video_start.current < video_end.current:
                    break
                else:
                    print('Unacceptable range. Choose again.')
                    video_start.current = get_input(int, 'What is the starting frame?')
                    video_end.current = get_input(int, 'What is the ending frame?')
            
            video.set(cv.CAP_PROP_POS_FRAMES, video_start.current)
            ret, frame1 = video.read()
            cv.imwrite(videoname + '_Frame1.png', frame1)
            video.release()
            if rotate.user_set == 181:
                rotate.current = get_input(float, 'What is the video rotation angle?')
            else:
                rotate.current = rotate.user_set
            if crop_tlx.user_set == -1:
                crop_tlx.current = get_input(int, 'Enter the x-coordinate of the top-left corner of the cropping area:')
            else:
                crop_tlx.current = crop_tlx.user_set
            if crop_tly.user_set == -1:
                crop_tly.current = get_input(int, 'Enter the y-coordinate of the top-left corner of the cropping area:')
            else:
                crop_tly.current = crop_tly.user_set
            if crop_x.user_set == -1:
                crop_x.current = get_input(int, 'Enter the width (x) of the cropping area:')
            else:
                crop_x.current = crop_x.user_set
            if crop_y.user_set == -1:
                crop_y.current = get_input(int, 'Enter the height (y) of the cropping area:')
            else:
                crop_y.current = crop_y.user_set
            while check.user_set:
                try:
                    rm = get_rm(x_original, y_original, rotate.current)
                    frame_t = frame_rotate(frame1, x_original, y_original, rotate.current, rm)
                    crop_brx = crop_tlx.current + crop_x.current
                    crop_bry = crop_tly.current + crop_y.current
                    if crop_x.current != 0 and crop_y.current != 0:
                        frame_t = frame_t[crop_tly.current:crop_bry, crop_tlx.current:crop_brx]
                    cv.imwrite(videoname + '_edited_frame.png', frame_t)
                    if input('The edited frame is saved. Enter f to carry on, others to change:') == 'f':
                        break
                except Exception:
                    print('An error occurred when editing the frame. Change parameters.')
                    traceback.print_exc()
                rotate.current = get_input(float, 'What is the video rotation angle?')
                crop_tlx.current = get_input(int, 'Enter the x-coordinate of the top-left corner of the cropping area:')
                crop_tly.current = get_input(int, 'Enter the y-coordinate of the top-left corner of the cropping area:')
                crop_x.current = get_input(int, 'Enter the width (x) of the cropping area:')
                crop_y.current = get_input(int, 'Enter the height (y) of the cropping area:')
            size = frame_t.shape
            x_current = size[1]
            y_current = size[0]
            
            if swimarea_tlx.user_set == -1:
                swimarea_tlx.current = get_input(int, 'Enter the x-coordinate of the top-left corner of the swimming area:')
            else:
                swimarea_tlx.current = swimarea_tlx.user_set
            if swimarea_tly.user_set == -1:
                swimarea_tly.current = get_input(int, 'Enter the y-coordinate of the top-left corner of the swimming area:')
            else:
                swimarea_tly.current = swimarea_tly.user_set
            if swimarea_x.user_set == -1:
                swimarea_x.current = get_input(int, 'Enter the width (x) of the swimming area:')
            else:
                swimarea_x.current = swimarea_x.user_set
            if swimarea_y.user_set == -1:
                swimarea_y.current = get_input(int, 'Enter the height (y) of the swimming area:')
            else:
                swimarea_y.current = swimarea_y.user_set
            while check.user_set:
                frame_l = frame_t
                cv.rectangle(frame_l, (swimarea_tlx.current, swimarea_tly.current),
                             (swimarea_tlx.current + swimarea_x.current, swimarea_tly.current + swimarea_y.current), (0, 0, 255), 3)
                cv.imwrite(videoname + '_labeled_frame.png', frame_l)
                if input('The swimming area is labeled in a new figure. Enter f to carry on, others to change:') == 'f':
                    break
                swimarea_tlx.current = get_input(int, 'Enter the x-coordinate of the top-left corner of the swimming area:')
                swimarea_tly.current = get_input(int, 'Enter the y-coordinate of the top-left corner of the swimming area:')
                swimarea_x.current = get_input(int, 'Enter the width (x) of the swimming area:')
                swimarea_y.current = get_input(int, 'Enter the height (y) of the swimming area:')
            
            metadata.update({
                'filename': filename,
                'fps': fps,
                'x_original': x_original,
                'y_original': y_original,
                'video_start': video_start.current,
                'video_end': video_end.current,
                'rotate': rotate.current,
                'crop_tlx': crop_tlx.current,
                'crop_tly': crop_tly.current,
                'crop_x': crop_x.current,
                'crop_y': crop_y.current,
                'x_current': x_current,
                'y_current': y_current,
                'swimarea_tlx': swimarea_tlx.current,
                'swimarea_tly': swimarea_tly.current,
                'swimarea_x': swimarea_x.current,
                'swimarea_y': swimarea_y.current
            })
            
            with open(path + '/' + videoname + '_metadata.csv', 'w') as f:
                for key in metadata:
                    f.write(key + ',' + str(metadata[key]) + '\n')
            metadata.clear()
                
    except Exception:
        
        print('An error occurred when setting up ' + filename + ' for tracking:')
        traceback.print_exc()

def max_entropy_threshold(image, threshold_reduction):
    hist, _ = np.histogram(image.ravel(), bins = 256, range=(0, 256))
    nhist = hist / hist.sum()
    pT = [0 for ii in range(256)]
    ii = 1
    while ii <= 255:
        pT[ii] = pT[ii - 1] + nhist[ii]
        ii += 1
    hB = [0 for ii in range(256)]
    hW = [0 for ii in range(256)]
    t = 0
    while t <= 255:
        if pT[t] > 0:
            hhB = 0
            ii = 0
            while ii <= t:
                if nhist[ii] > 0:
                    temp = nhist[ii] / pT[t]
                    hhB -= temp * math.log10(temp)
                ii += 1
            hB[t] = hhB
        pTW = 1 - pT[t]
        if pTW > 0:
            hhW = 0
            ii = t + 1
            while ii <= 255:
                if nhist[ii] > 0:
                    temp = nhist[ii] / pTW
                    hhW -= temp * math.log10(temp)
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

def pyth(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def sq_area(image, point, r):
    area = 0
    for i in range(point[0] - r, point[0] + r + 1):
        for j in range(point[1] - r, point[1] + r + 1):
            if image[j][i] == 255:
                area += 1
    return area

def cal_direction(point1, point2): #from point1 to point2
    if point1[0] == point2[0] and point2[1] > point1[1]:
        return math.pi / 2
    elif point1[0] == point2[0] and point2[1] < point1[1]:
        return -math.pi / 2
    elif point1[0] == point2[0] and point1[1] == point2[1]:
        print('cal_direction_Error')
        return 0
    else:
        return math.atan2(point2[1] - point1[1], point2[0] - point1[0])

def cal_direction_change(s1, s2): #from s1 to s2
    direction_change = s2 - s1
    if direction_change > math.pi:
        return direction_change - math.pi * 2
    elif direction_change <= -math.pi:
        return direction_change + math.pi * 2
    else:
        return direction_change

start_time = time()

for filename in videofilenames:
        
    try:
        video = cv.VideoCapture(filename)
        filename_split = os.path.splitext(filename)
        videoname = filename_split[0]
        path = './' + videoname
        if not os.path.isfile(path + '/' + videoname + '_metadata.csv'):
            print('Metadata missing for ' + videoname)
            continue
        print('\nProcessing ' + filename)
    except Exception:
        print('An error occurred when opening ' + videoname + ':')
        traceback.print_exc()
        continue
    
    try:
        metadata.clear()
        with open(path + '/' + videoname + '_metadata.csv', 'r') as f:
            metadata = {row[0]: row[1] for row in csv.reader(f)}
            video_start.current = int(metadata['video_start'])
            video_end.current = int(metadata['video_end'])
            fps = float(metadata['fps'])
            x_original = int(metadata['x_original'])
            y_original = int(metadata['y_original'])
            rotate.current = float(metadata['rotate'])
            crop_tlx.current = int(metadata['crop_tlx'])
            crop_tly.current = int(metadata['crop_tly'])
            crop_x.current = int(metadata['crop_x'])
            crop_y.current = int(metadata['crop_y'])
            x_current = int(metadata['x_current'])
            y_current = int(metadata['y_current'])
            swimarea_tlx.current = int(metadata['swimarea_tlx'])
            swimarea_tly.current = int(metadata['swimarea_tly'])
            swimarea_x.current = int(metadata['swimarea_x'])
            swimarea_y.current = int(metadata['swimarea_y'])
    except Exception:
        print('An error occurred when accessing the metadata of ' + videoname + ':')
        traceback.print_exc()
        continue
    
    try:
        
        rm = get_rm(x_original, y_original, rotate.current)
        crop_brx = crop_tlx.current + crop_x.current
        crop_bry = crop_tly.current + crop_y.current
        
        l = video_end.current - video_start.current
        j = video_start.current
        video.set(cv.CAP_PROP_POS_FRAMES, j)
        
        sampling = round(sampling_time.user_set * fps)
        threshold1s = [0 for i in range(l)]
        threshold2s = [0 for i in range(l)]
        fish_areas = [0 for i in range(l)]
        leftmosts = [0 for i in range(l)]
        rightmosts = [0 for i in range(l)]
        topmosts = [0 for i in range(l)]
        bottommosts = [0 for i in range(l)]
        
        create_trackdata = True
        if os.path.exists(path + '/' + videoname + '_trackdata.csv') and os.path.exists(path + '/' + videoname + '_background.png'):
            while True:
                response = input('Skip create_trackdata? [y]/o')
                if response == 'y':        
                    with open(path + '/' + videoname + '_trackdata.csv', 'r') as f:
                        reader = csv.DictReader(f)
                        i = 0
                        for row in reader:
                            threshold1s[i] = int(row['threshold1'])
                            fish_areas[i] = float(row['fish_area'])
                            leftmosts[i] = int(row['leftmost'])
                            rightmosts[i] = int(row['rightmost'])
                            topmosts[i] = int(row['topmost'])
                            bottommosts[i] = int(row['bottommost'])
                            threshold2s[i] = int(row['threshold2'])
                            i += 1
                    background = cv.imread(path + '/' + videoname + '_background.png')
                    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
                    create_trackdata = False
                elif response != 'o':
                    print('Try again')
                    continue
                break
        
        i = 0
        if create_trackdata:
            
            while j < video_end.current:
                
                video.set(cv.CAP_PROP_POS_FRAMES, j)
                ret, frame = video.read()
                
                if ret:
                    
                    frame_t = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    frame_t = frame_rotate(frame_t, x_original, y_original, rotate.current, rm)
                    if crop_x.current != 0 and crop_y.current != 0:
                        frame_t = frame_t[crop_tly.current:crop_bry, crop_tlx.current:crop_brx]
                    
                    blurred_frame = frame_t
                    if ksize.user_set > 0:
                        blurred_frame = cv.GaussianBlur(frame_t, (ksize.user_set, ksize.user_set), 0)
                    
                    threshold1s[i] = max_entropy_threshold(blurred_frame, threshold1_reduction.user_set)
                    ret, t1frame = cv.threshold(blurred_frame, threshold1s[i], 255, cv.THRESH_BINARY_INV)
                    
                    contours, hierarchy = cv.findContours(t1frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                    contour_number = len(contours)
                    max_perimeter = 0
                    for ii in range(contour_number):
                        contour_len = len(contours[ii])
                        if contour_len > max_perimeter:
                            max_perimeter = contour_len
                            max_contour_index = ii
                            fish_contour = contours[ii]
                    leftmost = tuple(fish_contour[fish_contour[:,:,0].argmin()][0])
                    rightmost = tuple(fish_contour[fish_contour[:,:,0].argmax()][0])
                    topmost = tuple(fish_contour[fish_contour[:,:,1].argmin()][0])
                    bottommost = tuple(fish_contour[fish_contour[:,:,1].argmax()][0])
                    leftmosts[i] = leftmost[0]
                    rightmosts[i] = rightmost[0]
                    topmosts[i] = topmost[1]
                    bottommosts[i] = bottommost[1]
                    fish_areas[i] += cv.contourArea(fish_contour)
            
                else:
                
                    break
        
                print('\rSampling progress: ', i, '/', l, end = '')
                j += sampling
                i += sampling
            
            print()
        
            i = sampling
            while i < l:
                start = i - sampling
                j = start + 1
                while j < i:
                    threshold1s[j] = round((threshold1s[start] * (i - j) + threshold1s[i] * (j - start)) / sampling)
                    fish_areas[j] = round((fish_areas[start] * (i - j) + fish_areas[i] * (j - start)) / sampling)
                    j += 1
                i += sampling
            i = l - sampling
            while i < l:
                threshold1s[i] = threshold1s[l - sampling]
                fish_areas[i] = fish_areas[l - sampling]
                i += 1
        
            if spine_analysis.user_set:
                
                if not auto_bg.user_set:
                    
                    background = cv.imread(path + '/' + videoname + '_background.png')
                    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
                    
                else:
                    
                    print('Creating background...')
                    background_frame = 0
                    center_x0 = (leftmosts[0] + rightmosts[0]) / 2
                    center_y0 = (topmosts[0] + bottommosts[0]) / 2
                    thickness0 = max(rightmosts[0] - leftmosts[0], bottommosts[0] - topmosts[0]) * fish_cover_size.user_set / 2
                    half_x0 = (rightmosts[0] - leftmosts[0]) / 2 + thickness0
                    half_y0 = (bottommosts[0] - topmosts[0]) / 2 + thickness0
                    left_boundary0 = round(center_x0 - half_x0)
                    right_boundary0 = round(center_x0 + half_x0)
                    top_boundary0 = round(center_y0 - half_y0)
                    bottom_boundary0 = round(center_y0 + half_y0)
                    i = sampling
                    while i < l:
                        center_x = (leftmosts[i] + rightmosts[i]) / 2
                        center_y = (topmosts[i] + bottommosts[i]) / 2
                        thickness = max(rightmosts[i] - leftmosts[i], bottommosts[i] - topmosts[i]) * fish_cover_size.user_set / 2
                        half_x = (rightmosts[i] - leftmosts[i]) / 2 + thickness
                        half_y = (bottommosts[i] - topmosts[i]) / 2 + thickness
                        left_boundary = round(center_x - half_x)
                        right_boundary = round(center_x + half_x)
                        top_boundary = round(center_y - half_y)
                        bottom_boundary = round(center_y + half_y)
                        bool_x = (left_boundary > right_boundary0) or (right_boundary < left_boundary0)
                        bool_y = (bottom_boundary < top_boundary0) or (top_boundary > bottom_boundary0)
                        print(left_boundary, right_boundary, top_boundary, bottom_boundary,
                              left_boundary0, right_boundary0, top_boundary0, bottom_boundary0,
                              bool_x, bool_y)
                        if bool_x or bool_y:
                            background_frame = i
                            break
                        else:
                            i += sampling
                    
                    video.set(cv.CAP_PROP_POS_FRAMES, video_start.current)
                    ret, frame0 = video.read()
                    frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
                    frame0 = frame_rotate(frame0, x_original, y_original, rotate.current, rm)
                    if crop_x.current != 0 and crop_y.current != 0:
                        frame0 = frame0[crop_tly.current:crop_bry, crop_tlx.current:crop_brx]
                    background = frame0
                    video.set(cv.CAP_PROP_POS_FRAMES, video_start.current + background_frame)
                    ret, framei = video.read()
                    framei = cv.cvtColor(framei, cv.COLOR_BGR2GRAY)
                    framei = frame_rotate(framei, x_original, y_original, rotate.current, rm)
                    if crop_x.current != 0 and crop_y.current != 0:
                        framei = framei[crop_tly.current:crop_bry, crop_tlx.current:crop_brx]
                    for ii in range(top_boundary0, bottom_boundary0 + 1):
                        for jj in range(left_boundary0, right_boundary0 + 1):
                            background[ii][jj] = framei[ii][jj]
                    print('Background created with frames ' + str(video_start.current) + ' and ' + str(video_start.current + background_frame))
                    cv.imwrite(path + '/' + videoname + '_0.png', frame0)
                    cv.imwrite(path + '/' + videoname + '_i.png', framei)
                    cv.imwrite(path + '/' + videoname + '_background.png', background)
                    print(videoname + '_background.png' + ' saved.')
                
                i = 0
                j = video_start.current
                
                while j < video_end.current:
                    
                    video.set(cv.CAP_PROP_POS_FRAMES, j)
                    ret, frame = video.read()
                    
                    if ret:
                        
                        frame_t = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        frame_t = frame_rotate(frame_t, x_original, y_original, rotate.current, rm)
                        if crop_x.current != 0 and crop_y.current != 0:
                            frame_t = frame_t[crop_tly.current:crop_bry, crop_tlx.current:crop_brx]
                        
                        frame_d = 255 - cv.absdiff(frame_t, background)
                        blurred_frame = frame_d
                        if ksize.user_set > 0:
                            blurred_frame = cv.GaussianBlur(frame_d, (ksize.user_set, ksize.user_set), 0)
                        threshold2s[i] = max_entropy_threshold(blurred_frame, threshold2_reduction.user_set)
                        
                    else:
                    
                        break
            
                    print('\rSampling progress: ', i, '/', l, end = '')
                    j += sampling
                    i += sampling
                
                print()
            
                i = sampling
                while i < l:
                    start = i - sampling
                    j = start + 1
                    while j < i:
                        threshold2s[j] = round((threshold2s[start] * (i - j) + threshold2s[i] * (j - start)) / sampling)
                        j += 1
                    i += sampling
                i = l - sampling
                while i < l:
                    threshold2s[i] = threshold2s[l - sampling]
                    i += 1
        
        if save_binaryvideo.user_set:
            binary = cv.VideoWriter(path + '/' + videoname + '_t.avi', cv.VideoWriter_fourcc('F','F','V','1'), fps, (x_current, y_current), 0)
        if save_annotatedvideo.user_set:
            annotated = cv.VideoWriter(path + '/' + videoname + '_a.avi', cv.VideoWriter_fourcc('F','F','V','1'), fps, (x_current, y_current))
            
        i = 0
        cen = [() for i in range(l)]
        spine = [[] for i in range(l)]
        spine_len = [0 for i in range(l)]
        fish_perimeters = [0 for i in range(l)]
        heads = [() for i in range(l)]
        tails = [() for i in range(l)]
        directions = [0 for i in range(l)]
        turns = [0 for i in range(l)]
        amplitudes = [0 for i in range(l)]
        fish_lengths = [0 for i in range(l)]
        abnormal_frames = []
        
        video.set(cv.CAP_PROP_POS_FRAMES, video_start.current)
        i = 0
        j = video_start.current
        
        while j < video_end.current:
            
            ret, frame = video.read()
            
            if ret:
            
                frame_t = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame_t = frame_rotate(frame_t, x_original, y_original, rotate.current, rm)
                if crop_x.current != 0 and crop_y.current != 0:
                    frame_t = frame_t[crop_tly.current:crop_bry, crop_tlx.current:crop_brx]
                
                if save_annotatedvideo.user_set:
                    aframe = cv.cvtColor(frame_t, cv.COLOR_GRAY2BGR)
                
                blurred_frame = frame_t
                if ksize.user_set > 0:
                    blurred_frame = cv.GaussianBlur(frame_t, (ksize.user_set, ksize.user_set), 0)
                ret, t1frame = cv.threshold(blurred_frame, threshold1s[i], 255, cv.THRESH_BINARY_INV)
                
                contours1, hierarchy1 = cv.findContours(t1frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                contours1_number = len(contours1)
                max_perimeter1 = 0
                for ii in range(contours1_number):
                    contour1_len = len(contours1[ii])
                    if contour1_len > max_perimeter1:
                        max_perimeter1 = contour1_len
                        fish_contour1 = contours1[ii]
                        max_contour1_index = ii
                fish_perimeters[i] = max_perimeter1
                fish_perimeter = max_perimeter1
                
                s1frame = np.zeros((y_current, x_current), dtype = np.uint8)
                cv.drawContours(s1frame, contours1, max_contour1_index, 255, -1)
                if save_binaryvideo.user_set:
                    binary.write(s1frame)
                
                moment = cv.moments(fish_contour1)
                cen[i] = (moment['m10'] / moment['m00'], moment['m01'] / moment['m00'])
                
                if spine_analysis.user_set:
                    
                    tail_search_area = 99999999
                    sq_length = round(4 + fish_areas[i] // 300)
                    for ii in range(fish_perimeter):
                        current_area = sq_area(s1frame, (fish_contour1[ii][0][0], fish_contour1[ii][0][1]), sq_length)
                        if current_area < tail_search_area:
                            tail_search_area = current_area
                            tail1_index = ii
                    
                    fish_contour1_points = []             
                    ii = tail1_index
                    loop_terminator = 0
                    while loop_terminator < fish_perimeter:
                        fish_contour1_points.append((fish_contour1[ii][0][0], fish_contour1[ii][0][1]))
                        ii += 1
                        loop_terminator += 1
                        if ii >= fish_perimeter:
                            ii -= fish_perimeter
                    
                    head_arc = fish_perimeter * head_r.user_set / 100
                    start = contour_points_dist.user_set // 2
                    end = fish_perimeter - 1
                    while end - start > head_arc:
                        min_body_width = 99999999
                        ii = end
                        while ii > end - contour_points_dist.user_set and end - start > head_arc:
                            body_width = pyth(fish_contour1_points[start], fish_contour1_points[ii])
                            if body_width < min_body_width:
                                min_body_width = body_width
                                next_end = ii
                            ii -= 1
                        end = next_end
                        spine[i].append(((fish_contour1_points[start][0] + fish_contour1_points[end][0]) / 2,
                                         (fish_contour1_points[start][1] + fish_contour1_points[end][1]) / 2))
                        end -= 1
                        start += contour_points_dist.user_set
                    spine_len[i] = len(spine[i])
                    
                    head_len = max(2, spine_len[i] // 3)
                    
                    end += 1
                    start -= contour_points_dist.user_set
                    if spine[i][spine_len[i] - 1][0] == spine[i][spine_len[i] - head_len][0]:
                        x = spine[i][spine_len[i] - 1][0]
                        if spine[i][spine_len[i] - 1][1] < spine[i][spine_len[i] - head_len][1]:
                            directions[i] = -math.pi / 2
                        elif spine[i][spine_len[i] - 1][1] > spine[i][spine_len[i] - head_len][1]:
                            directions[i] = math.pi / 2
                        else:
                            print('cal_direction_Error at frame ' + str(i))
                        ii = start
                        snout_ys = []
                        while ii < end:
                            if fish_contour1_points[ii][0] - x < 1:
                                snout_ys.append(fish_contour1_points[ii][1])
                            ii += 1
                        snout_y = sum(snout_ys) / len(snout_ys)
                        heads[i] = (x, snout_y)
                    else:
                        directions[i] = math.atan2(spine[i][spine_len[i] - 1][1] - spine[i][spine_len[i] - head_len][1],
                                                   spine[i][spine_len[i] - 1][0] - spine[i][spine_len[i] - head_len][0])
                        m = (spine[i][spine_len[i] - 1][1] - spine[i][spine_len[i] - head_len][1]) / (spine[i][spine_len[i] - 1][0] - spine[i][spine_len[i] - head_len][0])
                        c = spine[i][spine_len[i] - 1][1] - m * spine[i][spine_len[i] - 1][0]
                        min_diff_from_line = 99999999
                        snout_pos = fish_perimeter // 2
                        ii = start
                        while ii < end:
                            x = fish_contour1_points[ii][0]
                            y = fish_contour1_points[ii][1]
                            diff_from_line = abs(m * x - y + c) / math.sqrt(m ** 2 + 1)
                            if diff_from_line < min_diff_from_line:
                                min_diff_from_line = diff_from_line
                                snout_pos = ii
                            ii += 1
                        heads[i] = (fish_contour1_points[snout_pos][0], fish_contour1_points[snout_pos][1])
                    
                    for ii in range(1, spine_len[i]):
                        fish_lengths[i] += pyth(spine[i][ii - 1], spine[i][ii])
                    fish_lengths[i] += pyth(spine[i][spine_len[i] - 1], heads[i])
                    
                    if i > 0:
                        turns[i] = cal_direction_change(directions[i - 1], directions[i]) * fps
                        if turns[i] > turn_max.user_set:
                            abnormal_frames.append(i)
                    
                    frame_d = 255 - cv.absdiff(frame_t, background)
                    blurred_frame = frame_d
                    if ksize.user_set > 0:
                        blurred_frame = cv.GaussianBlur(frame_d, (ksize.user_set, ksize.user_set), 0)
                    ret, t2frame = cv.threshold(blurred_frame, threshold2s[i], 255, cv.THRESH_BINARY_INV)
                    
                    tail1_direction = cal_direction(spine[i][0], spine[i][1])
                    contours2, hierarchy2 = cv.findContours(t2frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                    
                    max_perimeter2 = 0
                    for contour in contours2:
                        if len(contour) > max_perimeter2:
                            max_perimeter2 = len(contour)
                            fish_contour2 = contour
                    
                    tail2_point = ()
                    min_tail2_area = 99999999
                    for ii in range(max_perimeter2):
                        fish_contour2_point = (fish_contour2[ii][0][0], fish_contour2[ii][0][1])
                        dist_from_tail1 = pyth(fish_contour2_point, spine[i][0])
                        if dist_from_tail1 > fish_lengths[i] / 3:
                            continue
                        direction_to_tail1 = cal_direction(fish_contour2_point, spine[i][0])
                        bend_from_tail1 = abs(cal_direction_change(tail1_direction, direction_to_tail1))
                        if bend_from_tail1 > tail2_range.user_set:
                            continue
                        current_area = sq_area(t2frame, fish_contour2_point, sq_length)
                        if current_area < min_tail2_area:
                            min_tail2_area = current_area
                            tail2_point = fish_contour2_point
                    if tail2_point != ():
                        tails[i] = tail2_point
                    else:
                        tails[i] = spine[i][0]
                    
                    if spine[i][spine_len[i] - 1][0] == spine[i][spine_len[i] - head_len][0]:
                        amplitudes[i] = abs(tails[i][0] - x)
                    else:
                        amplitudes[i] = abs(m * tails[i][0] - tails[i][1] + c) / math.sqrt(m ** 2 + 1)
                    directiontoo = cal_direction(spine[i][spine_len[i] - head_len], tails[i])
                    if cal_direction_change(directions[i], directiontoo) < 0:
                        amplitudes[i] = -amplitudes[i]
                
                if save_annotatedvideo.user_set:
                    cv.circle(aframe, (int(cen[i][0]), int(cen[i][1])), 3, (0, 255, 255), -1)
                    if spine_analysis:
                        for ii in range(fish_perimeter):
                            colorn = int(ii / fish_perimeter * 255)
                            cv.circle(aframe, fish_contour1_points[ii], 1, (0, colorn, 255 - colorn), -1)
                        for ii in range(spine_len[i]):
                            colorn = int(ii / spine_len[i] * 255)
                            cv.circle(aframe, (round(spine[i][ii][0]), round(spine[i][ii][1])), 2, (colorn, 255 - colorn // 2, 255 - colorn), -1)
                        cv.circle(aframe, (round(heads[i][0]), round(heads[i][1])), 2, (255, 127, 0), -1)
                        if tails[i] != ():
                            cv.circle(aframe, tails[i], 3, (255, 0, 255), -1)
                    annotated.write(aframe)
            
            else:
                
                break
            
            print('\rProgress: ', i, '/', l, end = '')
            j += 1
            i += 1
        
        print()
        video.release()
        if save_binaryvideo.user_set:
            binary.release()
        if save_annotatedvideo.user_set:
            annotated.release()
        
        metadata.update({
            'k_size': ksize.user_set,
            'sampling_time': sampling_time.user_set,
            'threshold1_reduction': threshold1_reduction.user_set,
            'save_binaryvideo': save_binaryvideo.user_set,
            'spine_analysis': spine_analysis.user_set,
            'contour_points_dist': contour_points_dist.user_set,
            'auto_bg': auto_bg.user_set,
            'fish_cover_size': fish_cover_size.user_set,
            'threshold2_reduction': threshold2_reduction.user_set,
            'tail2_range': tail2_range.user_set,
            'head_r': head_r.user_set,
            'show_errors': show_errors.user_set,
            'turn_max': turn_max.user_set,
            'save_annotatedvideo': save_annotatedvideo.user_set
        })
        with open(path + '/' + videoname + '_metadata.csv', 'w') as f:
            for key in metadata:
                f.write(key + ',' + str(metadata[key]) + '\n')
        
        with open(path + '/' + videoname + '_trackdata.csv', 'w', newline='') as f:
            fieldnames = ['threshold1', 'fish_area', 'fish_perimeter',
                          'leftmost', 'rightmost', 'topmost', 'bottommost', 'threshold2']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(l):
                writer.writerow({'threshold1': threshold1s[i],
                                 'fish_area': fish_areas[i],
                                 'fish_perimeter': fish_perimeters[i],
                                 'leftmost': leftmosts[i],
                                 'rightmost': rightmosts[i],
                                 'topmost': topmosts[i],
                                 'bottommost': bottommosts[i],
                                 'threshold2': threshold2s[i]})
        
        with open(path + '/' + videoname + '_cen.csv', 'w') as f:
            header = ['centroidX', 'centroidY']
            for word in header:
                f.write(str(word) + ',')
            f.write('\n')
            for i in range(l):
                row = [cen[i][0], cen[i][1]]
                for cell in row:
                    f.write(str(cell) + ',')
                f.write('\n')
        
        if spine_analysis:
            
            if len(abnormal_frames) > 0:
                print(len(abnormal_frames), 'possibly abnormal frames: ', abnormal_frames)
            
            with open(path + '/' + videoname + '_spine.csv', 'w') as f:
                f.write('Number of spine points' + ', ' + 'Spine points(XY, XY, ...)' + '\n')
                for i in range(l):
                    row = [spine_len[i]]
                    for j in range(spine_len[i]):
                        row.append(spine[i][j][0])
                        row.append(spine[i][j][1])
                    for cell in row:
                        f.write(str(cell) + ',')
                    f.write('\n')
            
            with open(path + '/' + videoname + '_headtail.csv', 'w') as f:
                header = ['Head_x', 'Head_y', 'Tail_x', 'Tail_y']
                for word in header:
                    f.write(str(word) + ',')
                f.write('\n')
                for i in range(l):
                    if tails[i] == ():
                        row = [heads[i][0], heads[i][1]]
                    else:
                        row = [heads[i][0], heads[i][1], tails[i][0], tails[i][1]]
                    for cell in row:
                        f.write(str(cell) + ',')
                    f.write('\n')
            
            with open(path + '/' + videoname + '_direction.csv', 'w') as f:
                header = ['Direction', 'Turn', 'Amplitude']
                for word in header:
                    f.write(str(word) + ',')
                f.write('\n')
                for i in range(l):
                    row = [directions[i], turns[i], amplitudes[i]]
                    for cell in row:
                        f.write(str(cell) + ',')
                    f.write('\n')
                    
            with open(path + '/' + videoname + '_trackdata(fishlength).csv', 'w') as f:
                for i in range(l):
                    f.write(str(fish_lengths[i]) + ', \n')
                
        print('Tracking of ' + videoname + ' complete.')
        
    except Exception:
        
        print('An error occurred when processing ' + videoname + ':')
        traceback.print_exc()

print('Runtime: ' + str(time() - start_time))
