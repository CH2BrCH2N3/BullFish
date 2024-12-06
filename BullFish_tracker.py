import os
import csv
import math
import cv2 as cv
import numpy as np
from copy import deepcopy
from traceback import print_exc
from statistics import median
from BullFish_pkg.general import csvtodict, load_settings
from BullFish_pkg.math import cal_direction, cal_direction_change
from BullFish_pkg.cv_editing import get_rm, frame_grc, frame_blur

default_settings = {
    "ksize": 5,
    "t_sampling_time": 0.2,
    "threshold1_reduction": 10,
    "save_binaryvideo": "FFV1",
    "save_annotatedvideo": "MJPG",
    "spine_analysis": 1,
    "contour_points_dist": 12,
    "turn_max": 6000,
    "show_errors": 1,
    'find_s0': 1,
    "auto_bg": 1,
    "fish_cover_size": 1.5,
    "threshold2_reduction": 0,
    "tail2_range": 0.7}

print('Welcome to BullFish_tracker.')
settings = load_settings('tracker', default_settings)

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

def sq_area(image, point, r):
    return sum([sum([(1 if image[j][i] else 0) for i in range(point[0] - r, point[0] + r + 1)]) for j in range(point[1] - r, point[1] + r + 1)])

for file in os.listdir('.'):
        
    try:
        filename = os.fsdecode(file)
        filename_split = os.path.splitext(filename)
        supported_formats = {'.avi', '.mp4'}
        if filename_split[1] not in supported_formats:
            continue
        video = cv.VideoCapture(filename)
        if not video.isOpened():
            print(filename + ' cannot be opened.')
            continue
        videoname = filename_split[0]
        path = './' + videoname
        if not os.path.isfile(path + '/' + videoname + '_metadata.csv'):
            print('Metadata missing for ' + videoname)
            continue
        print('\nProcessing ' + filename)
    except Exception:
        print('An error occurred when opening ' + videoname + ':')
        print_exc()
        continue
    
    try:
        metadata = csvtodict(path + '/' + videoname + '_metadata.csv')
    except Exception:
        print('An error occurred when accessing the metadata of ' + videoname + ':')
        print_exc()
        continue
    
    try:
        
        rm = get_rm(metadata['x_original'], metadata['y_original'], metadata['rotate'])
        
        l = (metadata['video_end'] - metadata['video_start']) // metadata['downsampling']
        j = metadata['video_start']
        video.set(cv.CAP_PROP_POS_FRAMES, j)
        
        t_sampling = settings['t_sampling_time'] * metadata['fps'] * metadata['downsampling'] # calculate threshold once every how many frames
        threshold1s = [0 for i in range(l)]
        if settings['find_s0']:
            threshold2s = [0 for i in range(l)]
            leftmosts = [0 for i in range(l)]
            rightmosts = [0 for i in range(l)]
            topmosts = [0 for i in range(l)]
            bottommosts = [0 for i in range(l)]
            fish_perimeter2s = [] # impression of fish perimeter in t2
        
        create_trackdata = True
        if os.path.exists(path + '/' + videoname + '_trackdata.csv'):
            while True:
                response = input('Trackdata has already been created. Enter y to skip this step, o to overwrite:')
                if response == 'y':        
                    with open(path + '/' + videoname + '_trackdata.csv', 'r') as f:
                        reader = csv.DictReader(f)
                        i = 0
                        for row in reader:
                            threshold1s[i] = int(row['threshold1'])
                            if settings['find_s0']:
                                leftmosts[i] = int(row['leftmost'])
                                rightmosts[i] = int(row['rightmost'])
                                topmosts[i] = int(row['topmost'])
                                bottommosts[i] = int(row['bottommost'])
                                threshold2s[i] = int(row['threshold2'])
                            i += 1
                    if settings['find_s0']:
                        print('Loading background...')
                        background = cv.imread(path + '/' + videoname + '_background.png')
                        background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
                        with open(path + '/' + videoname + '_fish_perimeter2_est.txt', 'r') as f:
                            fish_perimeter2_est = float(f.read())
                    create_trackdata = False
                elif response != 'o':
                    print('Try again')
                    continue
                break
        
        if create_trackdata:
            
            i = 0
            
            while j < metadata['video_end']:
                
                video.set(cv.CAP_PROP_POS_FRAMES, j)
                ret, frame = video.read()
                
                if ret:
                    
                    frame_t = frame_grc(frame, metadata['x_original'], metadata['y_original'], metadata['rotate'], rm, metadata['crop_tlx'], metadata['crop_tly'], metadata['crop_x'], metadata['crop_y'])
                    frame_b = frame_blur(frame_t, settings['ksize'])
                    
                    threshold1s[i] = max_entropy_threshold(frame_b, settings['threshold1_reduction'])
                    ret, t1frame = cv.threshold(frame_b, threshold1s[i], 255, cv.THRESH_BINARY_INV)
                    
                    if settings['find_s0']:
                        contours, hierarchy = cv.findContours(t1frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                        contour_number = len(contours)
                        max_perimeter = 0
                        for ii in range(contour_number):
                            contour_len = len(contours[ii])
                            if contour_len > max_perimeter:
                                max_perimeter = contour_len
                                fish_contour = contours[ii]
                        leftmost = tuple(fish_contour[fish_contour[:,:,0].argmin()][0])
                        rightmost = tuple(fish_contour[fish_contour[:,:,0].argmax()][0])
                        topmost = tuple(fish_contour[fish_contour[:,:,1].argmin()][0])
                        bottommost = tuple(fish_contour[fish_contour[:,:,1].argmax()][0])
                        leftmosts[i] = leftmost[0]
                        rightmosts[i] = rightmost[0]
                        topmosts[i] = topmost[1]
                        bottommosts[i] = bottommost[1]
            
                else:
                
                    break
        
                print('\rt1_sampling progress: ', i, '/', l, end='')
                j = round(j + t_sampling)
                i = round(i + t_sampling)
            
            print()
            
            i = 1
            start = 0
            while i < l:
                if threshold1s[i] != 0:
                    j = start + 1
                    while j < i:
                        threshold1s[j] = round((threshold1s[start] * (i - j) + threshold1s[i] * (j - start)) / (i - start))
                        j += 1
                    start = i
                i += 1
            i = start
            while i < l:
                threshold1s[i] = threshold1s[start]
                i += 1
            
            if settings['find_s0']:
                
                if not settings['auto_bg']:
                    
                    print('Loading background...')
                    background = cv.imread(path + '/' + videoname + '_background.png')
                    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
                    
                else:
                    
                    print('Creating background...')
                    background_frame = 0
                    center_x0 = (leftmosts[0] + rightmosts[0]) / 2
                    center_y0 = (topmosts[0] + bottommosts[0]) / 2
                    thickness0 = max(rightmosts[0] - leftmosts[0], bottommosts[0] - topmosts[0]) * settings['fish_cover_size'] / 2
                    half_x0 = (rightmosts[0] - leftmosts[0]) / 2 + thickness0
                    half_y0 = (bottommosts[0] - topmosts[0]) / 2 + thickness0
                    left_boundary0 = round(center_x0 - half_x0)
                    right_boundary0 = round(center_x0 + half_x0)
                    top_boundary0 = round(center_y0 - half_y0)
                    bottom_boundary0 = round(center_y0 + half_y0)
                    i = round(t_sampling)
                    while i < l:
                        center_x = (leftmosts[i] + rightmosts[i]) / 2
                        center_y = (topmosts[i] + bottommosts[i]) / 2
                        thickness = max(rightmosts[i] - leftmosts[i], bottommosts[i] - topmosts[i]) * settings['fish_cover_size'] / 2
                        half_x = (rightmosts[i] - leftmosts[i]) / 2 + thickness
                        half_y = (bottommosts[i] - topmosts[i]) / 2 + thickness
                        left_boundary = round(center_x - half_x)
                        right_boundary = round(center_x + half_x)
                        top_boundary = round(center_y - half_y)
                        bottom_boundary = round(center_y + half_y)
                        bool_x = (left_boundary > right_boundary0) or (right_boundary < left_boundary0)
                        bool_y = (bottom_boundary < top_boundary0) or (top_boundary > bottom_boundary0)
                        if bool_x or bool_y:
                            background_frame = i
                            break
                        else:
                            i = round(i + t_sampling)
                    
                    video.set(cv.CAP_PROP_POS_FRAMES, metadata['video_start'])
                    ret, frame0 = video.read()
                    background = frame_grc(frame0, metadata['x_original'], metadata['y_original'], metadata['rotate'], rm, metadata['crop_tlx'], metadata['crop_tly'], metadata['crop_x'], metadata['crop_y'])
                    
                    video.set(cv.CAP_PROP_POS_FRAMES, metadata['video_start'] + background_frame)
                    ret, framei = video.read()
                    framei = frame_grc(framei, metadata['x_original'], metadata['y_original'], metadata['rotate'], rm, metadata['crop_tlx'], metadata['crop_tly'], metadata['crop_x'], metadata['crop_y'])
                    
                    for ii in range(top_boundary0, bottom_boundary0 + 1):
                        for jj in range(left_boundary0, right_boundary0 + 1):
                            background[ii][jj] = framei[ii][jj]
                    print('Background created with frames ' + str(metadata['video_start']) + ' and ' + str(metadata['video_start'] + background_frame))
                    cv.imwrite(path + '/' + videoname + '_0.png', frame0)
                    cv.imwrite(path + '/' + videoname + '_i.png', framei)
                    cv.imwrite(path + '/' + videoname + '_background.png', background)
                    print(videoname + '_background.png' + ' saved.')
                
                i = 0
                j = metadata['video_start']
                
                while j < metadata['video_end']:
                    
                    video.set(cv.CAP_PROP_POS_FRAMES, j)
                    ret, frame = video.read()
                    
                    if ret:
                        
                        frame_t = frame_grc(frame, metadata['x_original'], metadata['y_original'], metadata['rotate'], rm, metadata['crop_tlx'], metadata['crop_tly'], metadata['crop_x'], metadata['crop_y'])
                        frame_d = 255 - cv.absdiff(frame_t, background)
                        frame_db = frame_blur(frame_d, settings['ksize'])
                        threshold2s[i] = max_entropy_threshold(frame_db, settings['threshold2_reduction'])
                        
                        ret, t2frame = cv.threshold(frame_db, threshold2s[i], 255, cv.THRESH_BINARY_INV)
                        contours, hierarchy = cv.findContours(t2frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                        fish_perimeter2s.append(max([len(contour) for contour in contours]))
                        
                    else:
                    
                        break
            
                    print('\rt2_sampling progress: ', i, '/', l, end='')
                    j = round(j + t_sampling)
                    i = round(i + t_sampling)
                
                print()
            
                i = 1
                start = 0
                while i < l:
                    if threshold2s[i] != 0:
                        j = start + 1
                        while j < i:
                            threshold2s[j] = round((threshold2s[start] * (i - j) + threshold2s[i] * (j - start)) / (i - start))
                            j += 1
                        start = i
                    i += 1
                i = start
                while i < l:
                    threshold2s[i] = threshold2s[start]
                    i += 1
        
    except Exception:
        print_exc()
        
    try:
    
        if settings['save_binaryvideo']:
            c = settings['save_binaryvideo']
            binary1 = cv.VideoWriter(path + '/' + videoname + '_t1.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), metadata['fps'], (metadata['x_current'], metadata['y_current']), 0)
            if settings['find_s0']:
                binary2 = cv.VideoWriter(path + '/' + videoname + '_t2.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), metadata['fps'], (metadata['x_current'], metadata['y_current']), 0)
        if settings['save_annotatedvideo']:
            c = settings['save_annotatedvideo']
            annotated = cv.VideoWriter(path + '/' + videoname + '_a.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), metadata['fps'], (metadata['x_current'], metadata['y_current']))
            
        i = 0
        cen = [() for i in range(l)]
        spine = [[0] for i in range(l)]
        spine_len = [0 for i in range(l)]
        fish_perimeters = [0 for i in range(l)]
        heads = [() for i in range(l)]
        tails = [() for i in range(l)]
        directions = [0 for i in range(l)]
        turns = [0 for i in range(l)]
        fish_lengths = [0 for i in range(l)]
        errors = {
            'fish_not_found': [],
            'tail_not_found': [],
            'fish_too_short': [],
            'high_turn': []
            }
        
        video.set(cv.CAP_PROP_POS_FRAMES, metadata['video_start'])
        i = 0
        j = metadata['video_start']
        
        while j < metadata['video_end']:
            
            ret, frame = video.read()
            
            if ret:
            
                frame_t = frame_grc(frame, metadata['x_original'], metadata['y_original'], metadata['rotate'], rm, metadata['crop_tlx'], metadata['crop_tly'], metadata['crop_x'], metadata['crop_y'])
                
                if settings['save_annotatedvideo']:
                    aframe = cv.cvtColor(frame_t, cv.COLOR_GRAY2BGR)
                
                frame_b = frame_blur(frame_t, settings['ksize'])
                ret, t1frame = cv.threshold(frame_b, threshold1s[i], 255, cv.THRESH_BINARY_INV)
                
                contours1, hierarchy1 = cv.findContours(t1frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                contours1_number = len(contours1)
                if contours1_number == 0:
                    errors['fish_not_found'].append(i)
                    continue
                fish_perimeter1 = 0
                for ii in range(contours1_number):
                    contour1_len = len(contours1[ii])
                    if contour1_len > fish_perimeter1:
                        fish_perimeter1 = contour1_len
                        fish_contour1 = contours1[ii]
                        fish_contour1_index = ii
                fish_perimeters[i] = fish_perimeter1
                
                s1frame = np.zeros((metadata['y_current'], metadata['x_current']), dtype = np.uint8)
                cv.drawContours(s1frame, contours1, fish_contour1_index, 255, -1)
                if settings['save_binaryvideo']:
                    binary1.write(s1frame)
                
                moment = cv.moments(fish_contour1)
                cen[i] = (moment['m10'] / moment['m00'], moment['m01'] / moment['m00'])
                
                if settings['spine_analysis']:
                    
                    fish_area = cv.contourArea(fish_contour1)
                    sq_length = round(4 + fish_area // 300)
                    
                    contour1_points_areas_t = [0 for ii in range(fish_perimeter1)]
                    for ii in range(fish_perimeter1):
                        contour1_points_areas_t[ii] = sq_area(s1frame, (fish_contour1[ii][0][0], fish_contour1[ii][0][1]), sq_length)
                    tail1_index = contour1_points_areas_t.index(min(contour1_points_areas_t))
                    
                    contour1_points_areas = [0 for ii in range(fish_perimeter1)]
                    fish_contour1_points = [0 for ii in range(fish_perimeter1)]             
                    jj = tail1_index
                    ii = 0
                    while ii < fish_perimeter1:
                        contour1_points_areas[ii] = contour1_points_areas_t[jj]
                        fish_contour1_points[ii] = (fish_contour1[jj][0][0], fish_contour1[jj][0][1])
                        jj += 1
                        ii += 1
                        if jj >= fish_perimeter1:
                            jj -= fish_perimeter1
                    
                    min_head_pos = []
                    head_search = range(fish_perimeter1 // 4, fish_perimeter1 * 3 // 4)
                    head_area_cutoff = round(np.percentile(contour1_points_areas[(fish_perimeter1 // 4):(fish_perimeter1 * 3 // 4)], 20))
                    for ii in head_search:
                        if contour1_points_areas[ii] < head_area_cutoff:
                            min_head_pos.append(ii)
                    head_index = round(np.percentile(min_head_pos, 50))
                    heads[i] = fish_contour1_points[head_index]
                    
                    spine[i][0] = fish_contour1_points[0]
                    if head_index < fish_perimeter1 - head_index:
                        spine_len[i] = round(head_index / settings['contour_points_dist'])
                        smaller_arc = head_index - 1
                        larger_arc = fish_perimeter1 - head_index - 1
                        for ii in range(1, spine_len[i]):
                            current_point = fish_contour1_points[round(smaller_arc * ii / spine_len[i])]
                            cor_point = fish_contour1_points[fish_perimeter1 - round(larger_arc * ii / spine_len[i])]
                            spine[i].append(((current_point[0] + cor_point[0]) / 2, (current_point[1] + cor_point[1]) / 2))
                    else:
                        spine_len[i] = round((fish_perimeter1 - head_index) / settings['contour_points_dist'])
                        smaller_arc = fish_perimeter1 - head_index - 1
                        larger_arc = head_index - 1
                        for ii in range(1, spine_len[i]):
                            current_point = fish_contour1_points[round(fish_perimeter1 - smaller_arc * ii / spine_len[i])]
                            cor_point = fish_contour1_points[round(larger_arc * ii / spine_len[i])]
                            spine[i].append(((current_point[0] + cor_point[0]) / 2, (current_point[1] + cor_point[1]) / 2))
                    
                    directions[i] = cal_direction(spine[i][spine_len[i] - 2], spine[i][spine_len[i] - 1])
                    
                    if spine_len[i] < 3:
                        errors['fish_too_short'].append(i)
                    
                    if i > 0:
                        turns[i] = cal_direction_change(directions[i - 1], directions[i]) * metadata['fps']
                        if turns[i] > settings['turn_max']:
                            errors['high_turn'].append(i)
                    
                    if settings['find_s0']:
                        
                        midpt = spine[i][spine_len[i] // 2]
                        fish_perimeter2_est = median(fish_perimeter2s)
                        frame_d = 255 - cv.absdiff(frame_t, background)
                        frame_db = frame_blur(frame_d, settings['ksize'])
                        
                        for jj in range(threshold2s[i], 0, -1):
                            
                            ret, t2frame = cv.threshold(frame_db, jj, 255, cv.THRESH_BINARY_INV)
                            
                            contours2, hierarchy2 = cv.findContours(t2frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                            contours2_number = len(contours2)
                        
                            if contours2_number == 0:
                                
                                errors['tail_not_found'].append(i)
                                tails[i] = spine[i][0]
                                break
                        
                            else:
                                
                                fish_contour2 = contours2[0]
                                fish_contour2_index = 0
                                fish_perimeter2 = len(fish_contour2)
                                for ii in range(contours2_number):
                                    if cv.pointPolygonTest(contours2[ii], midpt, False) > 0:
                                        fish_contour2 = contours2[ii]
                                        fish_contour2_index = ii
                                        fish_perimeter2 = len(fish_contour2)
                                        break
                                
                                if fish_perimeter2 > fish_perimeter2_est * 3:
                                    continue
                                
                                s2frame = np.zeros((metadata['y_current'], metadata['x_current']), dtype = np.uint8)
                                cv.drawContours(s2frame, contours2, fish_contour2_index, 255, -1)
                                if settings['save_binaryvideo']:
                                    binary2.write(s2frame)
                            
                                direction_tail1_to_midpt = cal_direction(spine[i][0], midpt)
                                fish_contour2_copy = deepcopy(fish_contour2)
                                fish_contour2 = []
                                for ii in range(fish_perimeter2):
                                    tail2 = [fish_contour2_copy[ii][0][0], fish_contour2_copy[ii][0][1]]
                                    direction_tail2_to_midpt = cal_direction(tail2, midpt)
                                    deviation = cal_direction_change(direction_tail1_to_midpt, direction_tail2_to_midpt)
                                    if deviation < settings['tail2_range']:
                                        fish_contour2.append(tail2)
                                tail2_number = len(fish_contour2)
                                
                                tail2_area = 9999999
                                for ii in range(tail2_number):
                                    point_area = sq_area(s2frame, fish_contour2[ii], sq_length)
                                    if point_area < tail2_area:
                                        tail2_area = point_area
                                        tail2 = fish_contour2[ii]
                                tails[i] = tail2
                                
                                threshold2s[i] = jj
                                break
                
                if settings['save_annotatedvideo']:
                    cv.circle(aframe, (int(cen[i][0]), int(cen[i][1])), 3, (0, 255, 255), -1)
                    if settings['spine_analysis']:
                        for ii in range(fish_perimeter1):
                            colorn = int(ii / fish_perimeter1 * 255)
                            cv.circle(aframe, fish_contour1_points[ii], 1, (0, colorn, 255 - colorn), -1)
                        for ii in range(spine_len[i]):
                            colorn = int(ii / spine_len[i] * 255)
                            cv.circle(aframe, (round(spine[i][ii][0]), round(spine[i][ii][1])), 2, (colorn, 255 - colorn // 2, 255 - colorn), -1)
                        heads[i] = fish_contour1_points[head_index]
                        cv.circle(aframe, (round(heads[i][0]), round(heads[i][1])), 3, (255, 0, 127), -1)
                        if settings['find_s0']:
                            cv.circle(aframe, (round(tails[i][0]), round(tails[i][1])), 3, (255, 0, 255), -1)
                    annotated.write(aframe)
            
            else:
                
                break
            
            print('\rProgress: ', i, '/', l, end='')
            j += metadata['downsampling']
            i += 1
        
        print()
        video.release()
        if settings['save_binaryvideo']:
            binary1.release()
            if settings['find_s0']:
                binary2.release() 
        if settings['save_annotatedvideo']:
            annotated.release()
        
        metadata.update(settings)
        with open(path + '/' + videoname + '_metadata.csv', 'w') as f:
            for key in metadata.keys():
                f.write(key + ',' + str(metadata[key]) + '\n')
        
        with open(path + '/' + videoname + '_trackdata.csv', 'w', newline='') as f:
            if settings['find_s0']:
                fieldnames = ['threshold1', 'fish_perimeter', 'leftmost', 'rightmost', 'topmost', 'bottommost', 'threshold2']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(l):
                    writer.writerow({'threshold1': threshold1s[i],
                                     'fish_perimeter': fish_perimeters[i],
                                     'leftmost': leftmosts[i],
                                     'rightmost': rightmosts[i],
                                     'topmost': topmosts[i],
                                     'bottommost': bottommosts[i],
                                     'threshold2': threshold2s[i]})
            else:
                fieldnames = ['threshold1', 'fish_perimeter']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(l):
                    writer.writerow({'threshold1': threshold1s[i],
                                     'fish_perimeter': fish_perimeters[i]})
        
        with open(path + '/' + videoname + '_fish_perimeter2_est.txt', 'w') as f:
            f.write(str(fish_perimeter2_est))
        
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
        
        if settings['spine_analysis']:
            
            with open(path + '/' + videoname + '_errors.csv', 'w') as f:
                for key in errors.keys():
                    f.write(key + ',')
                    for item in errors[key]:
                        f.write(str(item) + ',')
                    f.write('\n')
            
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
            
            with open(path + '/' + videoname + '_sn+1s.csv', 'w') as f:
                header = ['sn+1_x', 'sn+1_y']
                for word in header:
                    f.write(str(word) + ',')
                f.write('\n')
                for i in range(l):
                    row = [heads[i][0], heads[i][1]]
                    for cell in row:
                        f.write(str(cell) + ',')
                    f.write('\n')
            
            if settings['find_s0']:
                with open(path + '/' + videoname + '_s0s.csv', 'w') as f:
                    header = ['s0_x', 's0_y']
                    for word in header:
                        f.write(str(word) + ',')
                    f.write('\n')
                    for i in range(l):
                        if tails[i] != ():
                            row = [tails[i][0], tails[i][1]]
                            for cell in row:
                                f.write(str(cell) + ',')
                        f.write('\n')
            
            with open(path + '/' + videoname + '_directions.csv', 'w') as f:
                header = ['Direction', 'Turn']
                for word in header:
                    f.write(str(word) + ',')
                f.write('\n')
                for i in range(l):
                    row = [directions[i], turns[i]]
                    for cell in row:
                        f.write(str(cell) + ',')
                    f.write('\n')
                
        print('Tracking of ' + videoname + ' complete.')
        
    except Exception:
        
        print('An error occurred when processing ' + videoname + ':')
        print_exc()
