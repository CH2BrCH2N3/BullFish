from pathlib import Path
import csv
import pandas as pd
import cv2 as cv
import numpy as np
from statistics import median
from traceback import print_exc
from BullFish_pkg.general import create_path, load_settings
from BullFish_pkg.math import pyth, cal_direction, cal_direction_change
from BullFish_pkg.cv_editing import get_rm, frame_grc, frame_blur, max_entropy_threshold, find_com, sq_area

default_settings = {
    "t3": 0,
    "auto_bg": 1,
    "threshold3_reduction": 0,
    "ksize": 5,
    "t_sampling_time": 0.2,
    "threshold1_reduction": 10,
    "save_edittedvideo": 0,
    "save_binaryvideo": "FFV1",
    "save_annotatedvideo": "MJPG",
    "spine_analysis": 1,
    "spine_points": 10,
    "find_s0": 0,
    "fish_cover_size": 1.5,
    "threshold2_reduction": 0,
    "s0_range": 1}
settings = load_settings('tracker', default_settings)

metadata_all = pd.read_csv('metadata.csv')
metadata_track_all = []

for _, metadata in metadata_all.iterrows():
    
    try:
        filepath = metadata['filepath']
        p = Path(filepath)
        video = cv.VideoCapture(filepath)
        name = metadata['name']
        subpath = f'{p.parent}/{name}'
        create_path(subpath)
        video_start = metadata['video_start']
        video_end = metadata['video_end']
        fps = metadata['fps']
        x_original = metadata['x_original']
        y_original = metadata['y_original']
        rotate = metadata['rotate']
        rm = get_rm(x_original, y_original, rotate)
        crop_tlx = metadata['crop_tlx']
        crop_tly = metadata['crop_tly']
        crop_x = metadata['crop_x']
        crop_y = metadata['crop_y']
        x_current = metadata['x_current']
        y_current = metadata['y_current']
        l = video_end - video_start
        t_sampling = settings['t_sampling_time'] * fps # calculate threshold once every how many frames
    except Exception:
        print(f'An error occurred when accessing the metadata of {name}:')
        print_exc()
        continue
    
    cen_start = (0, 0)
    threshold1s = np.zeros(l, dtype=np.int32)
    fish_perimeter1s = []
    fish_area1s = []
    threshold2s = np.zeros(l, dtype=np.int32)
    leftmosts = [0 for i in range(l)]
    rightmosts = [0 for i in range(l)]
    topmosts = [0 for i in range(l)]
    bottommosts = [0 for i in range(l)]
    fish_perimeter2s = []
    
    if not settings['t3']:
        
        i = 0
        j = video_start
            
        while j < video_end:
            
            video.set(cv.CAP_PROP_POS_FRAMES, j)
            ret, frame = video.read()
            
            try:
                
                frame_edited = frame_grc(frame, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
                frame_b = frame_blur(frame_edited, settings['ksize'])
                
                threshold1s[i] = max_entropy_threshold(frame_b, settings['threshold1_reduction'])
                ret, frame_t = cv.threshold(frame_b, threshold1s[i], 255, cv.THRESH_BINARY_INV)
                contours, hierarchy = cv.findContours(frame_t, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                contour_number = len(contours)
                max_perimeter = 0
                for ii in range(contour_number):
                    contour_len = len(contours[ii])
                    if contour_len > max_perimeter:
                        max_perimeter = contour_len
                        fish_contour = contours[ii]
                fish_perimeter1s.append(max_perimeter)
                fish_area1s.append(cv.contourArea(fish_contour))
                if i == 0:
                    cen_start = find_com(fish_contour, 'a')
                
                if settings['find_s0']:
                    leftmost = tuple(fish_contour[fish_contour[:,:,0].argmin()][0])
                    rightmost = tuple(fish_contour[fish_contour[:,:,0].argmax()][0])
                    topmost = tuple(fish_contour[fish_contour[:,:,1].argmin()][0])
                    bottommost = tuple(fish_contour[fish_contour[:,:,1].argmax()][0])
                    leftmosts[i] = leftmost[0]
                    rightmosts[i] = rightmost[0]
                    topmosts[i] = topmost[1]
                    bottommosts[i] = bottommost[1]
        
            except Exception:
            
                print(f'\nAn error occurred when calculating the threshold at Frame {j} of {p.name}:')
                if i == 0:
                    threshold1s[i] = 90
                else:
                    threshold1s[i] = threshold1s[i - t_sampling]
                print_exc()
    
            print(f'\rt1_sampling progress: {i}/{l}', end='')
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
        
    if settings['auto_bg'] and settings['find_s0']:
        
        try:
            
            print('Creating background...')
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
            background_frame = i
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
            
            video.set(cv.CAP_PROP_POS_FRAMES, video_start)
            ret, frame0 = video.read()
            background = frame_grc(frame0, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
            
            video.set(cv.CAP_PROP_POS_FRAMES, video_start + background_frame)
            ret, framei = video.read()
            framei = frame_grc(framei, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
            
            for ii in range(top_boundary0, bottom_boundary0 + 1):
                for jj in range(left_boundary0, right_boundary0 + 1):
                    try:
                        background[ii][jj] = framei[ii][jj]
                    except:
                        pass
            print(f'Background created with frames {video_start} and {video_start + background_frame}')
            cv.imwrite(f'{subpath}/{name}_0.png', frame0)
            cv.imwrite(f'{subpath}/{name}_i.png', framei)
            cv.imwrite(f'{subpath}/{name}_background.png', background)
            print(f'{name}_background.png saved.')
        
        except Exception:
            
            print(f'\nAn error occurred when creating background for {name}:')
            print_exc()
            continue
    
    elif settings['auto_bg'] and settings['t3']:
        
        print(f'Creating background of {name}...')
        try:
            frames = []
            i = 0
            j = video_start
            while j < video_end:
                video.set(cv.CAP_PROP_POS_FRAMES, j)
                ret, frame = video.read()
                frame_edited = frame_grc(frame, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
                frames.append(frame_edited)
                j = round(j + settings['auto_bg'] * fps)
                i = round(i + settings['auto_bg'] * fps)
            background = np.median(frames, axis=0).astype(dtype=np.uint8)
            cv.imwrite(f'{subpath}/{name}_background.png', background)
        except Exception:
            print(f'An error occurred when creating the background of {name}:')
            print_exc()
            continue
    
    elif (settings['find_s0'] or settings['t3']) and not settings['auto_bg']:
        
        try:
            print('Loading background...')
            background = cv.imread(f'{subpath}/{name}_background.png')
            background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
        except Exception:
            print(f'An error occurred when accessing the background of {name}:')
            print_exc()
            continue
    
    if settings['find_s0'] and not settings['t3']:
        i = 0
        j = video_start
        while j < video_end:
            video.set(cv.CAP_PROP_POS_FRAMES, j)
            ret, frame = video.read()
            if ret:
                try:
                    frame_edited = frame_grc(frame, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
                    frame_d = 255 - cv.absdiff(frame_edited, background)
                    frame_db = frame_blur(frame_d, settings['ksize'])
                    threshold2s[i] = max_entropy_threshold(frame_db, settings['threshold2_reduction'])
                    ret, frame_t = cv.threshold(frame_db, threshold2s[i], 255, cv.THRESH_BINARY_INV)
                    contours, hierarchy = cv.findContours(frame_t, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                    fish_perimeter2s.append(max([len(contour) for contour in contours]))
                except Exception:
                    print(f'\nAn error occurred when calculating the threshold at Frame {j} of {p.name}:')
                    if i == 0:
                        threshold2s[i] = 10
                    else:
                        threshold2s[i] = threshold2s[i - t_sampling]
                    print_exc()
            else:
                break
            print(f'\rt2_sampling progress: {i}/{l}', end='')
            j = round(j + t_sampling)
            i = round(i + t_sampling)
        print()
    elif settings['t3']:
        i = 0
        j = video_start
        while j < video_end:
            video.set(cv.CAP_PROP_POS_FRAMES, j)
            ret, frame = video.read()
            if ret:
                try:
                    frame_edited = frame_grc(frame, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
                    frame_d = 255 - cv.absdiff(frame_edited, background)
                    frame_db = frame_blur(frame_d, settings['ksize'])
                    threshold2s[i] = max_entropy_threshold(frame_db, settings['threshold3_reduction'])
                    ret, frame_t = cv.threshold(frame_db, threshold2s[i], 255, cv.THRESH_BINARY_INV)
                    contours, hierarchy = cv.findContours(frame_t, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                    contour_number = len(contours)
                    max_perimeter = 0
                    for ii in range(contour_number):
                        contour_len = len(contours[ii])
                        if contour_len > max_perimeter:
                            max_perimeter = contour_len
                            fish_contour = contours[ii]
                    fish_perimeter1s.append(max_perimeter)
                    fish_area1s.append(cv.contourArea(fish_contour))
                    if i == 0:
                        cen_start = find_com(fish_contour, 'a')
                except Exception:
                    print(f'\nAn error occurred when calculating the threshold at Frame {j} of {p.name}:')
                    if i == 0:
                        threshold2s[i] = 10
                    else:
                        threshold2s[i] = threshold2s[i - t_sampling]
                    print_exc()
            else:
                break
            print(f'\rt3_sampling progress: {i}/{l}', end='')
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
    
    try:
        
        if settings['save_edittedvideo']:
            c = settings['save_edittedvideo']
            edited = cv.VideoWriter(f'{subpath}/{name}_n.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), fps, (x_current, y_current), 0)
        if settings['save_binaryvideo']:
            c = settings['save_binaryvideo']
            binary1 = cv.VideoWriter(f'{subpath}/{name}_t1.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), fps, (x_current, y_current), 0)
            if settings['find_s0']:
                binary2 = cv.VideoWriter(f'{subpath}/{name}_t2.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), fps, (x_current, y_current), 0)
        if settings['save_annotatedvideo']:
            c = settings['save_annotatedvideo']
            annotated = cv.VideoWriter(f'{subpath}/{name}_a.avi', cv.VideoWriter_fourcc(c[0], c[1], c[2], c[3]), fps, (x_current, y_current))
    
    except Exception:
        
        print(f'An error occurred when initializing output videos for {name}:')
        print_exc()
        continue
    
    cens = np.zeros((l, 2))
    fish_perimeter1_est = median(fish_perimeter1s)
    fish_area1_est = median(fish_area1s)
    spine_len = settings['spine_points']
    spines = np.zeros((l, spine_len, 2))
    fish_perimeters = np.zeros(l, dtype=np.int32)
    heads = np.zeros((l, 2), dtype=np.int32)
    s0s = np.zeros((l, 2), dtype=np.int32)
    errors = {
        'fish_not_found': [],
        's0_not_found': []}
        
    video.set(cv.CAP_PROP_POS_FRAMES, video_start)
    i = 0
    j = video_start
    
    while j < video_end:
        
        ret, frame = video.read()
        
        if ret:
            
            try:
                
                frame_edited = frame_grc(frame, x_original, y_original, rotate, rm, crop_tlx, crop_tly, crop_x, crop_y)
                if settings['save_edittedvideo']:
                    edited.write(frame_edited)
                if settings['save_annotatedvideo']:
                    aframe = cv.cvtColor(frame_edited, cv.COLOR_GRAY2BGR)
                
                if not settings['t3']:
                    frame_b = frame_blur(frame_edited, settings['ksize']) # Blurring and thresholding without background subtraction
                    ret, t1frame = cv.threshold(frame_b, threshold1s[i], 255, cv.THRESH_BINARY_INV)
                else:
                    frame_d = 255 - cv.absdiff(frame_edited, background)
                    frame_db = frame_blur(frame_d, settings['ksize'])
                    ret, t1frame = cv.threshold(frame_db, threshold2s[i], 255, cv.THRESH_BINARY_INV)
                
                contour1s, hierarchy1 = cv.findContours(t1frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # Identify fish contour
                candidates = []
                for ii, contour in enumerate(contour1s):
                    perimeter = len(contour)
                    area = cv.contourArea(contour)
                    if area > fish_area1_est / 10 and area < fish_area1_est * 3:
                        if perimeter > max(fish_perimeter1_est / 10, spine_len * 2) and perimeter < fish_perimeter1_est * 3:
                            candidates.append(ii)
                candidates_count = len(candidates)
                if candidates_count == 0:
                    errors['fish_not_found'].append(i)
                    continue
                min_dist = 999999
                for candidate in candidates:
                    com = find_com(contour1s[candidate], 'a')
                    dist = pyth(cen_start, com)
                    if dist < min_dist:
                        min_dist = dist
                        cen = com
                        fish_contour1_index = candidate
                fish_contour1 = contour1s[fish_contour1_index]
                fish_perimeter1 = len(fish_contour1)
                fish_perimeters[i] = fish_perimeter1
                cen_start = cen
                cens[i, 0] = cen[0]
                cens[i, 1] = cen[1]
                
                s1frame = np.zeros((y_current, x_current), dtype=np.uint8)
                cv.drawContours(s1frame, contour1s, fish_contour1_index, 255, -1)
                if settings['save_binaryvideo']:
                    binary1.write(s1frame)
                
                if settings['spine_analysis']:
                    
                    sq_length = round(4 + cv.contourArea(fish_contour1) / 300)
                    s1_index = 0
                    min_fish_area = 99999999
                    
                    contour1_points_sq = np.zeros(fish_perimeter1, dtype=np.int32)
                    for ii in range(fish_perimeter1):
                        point = (fish_contour1[ii, 0, 0], fish_contour1[ii, 0, 1])
                        contour1_points_sq[ii] = sq_area(s1frame, point, sq_length)
                        if contour1_points_sq[ii] < min_fish_area:
                            s1_index = ii
                            min_fish_area = contour1_points_sq[ii]
                    
                    fish_contour1_points = np.zeros((fish_perimeter1, 2), dtype=np.int32)    
                    jj = s1_index
                    ii = 0
                    while ii < fish_perimeter1:
                        fish_contour1_points[ii, 0] = fish_contour1[jj, 0, 0]
                        fish_contour1_points[ii, 1] = fish_contour1[jj, 0, 1]
                        jj += 1
                        ii += 1
                        if jj >= fish_perimeter1:
                            jj -= fish_perimeter1
                    
                    min_head_pos = []
                    head_areas = []
                    for ii in range(fish_perimeter1 // 4, fish_perimeter1 * 3 // 4, 2):
                        head_areas.append(sq_area(s1frame, (fish_contour1_points[ii, 0], fish_contour1_points[ii, 1]), sq_length))
                    head_area_cutoff = round(np.percentile(head_areas, 20))
                    for ii in range(len(head_areas)):
                        if head_areas[ii] <= head_area_cutoff:
                            min_head_pos.append(ii)
                    head_index = fish_perimeter1 // 4 + round(np.percentile(min_head_pos, 50)) * 2
                    heads[i, 0] = fish_contour1_points[head_index, 0]
                    heads[i, 1] = fish_contour1_points[head_index, 1]
                    
                    spines[i, 0, 0] = fish_contour1_points[0, 0]
                    spines[i, 0, 1] = fish_contour1_points[0, 1]
                    if head_index < fish_perimeter1 - head_index:
                        smaller_arc = head_index - 1
                        larger_arc = fish_perimeter1 - head_index - 1
                        for ii in range(1, spine_len):
                            current_pos = round(smaller_arc * ii / spine_len)
                            cor_pos = fish_perimeter1 - round(larger_arc * ii / spine_len)
                            spines[i, ii, 0] = (fish_contour1_points[current_pos, 0] + fish_contour1_points[cor_pos, 0]) / 2
                            spines[i, ii, 1] = (fish_contour1_points[current_pos, 1] + fish_contour1_points[cor_pos, 1]) / 2
                    else:
                        smaller_arc = fish_perimeter1 - head_index - 1
                        larger_arc = head_index - 1
                        for ii in range(1, spine_len):
                            current_pos = round(fish_perimeter1 - smaller_arc * ii / spine_len)
                            cor_pos = round(larger_arc * ii / spine_len)
                            spines[i, ii, 0] = (fish_contour1_points[current_pos, 0] + fish_contour1_points[cor_pos, 0]) / 2
                            spines[i, ii, 1] = (fish_contour1_points[current_pos, 1] + fish_contour1_points[cor_pos, 1]) / 2
                    
                    if settings['find_s0']:
                        
                        midpos = spine_len // 2
                        fish_perimeter2_est = median(fish_perimeter2s)
                        frame_d = 255 - cv.absdiff(frame_edited, background)
                        frame_db = frame_blur(frame_d, settings['ksize'])
                        
                        for jj in range(threshold2s[i], 0, -1):
                            
                            ret, t2frame = cv.threshold(frame_db, jj, 255, cv.THRESH_BINARY_INV)
                            
                            contours2, hierarchy2 = cv.findContours(t2frame, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                            contours2_number = len(contours2)
                        
                            if contours2_number == 0:
                                
                                errors['s0_not_found'].append(i)
                                s0s[i, 0] = spines[i, 0, 0]
                                s0s[i, 1] = spines[i, 0, 1]
                                break
                        
                            else:
                                
                                fish_contour2 = contours2[0]
                                fish_contour2_index = 0
                                fish_perimeter2 = len(fish_contour2)
                                for ii in range(contours2_number):
                                    if cv.pointPolygonTest(contours2[ii], (spines[i, midpos, 0], spines[i, midpos, 1]), False) > 0:
                                        fish_contour2 = contours2[ii]
                                        fish_contour2_index = ii
                                        fish_perimeter2 = len(fish_contour2)
                                        break
                                
                                if fish_perimeter2 > fish_perimeter2_est * 3:
                                    continue
                                
                                s2frame = np.zeros((y_current, x_current), dtype=np.uint8)
                                cv.drawContours(s2frame, contours2, fish_contour2_index, 255, -1)
                                if settings['save_binaryvideo']:
                                    binary2.write(s2frame)
                            
                                dir_s1_to_s2 = cal_direction((spines[i, 0, 0], spines[i, 0, 1]), (spines[i, 1, 0], spines[i, 1, 1]))
                                s0_choices = []
                                for ii in range(fish_perimeter2):
                                    s0 = (fish_contour2[ii, 0, 0], fish_contour2[ii, 0, 1])
                                    dir_s0_to_s1 = cal_direction(s0, (spines[i, 0, 0], spines[i, 0, 1]))
                                    deviation = abs(cal_direction_change(dir_s0_to_s1, dir_s1_to_s2))
                                    if deviation < settings['s0_range']:
                                        s0_choices.append(s0)
                                s0_number = len(s0_choices)
                                
                                if s0_number == 0:
                                    s0s[i, 0] = spines[i, 0, 0]
                                    s0s[i, 1] = spines[i, 0, 1]
                                else:
                                    s0_area = 9999999
                                    for ii in range(s0_number):
                                        point_area = sq_area(s2frame, s0_choices[ii], sq_length)
                                        if point_area < s0_area:
                                            s0_area = point_area
                                            s0 = s0_choices[ii]
                                    s0s[i, 0] = s0[0]
                                    s0s[i, 1] = s0[1]
                                
                                threshold2s[i] = jj
                                break
                
                if settings['save_annotatedvideo']:
                    cv.circle(aframe, (int(cens[i, 0]), int(cens[i, 1])), 3, (0, 255, 255), -1)
                    if settings['spine_analysis']:
                        for ii in range(fish_perimeter1):
                            colorn = int(ii / fish_perimeter1 * 255)
                            cv.circle(aframe, (round(fish_contour1_points[ii, 0]), round(fish_contour1_points[ii, 1])), 1, (0, colorn, 255 - colorn), -1)
                        for ii in range(spine_len):
                            colorn = int(ii / spine_len * 255)
                            cv.circle(aframe, (round(spines[i, ii, 0]), round(spines[i, ii, 1])), 2, (colorn, 255 - colorn // 2, 255 - colorn), -1)
                        heads[i] = fish_contour1_points[head_index]
                        cv.circle(aframe, (round(heads[i, 0]), round(heads[i, 1])), 3, (255, 0, 127), -1)
                        if settings['find_s0']:
                            cv.circle(aframe, (round(s0s[i, 0]), round(s0s[i, 1])), 3, (255, 0, 255), -1)
                    annotated.write(aframe)
            
            except Exception:
                
                print(f'\nAn error occurred when tracking at Frame {j} of {p.name}:')
                print_exc()
                continue
            
        else:
            
            break
        
        print(f'\rProgress: {i}/{l}', end='')
        j += 1
        i += 1
    
    print()
    video.release()
    if settings['save_edittedvideo']:
        edited.release()
    if settings['save_binaryvideo']:
        binary1.release()
        if settings['find_s0']:
            binary2.release() 
    if settings['save_annotatedvideo']:
        annotated.release()
    
    metadata_track = {
        'filepath': filepath,
        'name': name}
    metadata_track.update(settings)
    metadata_track_all.append(metadata_track)
    
    with open(f'{subpath}/{name}_trackdata.csv', 'w', newline='') as f:
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
    
    with open(f'{subpath}/{name}_cen.csv', 'w') as f:
        f.write('centroidX,centroidY\n')
        for i in range(l):
            f.write(f'{cens[i, 0]},{cens[i, 1]}\n')
    
    if settings['spine_analysis']:
        
        with open(f'{subpath}/{name}_errors.csv', 'w') as f:
            for key in errors.keys():
                f.write(key + ',')
                for item in errors[key]:
                    f.write(str(item) + ',')
                f.write('\n')
        
        with open(f'{subpath}/{name}_spine.csv', 'w') as f:
            f.write('Spine points(XY\, XY\, ...)' + '\n')
            for i in range(l):
                row = []
                for j in range(spine_len):
                    row.append(spines[i, j, 0])
                    row.append(spines[i, j, 1])
                for cell in row:
                    f.write(str(cell) + ',')
                f.write('\n')
        
        with open(f'{subpath}/{name}_sn+1s.csv', 'w') as f:
            f.write('sn+1_x,sn+1_y\n')
            for i in range(l):
                f.write(f'{heads[i, 0]},{heads[i, 1]}\n')
        
        if settings['find_s0']:
            with open(f'{subpath}/{name}_s0s.csv', 'w') as f:
                f.write('s0_x,s0_y\n')
                for i in range(l):
                    f.write(f'{s0s[i, 0]},{s0s[i, 1]}\n')
        
    print(f'Tracking of {name} complete.')

metadata_track_all = pd.DataFrame(metadata_track_all)
metadata_track_all.to_csv('metadata_track.csv')