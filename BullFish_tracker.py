import os
import csv
import cv2 as cv
import numpy as np
import math
from decimal import Decimal

pi = Decimal(math.pi)

with open('settings_bullfish.csv', 'r') as f:
    
    settings = {row[0]: row[1] for row in csv.reader(f)}
    
    spine_analysis = bool(int(settings['angle and turn analysis (0/1)']))
    
    thresholding = bool(int(settings['Thresholding by Max Entropy? (0/1)']))
    ksize = int(settings['Gaussian kernel size'])
    thresholding_sampling = int(settings['Calculate threshold once every ? frame(s)'])
    threshold_reduction = float(settings['Lower threshold by ? %'])
    contour_points_dist = int(settings['Approximate distance between spine points (pixels)'])
    head_r = float(settings['Perimeter of head is ? % of the whole fish perimeter']) / 100
    show_binary = bool(int(settings['Show thresholded video? (0/1)']))
    show_annotated = bool(int(settings['Show annotated video? (0/1)']))
    turn_max = Decimal(settings['Maximum acceptable turn (deg/s)']) * pi / 180

def pyth(x1, y1, x2, y2):
    try:
        return Decimal.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def sq_area(image, x, y, r):
    area = 0
    for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
            if image[i][j] == 255:
                area += 1
    return area

def runavg(inputlist, start, end):
    outputlist = [0 for index in range(start)]
    outputlist.append(sum(inputlist[1:4]) / 3)
    outputlist.append(sum(inputlist[1:5]) / 4)
    outputlist.append(sum(inputlist[1:6]) / 5)
    for index in range(start + 3, end - 2):
        outputlist.append((outputlist[index - 1] * 5 + inputlist[index + 2] - inputlist[index - 3]) / 5)
    outputlist.append(sum(inputlist[(end - 4):end]) / 4)
    outputlist.append(sum(inputlist[(end - 3):end]) / 3)
    return outputlist

def cal_direction(x1, y1, x2, y2): #caudal (x2, y2) to cranial (x1, y1)
    if x1 == x2 and y1 > y2:
        return pi / 2
    elif x1 == x2 and y1 < y2:
        return -pi / 2
    elif x1 == x2 and y1 == y2:
        print('cal_direction_Error')
        return Decimal(0)
    inclin = math.atan((y1 - y2) / (x1 - x2))
    if x1 > x2:
        return Decimal(inclin)
    elif x1 < x2 and y1 >= y2:
        return Decimal(inclin) + pi
    elif x1 < x2 and y1 < y2:
        return Decimal(inclin) - pi

def cal_direction_change(s1, s2): #from s1 to s2
    direction_change = Decimal(s2) - Decimal(s1)
    if direction_change > pi:
        return direction_change - pi * 2
    elif direction_change <= -pi:
        return direction_change + pi * 2
    else:
        return direction_change

for file in os.listdir('.'):
     
    filename = os.fsdecode(file)
    if filename.endswith("_p.avi"):
        
        try:
            
            gray = cv.VideoCapture(filename)
            if gray.isOpened():
                print('Processing ' + filename)
            else:
                print('Unable to open ' + filename)
                continue
            width = int(gray.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(gray.get(cv.CAP_PROP_FRAME_HEIGHT))
            fps = int(gray.get(cv.CAP_PROP_FPS))
            l = int(gray.get(cv.CAP_PROP_FRAME_COUNT))
            split_tup = os.path.splitext(filename)
            videoname = split_tup[0]
            
            path = './' + videoname
            if not os.path.exists(path):
                os.mkdir(path)
            
            if thresholding:
                
                thresholds = [0 for i in range(l)]
                
                for i in range(0, l, thresholding_sampling):
                    
                    gray.set(cv.CAP_PROP_POS_FRAMES, i)
                    ret, frame = gray.read()
                    
                    if ret:
                        
                        print('\rCalculating threshold for Frame ' + str(i), end = '')
                        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        blurred_frame = cv.GaussianBlur(gray_frame, (ksize, ksize), 0)
                        
                        hist, _ = np.histogram(blurred_frame.ravel(), bins = 256, range=(0, 256))
                        nhist = hist / Decimal(int(hist.sum()))
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
                                        hhB -= temp * temp.log10()
                                    ii += 1
                                hB[t] = hhB
                            pTW = 1 - pT[t]
                            if pTW > 0:
                                hhW = 0
                                ii = t + 1
                                while ii <= 255:
                                    if nhist[ii] > 0:
                                        temp = nhist[ii] / pTW
                                        hhW -= temp * temp.log10()
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
                        
                        thresholds[i] = tmax * (1 - threshold_reduction / 100)
                        
                for i in range(thresholding_sampling, l, thresholding_sampling):
                    for j in range(i - thresholding_sampling + 1, i):
                        thresholds[j] = (thresholds[i - thresholding_sampling] * (i - j) + thresholds[i] * (j - i + thresholding_sampling)) / thresholding_sampling
                for i in range(l - thresholding_sampling + 1, l):
                    thresholds[i] = thresholds[l - thresholding_sampling]
                for i in range(0, l):
                    thresholds[i] = round(thresholds[i])
                
                print()
            
            gray.set(cv.CAP_PROP_POS_FRAMES, 0)
            fish_area = 0
            ret, frame = gray.read()
            if ret:
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                blurred_frame = cv.GaussianBlur(gray_frame, (ksize, ksize), 0)
                ret1, tframe = cv.threshold(blurred_frame, thresholds[0], 255, cv.THRESH_BINARY_INV)
                contours, hierarchy = cv.findContours(tframe, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                max_diameter = 0
                for contour in contours:
                    if len(contour) > max_diameter:
                        max_diameter = len(contour)
                        fish_contour = contour
                sframe = np.zeros((height, width), dtype = np.uint8)
                cv.drawContours(sframe, fish_contour, -1, 255, -1)
                leftmost = tuple(fish_contour[fish_contour[:,:,0].argmin()][0])
                rightmost = tuple(fish_contour[fish_contour[:,:,0].argmax()][0])
                topmost = tuple(fish_contour[fish_contour[:,:,1].argmin()][0])
                bottommost = tuple(fish_contour[fish_contour[:,:,1].argmax()][0])
                for ii in range(topmost[1], bottommost[1] + 1):
                    for jj in range(leftmost[0], rightmost[0] + 1):
                        if cv.pointPolygonTest(fish_contour, (jj, ii), True) >= 0:
                            sframe[ii][jj] = 255
                            fish_area += 1
            cen = [[0, 0] for i in range(l)]
            spine = [[] for i in range(l)]
            spine_len = [0 for i in range(l)]
            if show_binary:
                binary = cv.VideoWriter(path + '/' + videoname + '_t.avi', cv.VideoWriter_fourcc('F','F','V','1'), fps, (width, height), 0)
            if show_annotated:
                annotated = cv.VideoWriter(path + '/' + videoname + '_a.avi', cv.VideoWriter_fourcc('F','F','V','1'), fps, (width, height))
    
            gray.set(cv.CAP_PROP_POS_FRAMES, 0)
            i = 0
            while i < l:
                
                ret, frame = gray.read()
                if ret:
                    
                    print('\rProcessing frame ' + str(i), end = '')
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    blurred_frame = cv.GaussianBlur(gray_frame, (ksize, ksize), 0)
                    ret1, tframe = cv.threshold(blurred_frame, thresholds[i], 255, cv.THRESH_BINARY_INV)
                    
                    contours, hierarchy = cv.findContours(tframe, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                    max_diameter = 0
                    for contour in contours:
                        if len(contour) > max_diameter:
                            max_diameter = len(contour)
                            fish_contour = contour
                    
                    sframe = np.zeros((height, width), dtype = np.uint8)
                    cv.drawContours(sframe, fish_contour, -1, 255, -1)
                    leftmost = tuple(fish_contour[fish_contour[:,:,0].argmin()][0])
                    rightmost = tuple(fish_contour[fish_contour[:,:,0].argmax()][0])
                    topmost = tuple(fish_contour[fish_contour[:,:,1].argmin()][0])
                    bottommost = tuple(fish_contour[fish_contour[:,:,1].argmax()][0])
                    for ii in range(topmost[1], bottommost[1] + 1):
                        for jj in range(leftmost[0], rightmost[0] + 1):
                            if cv.pointPolygonTest(fish_contour, (jj, ii), True) >= 0:
                                sframe[ii][jj] = 255
                    if show_binary:
                        binary.write(sframe)
                    
                    moment = cv.moments(fish_contour)
                    cen[i][0] = moment['m10'] / moment['m00']
                    cen[i][1] = moment['m01'] / moment['m00']
                    
                    if spine_analysis:
                    
                        tail_search_area = 99999999
                        for j in range(max_diameter):
                            current_area = sq_area(tframe, fish_contour[j][0][1], fish_contour[j][0][0], 4 + fish_area // 300)
                            if current_area < tail_search_area:
                                tail_search_area = current_area
                                tail_index = j
                        
                        fish_contour_points = []             
                        j = tail_index
                        loop_terminator = 0
                        while loop_terminator < max_diameter:
                            fish_contour_points.append([fish_contour[j][0][0], fish_contour[j][0][1]])
                            j += 1
                            loop_terminator += 1
                            if j >= max_diameter:
                                j -= max_diameter
                        
                        head_arc = max_diameter * head_r
                        start = contour_points_dist // 2
                        end = max_diameter - 1
                        while end - start > head_arc:
                            min_body_width = 99999999
                            j = end
                            while j > end - contour_points_dist and end - start > head_arc:
                                body_width = pyth(fish_contour_points[start][0], fish_contour_points[start][1],
                                                  fish_contour_points[j][0], fish_contour_points[j][1])
                                if body_width < min_body_width:
                                    min_body_width = body_width
                                    end1 = j
                                j -= 1
                            end = end1
                            spine[i].append([(fish_contour_points[start][0] + fish_contour_points[end][0]) / 2,
                                             (fish_contour_points[start][1] + fish_contour_points[end][1]) / 2])
                            end -= 1
                            start += contour_points_dist
                        spine_len[i] = len(spine[i])
                    
                    if show_annotated:
                        cv.circle(frame, (int(cen[i][0]), int(cen[i][1])), 3, (0, 255, 255), -1)
                        if spine_analysis:
                            for j in range(max_diameter):
                                colorn = int(j / max_diameter * 255)
                                cv.circle(frame, (fish_contour_points[j][0], fish_contour_points[j][1]), 1, (0, colorn, 255 - colorn), -1)
                            for j in range(spine_len[i]):
                                colorn = int(j / spine_len[i] * 255)
                                cv.circle(frame, (round(spine[i][j][0]), round(spine[i][j][1])), 2, (colorn, 255 - colorn // 2, 255 - colorn), -1)
                        annotated.write(frame)
                    
                    i += 1
                
                else:
                    break
                
            print()
            gray.release()
            if show_binary:
                binary.release()
            if show_annotated:
                annotated.release()
            
            with open(path + '/' + videoname + '_raw_cen.csv', 'w') as f:
                header = ['centroidX', 'centroidY']
                for word in header:
                    f.write(str(word) + ', ')
                f.write('\n')
                for i in range(l):
                    row = [cen[i][0], cen[i][1]]
                    for cell in row:
                        f.write(str(cell) + ', ')
                    f.write('\n')
            
            if spine_analysis:
                
                abnormal_frames = []
                direction = [0 for i in range(l)]
                for i in range(l):
                    head_len = spine_len[i] // 3
                    direction[i] = cal_direction(spine[i][spine_len[i] - head_len - 1][0], spine[i][spine_len[i] - head_len - 1][1],
                                                 spine[i][spine_len[i] - 1][0], spine[i][spine_len[i] - 1][1])
                turn = [0 for i in range(l)]
                for i in range(1, l):
                    turn[i] = cal_direction_change(direction[i - 1], direction[i]) * Decimal(fps)
                    if turn[i] > turn_max:
                        abnormal_frames.append(i)
                if len(abnormal_frames) > 0:
                    print('Abnormal frames: ', abnormal_frames)
                
                with open(path + '/' + videoname + '_raw_spine.csv', 'w') as f:
                    f.write('spine(XY, XY, ...)' + '\n')
                    for i in range(l):
                        row = []
                        for j in range(spine_len[i]):
                            row.append(spine[i][j][0])
                            row.append(spine[i][j][1])
                        for cell in row:
                            f.write(str(cell) + ', ')
                        f.write('\n')
            
            print('Tracking of ' + videoname + ' complete.')
        
        except Exception as error:
            print('An error occurred when processing ' + videoname + ': ' + str(error))