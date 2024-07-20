import os
import csv
import decimal
from decimal import Decimal
import math
import statistics
from statistics import stdev
import cv2 as cv
import numpy as np
import time

stime = time.time()

pi = Decimal(math.pi)

with open('settings_spine_correction.csv', 'r') as csvfile:
    settings = {row[0]: row[1] for row in csv.reader(csvfile)}
    tank_length = Decimal(settings['tank length (mm)'])
    tank_width = Decimal(settings['tank width (mm)'])
    dist_sd_cutoff = Decimal(settings['maximum acceptable standard deviation in the distances moved by the 7 points'])
    show_original_video = bool(int(settings['show original video (0/1)']))
    rinv = int(settings['length of fitting square for detecting spine inversion (pixels)'])
    sl_mf = Decimal(settings['minimum acceptable fraction of median for spine length (decimal)'])
    turn_cutoff = Decimal(settings['maximum acceptable turn (deg/s)']) * pi / 180
    angle_1_cutoff = Decimal(settings['maximum acceptable value for any one of the spine angles (deg)']) * pi / 180
    angle_t_cutoff = Decimal(settings['maximum acceptable value for total_abs_spine_angle (deg)']) * pi / 180
    show_corrected_video = bool(int(settings['show corrected video (0/1)']))
    spine_inversion = bool(int(settings['spine inversion (0/1)']))
    manual_corr = bool(int(settings['manual correction (0/1)']))

for file in os.listdir('.'):
    filename = os.fsdecode(file)
    if filename.endswith('info.csv'):
        filename_tup = filename.partition('_info.csv')
        videoname = filename_tup[0]
        with open(filename, 'r') as csvfile:
            info = {row[0]: row[1] for row in csv.reader(csvfile)}
            xl = int(info['x pixel at the left border'])
            xr = int(info['x pixel at the right border'])
            yt = int(info['y pixel at the top border'])
            yb = int(info['y pixel at the bottom border'])
            ratiox = Decimal(tank_length) / Decimal(xr - xl)
            ratioy = Decimal(tank_width) / Decimal(yb - yt)
    elif filename.endswith('com.csv'):
        with open(filename, 'r') as csvfile:
            com = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
    elif filename.endswith('spine.csv'):
        with open(filename, 'r') as csvfile:
            video = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
        l = len(video)
        with open(filename, 'r') as csvfile:
            spine = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]

for i in range(l):
    for j in range(6):
        video[i].pop(1)
        spine[i].pop(1)
    for j in range(1, 14, 2):
        video[i][j] *= ratiox
        spine[i][j] *= ratiox
        video[i][j + 1] *= ratioy
        spine[i][j + 1] *= ratioy

if show_original_video or show_corrected_video or spine_inversion:
    cap = cv.VideoCapture(videoname + '_t.avi')
    print('Loading: ' + videoname + '_t.avi')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

if show_original_video:
    output = cv.VideoWriter(videoname + '(original)_labeled.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    print('Producing ' + videoname + '(original)_labeled.avi')
    i = 0
    while cap.isOpened() == True:
        ret, frame = cap.read()
        if ret == True:
            for j in range(7):
                cv.circle(frame, (int(spine[i][j * 2 + 1] / ratiox), int(spine[i][j * 2 + 2] / ratioy)), 2, (0, j * 42, 255), -1)
            output.write(frame)
            i += 1
        else:
            break
        print('\rProgress:', i, end = '')
    output.release()
    print('\n' + videoname + '(original)_labeled.avi saved.')
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

sdist = [[0 for j in range(9)] for i in range(l)]
def pyth(x1, y1, x2, y2):
    return Decimal.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def find_sdist(start, end):
    for i in range(start, end):
        for j in range(1, 8):
            sdist[i][j] = pyth(spine[i][j * 2 - 1], spine[i][j * 2], spine[i - 1][j * 2 - 1], spine[i - 1][j * 2])
        sdist[i][0] = stdev([sdist[i][j] for j in range(1, 8)])
def find_sl(start, end):
    for i in range(start, end):
        for j in range(3, 14, 2):
            sdist[i][8] += pyth(spine[i][j], spine[i][j + 1], spine[i][j - 2], spine[i][j - 1])
find_sdist(1, l)
find_sl(0, l)

direction = [0 for i in range(l)]
turn = [0 for i in range(l)]
directionc = [0 for i in range(l)]
turnc = [0 for i in range(l)]
angle = [[0 for j in range(7)] for i in range(l)]
angle_change = [[0 for j in range(7)] for i in range(l)]
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
def cal_turn(s1, s2): #from s1 to s2
    direction_change = Decimal(s2) - Decimal(s1)
    if direction_change > pi:
        return direction_change - pi * 2
    elif direction_change <= -pi:
        return direction_change + pi * 2
    else:
        return direction_change
def find_angle(start, end):
    for i in range(start, end):
        for j in range(1, 10, 2):
            angle[i][j // 2 + 1] = cal_turn(cal_direction(spine[i][j], spine[i][j + 1], spine[i][j + 2], spine[i][j + 3]),
                                            cal_direction(spine[i][j + 2], spine[i][j + 3], spine[i][j + 4], spine[i][j + 5]))
            angle[i][0] = sum([angle[i][k] for k in range(1, 6)])
            angle[i][6] = sum([abs(angle[i][k]) for k in range(1, 6)])
for i in range(l):
    direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
for i in range(1, l):
    turn[i] = cal_turn(direction[i - 1], direction[i]) * Decimal(fps)
for i in range(1, l):
    directionc[i] = cal_direction(com[i - 1][0], com[i - 1][1], com[i][0], com[i][1])
for i in range(2, l):
    turnc[i] = cal_turn(directionc[i - 1], directionc[i]) * Decimal(fps)
find_angle(0, l)
for i in range(1, l):
    for j in range(7):
        angle_change[i][j] = (angle[i][j] - angle[i][j - 1]) * Decimal(fps)

error = [[False for j in range(9)] for i in range(l)]
abnormal_frame = []
spine_cutoff = statistics.median([sdist[i][8] for i in range(l)]) * Decimal(sl_mf)
for i in range(l):
    if sdist[i][0] > dist_sd_cutoff:
        error[i][0] = True
    if sdist[1][8] < spine_cutoff:
        error[i][8] = True
    for j in range(1, 6):
        if angle[i][j] > angle_1_cutoff:
            error[i][j] = True
    if angle[i][6] > angle_t_cutoff:
        error[i][6] = True
    if turn[i] > turn_cutoff:
        error[i][7] = True
    if sum(error[i]) > 0:
        abnormal_frame.append(i)
'''
on_fish = [7 for i in range(l)]
i = 0
while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        for j in range(7):
            if sum(frame[int(spine[i][j * 2 + 2] / ratioy)][int(spine[i][j * 2 + 1] / ratiox)]) > 0:
                on_fish[i] -= 1
        i += 1
    else:
        break
'''
print(str(len(abnormal_frame)) + ' abnormal frames detected.')

if spine_inversion:
    
    print('Correcting spine inversion...')
    inverted = []
    
    head_area = [0 for i in range(l)]
    tail_area = [0 for i in range(l)]
    i = 0
    while cap.isOpened() == True:
        ret, frame = cap.read()
        if ret == True:
            headx = int(spine[i][1] / ratiox)
            heady = int(spine[i][2] / ratioy)
            tailx = int(spine[i][13] / ratiox)
            taily = int(spine[i][14] / ratioy)
            #head_area = 0
            for j in range(headx - rinv, headx + rinv + 1):
                for k in range(heady - rinv, heady + rinv + 1):
                    head_area[i] += sum(frame[k][j])
            #tail_area = 0
            for j in range(tailx - rinv, tailx + rinv + 1):
                for k in range(taily - rinv, taily + rinv + 1):
                    tail_area[i] += sum(frame[k][j])
            if tail_area[i] < head_area[i]:
                inverted.append(i)
            print('\rProgress:', i, end = '')
            i += 1
        else:
            break
        
    print()
    invl = len(inverted)
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    for i in inverted:
        spine[i][1], spine[i][13] = spine[i][13], spine[i][1]
        spine[i][2], spine[i][14] = spine[i][14], spine[i][2]
        spine[i][3], spine[i][11] = spine[i][11], spine[i][3]
        spine[i][4], spine[i][12] = spine[i][12], spine[i][4]
        spine[i][5], spine[i][9] = spine[i][9], spine[i][5]
        spine[i][6], spine[i][10] = spine[i][10], spine[i][6]
    
    find_sdist(1, l)
    find_sl(0, l)
    for i in range(l):
        direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
    for i in range(1, l):
        turn[i] = cal_turn(direction[i - 1], direction[i]) * Decimal(fps)
    find_angle(0, l)
    for i in range(1, l):
        for j in range(7):
            angle_change[i][j] = (angle[i][j] - angle[i][j - 1]) * Decimal(fps)

    error = [[False for j in range(9)] for i in range(l)]
    abnormal_frame = []
    for i in range(l):
        if sdist[i][0] > dist_sd_cutoff:
            error[i][0] = True
        if sdist[1][8] < spine_cutoff:
            error[i][8] = True
        for j in range(1, 6):
            if angle[i][j] > angle_1_cutoff:
                error[i][j] = True
        if angle[i][6] > angle_t_cutoff:
            error[i][6] = True
        if turn[i] > turn_cutoff:
            error[i][7] = True
        if sum(error[i]) > 0:
            abnormal_frame.append(i)
    print(str(len(abnormal_frame)) + ' abnormal frames remaining. They are:')
    print(abnormal_frame)    

if show_corrected_video:
    output = cv.VideoWriter(videoname + '(corrected)_labeled.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
    print('Producing ' + videoname + '(corrected)_labeled.avi')
    i = 0
    while cap.isOpened() == True:
        ret, frame = cap.read()
        if ret == True:
            for j in range(7):
                cv.circle(frame, (int(spine[i][j * 2 + 1] / ratiox), int(spine[i][j * 2 + 2] / ratioy)), 2, (0, j * 42, 255), -1)
            output.write(frame)
            i += 1
        else:
            break
        print('\rProgress:', i, end = '')
    output.release()
    print('\n' + videoname + '(corrected)_labeled.avi saved.')
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

if manual_corr and input('Enter 0 if no further corrections are needed, others if otherwise:') != '0':
    
    frames_avg = []
    frames_inv = []
    t = 'n'
    while t != 'f':
        frame_corr = (int(input('Enter the frame number for correction:')), input('Enter a for averaging, i for spine inversion:'))
        if frame_corr[1] == 'a':
            frames_avg.append(frame_corr[0])
        elif frame_corr[1] == 'i':
            frames_inv.append(frame_corr[0])
        else:
            print('Invalid input.')
        t = input('Enter f if all frames have been entered, others to continue the input:')
    
    for i in frames_inv:
        spine[i][1], spine[i][13] = spine[i][13], spine[i][1]
        spine[i][2], spine[i][14] = spine[i][14], spine[i][2]
        spine[i][3], spine[i][11] = spine[i][11], spine[i][3]
        spine[i][4], spine[i][12] = spine[i][12], spine[i][4]
        spine[i][5], spine[i][9] = spine[i][9], spine[i][5]
        spine[i][6], spine[i][10] = spine[i][10], spine[i][6]
    
    frames_avg_l = len(frames_avg)
    i = 0
    while i < frames_avg_l:
        j = i + 1
        while j < frames_avg_l and frames_avg[j] - frames_avg[i] == j - i:
            j += 1
        for ii in range(frames_avg[i], frames_avg[j - 1] + 1):
            for jj in range(1, 15):
                spine[ii][jj] = (spine[frames_avg[i] - 1][jj] * (frames_avg[j - 1] + 1 - ii)
                                 + spine[frames_avg[j - 1] + 1][jj] * (ii - frames_avg[i] + 1)) / (frames_avg[j - 1] - frames_avg[i] + 2)
        i = j
        
    find_sdist(1, l)
    find_sl(0, l)
    for i in range(l):
        direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
    for i in range(1, l):
        turn[i] = cal_turn(direction[i - 1], direction[i]) * Decimal(fps)
    find_angle(0, l)
    for i in range(1, l):
        for j in range(7):
            angle_change[i][j] = (angle[i][j] - angle[i][j - 1]) * Decimal(fps)

    error = [[False for j in range(9)] for i in range(l)]
    abnormal_frame = []
    for i in range(l):
        if sdist[i][0] > dist_sd_cutoff:
            error[i][0] = True
        if sdist[1][8] < spine_cutoff:
            error[i][8] = True
        for j in range(1, 6):
            if angle[i][j] > angle_1_cutoff:
                error[i][j] = True
        if angle[i][6] > angle_t_cutoff:
            error[i][6] = True
        if turn[i] > turn_cutoff:
            error[i][7] = True
        if sum(error[i]) > 0:
            abnormal_frame.append(i)
    print(str(len(abnormal_frame)) + ' abnormal frames remaining. They are:')
    print(abnormal_frame)

    if show_corrected_video:
        output = cv.VideoWriter(videoname + '(corrected)_labeled.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
        print('Producing ' + videoname + '(corrected)_labeled.avi')
        i = 0
        while cap.isOpened() == True:
            ret, frame = cap.read()
            if ret == True:
                for j in range(7):
                    cv.circle(frame, (int(spine[i][j * 2 + 1] / ratiox), int(spine[i][j * 2 + 2] / ratioy)), 2, (0, j * 42, 255), -1)
                output.write(frame)
                i += 1
            else:
                break
            print('\rProgress:', i, end = '')
        output.release()
        print('\n' + videoname + '(corrected)_labeled.avi saved.')
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

if show_corrected_video or show_original_video or spine_inversion:
    cap.release()

turn_avg = [0 for i in range(l)]
headtotail_angle_avg = [0 for i in range(l)]
total_abs_angle_avg = [0 for i in range(l)]
headtotail_angle_change = [0 for i in range(l)]
total_abs_angle_change = [0 for i in range(l)]

def runavg(inputlist, start, end):
    outputlist = [0 for i in range(start)]
    outputlist.append(sum(inputlist[1:4]) / 3)
    outputlist.append(sum(inputlist[1:5]) / 4)
    outputlist.append(sum(inputlist[1:6]) / 5)
    for i in range(start + 3, end - 2):
        outputlist.append((outputlist[i - 1] * 5 + inputlist[i + 2] - inputlist[i - 3]) / 5)
    outputlist.append(sum(inputlist[(end - 4):end]) / 4)
    outputlist.append(sum(inputlist[(end - 3):end]) / 3)
    return outputlist
    
turn_avg = runavg(turn, 1, l)

headtotail_angle_avg = runavg([angle[i][0] for i in range(l)], 0, l)
total_abs_angle_avg = runavg([angle[i][6] for i in range(l)], 0, l)

for i in range(1, l):
    headtotail_angle_change[i] = (headtotail_angle_avg[i] - headtotail_angle_avg[i - 1]) * Decimal(fps)
    total_abs_angle_change[i] = (total_abs_angle_avg[i] - total_abs_angle_avg[i - 1]) * Decimal(fps)

headtotail_angle_change_avg = runavg(headtotail_angle_change, 1, l)
total_abs_angle_change_avg = runavg(total_abs_angle_change, 1, l)

with open(videoname + '_converted_spine_distances.csv', 'w') as file:
    header = ['Frame', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 
              'Distance stdev', 'Distance 1', 'Distance 2', 'Distance 3', 'Distance 4', 'Distance 5', 'Distance 6', 'Distance 7', 'Spine length']
    for word in header:
        file.write(str(word) + ', ')
    file.write('\n')
    for i in range(l):
        file.write(str(spine[i][0]) + ', ')
        for j in range(1, 15):
            file.write(str(spine[i][j]) + ', ')
        for j in range(8):
            file.write(str(sdist[i][j]) + ', ')
        file.write(str(sdist[i][8]) + '\n')
    
with open(videoname + '_converted_spine_turns_and_angles.csv', 'w') as file:
    header = ['Frame', 'Direction', 'Turn', 'Turn avg',
              'Angle 1', 'Angle 2', 'Angle 3', 'Angle 4', 'Angle 5',
              'Angle 1 change', 'Angle 2 change', 'Angle 3 change', 'Angle 4 change', 'Angle 5 change',
              'headtotail_angle', 'headtotail_angle_avg', 'headtotail_angle_change', 'headtotail_angle_change_avg',
              'total_abs_angle', 'total_abs_angle_avg', 'total_abs_angle_change', 'total_abs_angle_change_avg']
    for word in header:
        file.write(str(word) + ', ')
    file.write('\n')
    for i in range(l):
        file.write(str(spine[i][0]) + ', ' + str(direction[i]) + ', '
                   + str(turn[i]) + ', ' + str(turn_avg[i]) + ', ')
        for j in range(1, 6):
            file.write(str(angle[i][j]) + ', ')
        for j in range(1, 6):
            file.write(str(angle_change[i][j]) + ', ')
        file.write(str(angle[i][0]) + ', ' + str(headtotail_angle_avg[i]) + ', '
                   + str(headtotail_angle_change[i]) + ', ' + str(headtotail_angle_change_avg[i]) + ', '
                   + str(angle[i][6]) + ', ' + str(total_abs_angle_avg[i]) + ', '
                   + str(total_abs_angle_change[i]) + ', ' + str(total_abs_angle_change_avg[i]) + '\n')

print('Conversion complete.')
cv.destroyAllWindows()
etime = time.time()
print('Runtime: ' + str(etime - stime))
