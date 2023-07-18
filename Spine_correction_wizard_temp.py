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
    fps = int(settings['fps'])
    tank_length = Decimal(settings['tank length (mm)'])
    tank_width = Decimal(settings['tank width (mm)'])
    dist_sd_cutoff = Decimal(settings['maximum acceptable standard deviation in the distances moved by the 7 points'])
    sl_mf = Decimal(settings['minimum acceptable fraction of median for spine length (decimal)'])
    turn_cutoff = Decimal(settings['maximum acceptable turn (deg/s)']) * pi / 180
    angle_1_cutoff = Decimal(settings['maximum acceptable value for any one of the spine angles (deg)']) * pi / 180
    angle_t_cutoff = Decimal(settings['maximum acceptable value for total_abs_spine_angle (deg)']) * pi / 180
    show_video = bool(int(settings['show video (0/1)']))
    spine_inversion = bool(int(settings['spine inversion (0/1)']))
    spine_averaging = bool(int(settings['spine averaging (0/1)']))
    manual_corr = bool(int(settings['manual correction (0/1)']))

for file in os.listdir('.'):
    filename = os.fsdecode(file)
    if filename.endswith('info.csv'):
        filename_tup = filename.partition('info.csv')
        videoname = filename_tup[0]
        with open(filename, 'r') as csvfile:
            info = {row[0]: row[1] for row in csv.reader(csvfile)}
            xl = int(info['x pixel at the left border'])
            xr = int(info['x pixel at the right border'])
            yt = int(info['y pixel at the top border'])
            yb = int(info['y pixel at the bottom border'])
            ratiox = Decimal(tank_length) / Decimal(xr - xl)
            ratioy = Decimal(tank_width) / Decimal(yb - yt)
            videoname = info['video name']
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
    
inv_timepts = []
i = 1
while i < l - 1:
    if error[i][0] == True:
        j = i - 1
        while error[i][8] == True:
            j -= 1
        headdistance = pyth(spine[i][1], spine[i][2], spine[j][1], spine[j][2])
        taildistance = pyth(spine[i][13], spine[i][14], spine[j][13], spine[j][14])
        alt_headdistance = pyth(spine[i][13], spine[i][14], spine[j][1], spine[j][2])
        alt_taildistance = pyth(spine[i][1], spine[i][2], spine[j][13], spine[j][14])
        if alt_headdistance < headdistance and alt_taildistance < taildistance:
            inv_timepts.append(i)
    i += 1
inv_timepts_len = len(inv_timepts)

abnormal_frame_count = 0
for i in range(l):
    if sum(error[i]) > 0:
        abnormal_frame_count += 1
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

print(str(abnormal_frame_count) + ' abnormal frames detected.')
print(str(inv_timepts_len) + ' points of spine inversion detected.')

'''
with open('error_report.csv', 'w') as file:
    file.write('Number of non-fish recognitions' + ', '
               + 'Number of frames with abnormally large spine angles (1-5)' + ', ' + '' + ', ' + '' + ', ' + '' + ', ' + '' + ', '
               + 'Number of fish inversions' + ', '
               + 'Number of frames with some other abnormalities' + '\n')
    file.write(str(len(wrong_target)) + ', ' + str(len(wrong_angle[1])) + ', ' + str(len(wrong_angle[2])) + ', '
               + str(len(wrong_angle[3])) + ', ' + str(len(wrong_angle[4])) + ', ' + str(len(wrong_angle[5])) + ', '
               + str(inv_timepts_len) + ', ' + str(abnormal_spine_count - inv_timepts_len) + '\n')
'''

if spine_inversion:
    
    cap = cv.VideoCapture(videoname)
    print('Loading: ' + videoname)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    if cap.isOpened() == True:
        ret, frame = cap.read()
        if ret == True:
            for i in range(7):
                cv.circle(frame, (int(spine[0][i * 2 + 1] / ratiox), int(spine[0][i * 2 + 2] / ratioy)), 2, (0, i * 42, 255), -1)
            cv.imwrite('First frame.png', frame)
            print('The first frame is labelled with the 7 points of the spine and saved as First frame.png')
    cap.set(cv.CAP_PROP_POS_MSEC, 0)
    first_frame_reversed = input('Enter 0 if the dots go from red on the head to yellow on the tail, others if reverse:')
    if first_frame_reversed == '0':
        first_frame_reversed = False
    else:
        first_frame_reversed = True
        inv_timepts.append(0)

    if inv_timepts_len % 2 == 1:
        inv_timepts.append(l)
    inv_timepts_len = len(inv_timepts)
        
    i = 0
    while i < inv_timepts_len:
        j = inv_timepts[i]
        while j < inv_timepts[i + 1]:
            spine[j][1], spine[j][13] = spine[j][13], spine[j][1]
            spine[j][2], spine[j][14] = spine[j][14], spine[j][2]
            spine[j][3], spine[j][11] = spine[j][11], spine[j][3]
            spine[j][4], spine[j][12] = spine[j][12], spine[j][4]
            spine[j][5], spine[j][9] = spine[j][9], spine[j][5]
            spine[j][6], spine[j][10] = spine[j][10], spine[j][6]
            j += 1
        if j >= l:
            break
        i += 2
        
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
    
if spine_averaging:
    
    i = 2
    while i < l - 1:
        if sum(error[i]) > 0:
            j = i + 1
            while j < l and sum(error[j]) > 0:
                j += 1
            for k in range(i, j):
                for ii in range(1, 15):
                    spine[k][ii] = (spine[i - 1][ii] * (j - k) + spine[j][ii] * (k - i + 1)) / (j - i + 1)
            if j >= l:
                break
            i = j
        i += 1
        
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

if spine_inversion or spine_averaging:
    abnormal_frame_count = 0
    for i in range(l):
        if sum(error[i]) > 0:
            abnormal_frame_count += 1
    print(str(abnormal_frame_count) + ' abnormal frames remaining.')

if show_video:
        
    output = cv.VideoWriter("output_video.avi", cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

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
    
    output.release()
    
    print('The output video is now complete with all spine points adjusted.')

if manual_corr:
    
    spine_inversion = input('Enter 0 if the points appear ok throughout the video, others if spine inversion occurred.')
    if spine_inversion != '0':
        print('Enter the frame in which spine inversion first occurs. Then enter the frame in which spine inversion is reversed.')
        print('Repeat until all spine inversion points have been entered.')
        print('If spine inversion remains until the end, enter the last frame as the end of spine inversion.')
        print('Enter f after you finished.')
        print('Note that the frame at 0:00.00 is #0, the frame at 0:01.00 is #100, so on and so forth.')
        inv_timepts = []
        t = 0
        while t != 'f':
            t = input()
            if t == 'f':
                break
            inv_timepts.append(int(t))
        
        inv_timepts_len = len(inv_timepts)
            
        i = 0
        while i < inv_timepts_len:
            j = inv_timepts[i]
            while j < inv_timepts[i + 1]:
                spine[j][1], spine[j][13] = spine[j][13], spine[j][1]
                spine[j][2], spine[j][14] = spine[j][14], spine[j][2]
                spine[j][3], spine[j][11] = spine[j][11], spine[j][3]
                spine[j][4], spine[j][12] = spine[j][12], spine[j][4]
                spine[j][5], spine[j][9] = spine[j][9], spine[j][5]
                spine[j][6], spine[j][10] = spine[j][10], spine[j][6]
                j += 1
            if j >= l:
                break
            i += 2
            
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
        
        output = cv.VideoWriter("output_video.avi", cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

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
        
        output.release()
        
        print('The output video is now complete with all spine points adjusted.')

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
etime = time.time()
print('Runtime: ' + str(etime - stime))