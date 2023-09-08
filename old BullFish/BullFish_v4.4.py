import csv
import decimal
from decimal import Decimal
import math
import statistics
from statistics import stdev
import cv2
import numpy as np
import time

stime = time.time()

def pyth(x1, y1, x2, y2):
    return Decimal.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

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

with open('settings.csv', 'r') as csvfile:
    settings = [[int(number) for number in row] for row in csv.reader(csvfile)]
    fps = settings[0][0]
    tank_length = settings[0][1]
    tank_width = settings[0][2]
    spine_analysis = bool(settings[0][3])
    spine_inversion = bool(settings[0][4])
    spine_othercorrections = bool(settings[0][5])
    show_video = bool(settings[0][6])

with open('info.csv', 'r') as csvfile: 
    info = [[int(number) for number in row] for row in csv.reader(csvfile)]
    xl = info[0][0] 
    xr = info[0][1]
    yt = info[0][2]
    yb = info[0][3]
    ratiox = Decimal(tank_length) / Decimal(xr - xl)
    ratioy = Decimal(tank_width) / Decimal(yb - yt)

with open('com.csv', 'r') as csvfile:
    com = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]

l = len(com)
for i in range(l):
    com[i][0] *= ratiox
    com[i][1] *= ratioy
'''
xl *= ratiox
xr *= ratiox
yt *= ratioy
yb *= ratioy
distancetowall = [(min([(com[i][0] - xl) / (xr - xl), (xr - com[i][0]) / (xr - xl),
                  (com[i][1] - yt) / (yb - yt), (yb - com[i][1]) / (yb - yt)]) * 2) for i in range(l)]
'''
cdist = [0]
speed = [0]
for i in range(1, l):
    cdist.append(pyth(com[i][0], com[i][1], com[i - 1][0], com[i - 1][1]))
    speed.append(cdist[i] * fps)

speed_avg = runavg(speed, 1, l)

freeze = [0 for i in range(l)]
for i in range(fps * 3, l):
    cdist1 = pyth(com[i - fps * 2][0], com[i - fps * 2][1], com[i - fps * 3][0], com[i - fps * 3][1])
    cdist2 = pyth(com[i - fps][0], com[i - fps][1], com[i - fps * 2][0], com[i - fps * 2][1])
    cdist3 = pyth(com[i][0], com[i][1], com[i - fps][0], com[i - fps][1])
    if cdist1 < 1 and cdist2 < 1 and cdist3 < 1:
        for j in range(i - fps * 3 + 1, i + 1):
            freeze[j] = 1
    elif cdist1 > 1 and cdist2 > 1 and cdist3 > 1:
        for j in range(i - fps * 2 + 1, i + 1):
            freeze[j] = 0

total_freeze_time = Decimal(sum(freeze)) / Decimal(fps)
freeze_count = 0
for i in range(l):
    if freeze[i] - freeze[i - 1] == 1:
        freeze_count += 1

total_distance = sum(cdist)
'''
max_distance_02s = sum([cdist[i] for i in range(1, fps // 5 + 1)])
distance_02s = max_distance_02s
for i in range(fps // 5 + 1, l):
    distance_02s += (cdist[i] - cdist[i - fps // 5])
    if distance_02s > max_distance_02s:
        max_distance_02s = distance_02s
'''
max_distance_1s = sum([cdist[j] for j in range(1, fps + 1)])
cdist1 = [max_distance_1s for i in range(fps + 1)]
for i in range(fps + 1, l):
    cdist1.append(cdist1[i - 1] + cdist[i] - cdist[i - fps])
    if cdist1[i] > max_distance_1s:
        max_distance_1s = cdist1[i]

accel = [0 for i in range(l)]
for i in range(2, l):
    accel[i] = (speed_avg[i] - speed_avg[i - 1]) * fps

accel_avg = runavg(accel, 2, l)

accel_pos = []
max_accel = []
total_accel = []
mean_accel = []
dur_accel = []
i = 2
while i < l:
    if accel_avg[i] > 100:
        j = i
        max_accel_t = 0
        total_accel_t = 0
        while j < l and accel_avg[j] > 100:
            total_accel_t += accel_avg[j]
            if accel_avg[j] > max_accel_t:
                max_accel_t = accel_avg[j]
            j += 1
        if j >= l:
            break
        accel_pos.append(i)
        max_accel.append(max_accel_t)
        dur_accel.append(j - i)
        total_accel.append(total_accel_t)
        mean_accel.append(total_accel_t / (j - i))
        i = j
    i += 1

i = 0
while i < len(accel_pos):
    if dur_accel[i] <= 1 or mean_accel[i] < 225:
        accel_pos.pop(i)
        max_accel.pop(i)
        total_accel.pop(i)
        mean_accel.pop(i)
        dur_accel.pop(i)
        i -= 1
    i += 1

accel_freq = len(accel_pos)

accel_freq_active = accel_freq
for i in range(accel_freq):
    if freeze[accel_pos[i]] == 1:
        accel_freq_active -= 1

with open('kinematics_data.csv', 'w') as file:
    file.write('X' + ', ' + 'Y' + ', ' + 'Distance' + ', ' + 'Freeze' + ', '
               + 'Speed' + ', ' + 'Speed avg' + ', ' + 'Accel' + ', ' + 'Accel avg' + '\n')
    for i in range(l):
        file.write(str(com[i][0]) + ', ' + str(com[i][1]) + ', ' + str(cdist[i]) + ', ' + str(freeze[i]) + ', '
                   + str(speed[i]) + ', ' + str(speed_avg[i]) + ', ' + str(accel[i]) + ', ' + str(accel_avg[i]) + '\n')

with open('kinematics_analysis.csv', 'w') as file:
    file.write('Total distance' + ', ' + 'Mean active speed' + ', ' 
               + 'Max speed' + ', ' + 'Max speed in 1s' + ', '
               + 'Total freeze time' + ', ' + '% of time freezing' + ', ' + 'Number of freezing episodes per min' + ', '
               + 'Number of accelerations per min' + ', ' + 'Number of accelerations per min of active mvt' + ', '
               + 'Mean peak acceleration' + ', ' + 'Mean acceleration' + ', ' + 'Mean acceleration duration' + '\n')
    file.write(str(total_distance) + ', ' + str(total_distance / Decimal(l // fps - total_freeze_time)) + ', ' 
               + str(max(speed_avg)) + ', ' + str(max_distance_1s) + ', '
               + str(total_freeze_time) + ', ' + str(total_freeze_time * fps / Decimal(l)) + ', ' + str(freeze_count) + ', ' 
               + str(accel_freq / (l // fps) * 60) + ', ' + str(accel_freq_active * (60 - total_freeze_time) / (l // fps)) + ', '
               + str(sum(max_accel) / accel_freq) + ', ' + str(sum(mean_accel) / accel_freq) + ', ' + str(sum(dur_accel) / fps / accel_freq) + '\n')
    #file.write('Thigmotaxis' + ', ' + str(Decimal(sum(distancetowall)) / Decimal(l)) + '\n')

vt_graph = np.zeros((1000, l, 3), np.uint8)
for i in range(l):
    cv2.circle(vt_graph, (i + 1, 1000 - int(round(speed_avg[i]))), 1, (255, 255, 255), -1)
for i in range(accel_freq):
    for j in range(accel_pos[i], accel_pos[i] + dur_accel[i]):
        cv2.circle(vt_graph, (j + 1, 1000 - int(round(speed_avg[j]))), 1, (0, 127, 255), -1)
cv2.imwrite('vt_graph.png', vt_graph)

if spine_analysis:
    
    pi = Decimal(math.pi)
    
    sdist = [[0 for i in range(l)] for j in range(8)]
    bl = [[0 for i in range(l)] for j in range(7)]
    direction = [0 for i in range(l)]
    turn = [0 for i in range(l)]
    turn_avg = [0 for i in range(l)]
    angle = [[0 for i in range(l)] for j in range(7)]
    angle_change = [[0 for i in range(l)] for j in range(6)]
    headtotail_angle_avg = [0 for i in range(l)]
    total_abs_angle_avg = [0 for i in range(l)]
    headtotail_angle_change = [0 for i in range(l)]
    total_abs_angle_change = [0 for i in range(l)]
    
    def find_sdist(start, end):
        for index in range(start, end):
            for jndex in range(1, 8):
                sdist[jndex][index] = pyth(spine[index][jndex * 2 - 1], spine[index][jndex * 2], spine[index - 1][jndex * 2 - 1], spine[index - 1][jndex * 2])
            sdist[0][index] = stdev([sdist[jndex][index] for jndex in range(1, 8)])

    def find_bl(start, end):
        for index in range(start, end):
            for jndex in range(3, 14, 2):
                bl[jndex // 2][index] = pyth(spine[index][jndex], spine[index][jndex + 1], spine[index][jndex - 2], spine[index][jndex - 1])
            bl[0][index] = sum([bl[jndex][index] for jndex in range(1, 7)])
    
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
    
    def cal_angle(index):
        for jndex in range(1, 10, 2):
            angle[jndex // 2 + 1][index] = cal_turn(cal_direction(spine[index][jndex], spine[index][jndex + 1], spine[index][jndex + 2], spine[index][jndex + 3]),
                                                    cal_direction(spine[index][jndex + 2], spine[index][jndex + 3], spine[index][jndex + 4], spine[index][jndex + 5]))
        angle[0][index] = sum([angle[kndex][index] for kndex in range(1, 6)])
        angle[6][index] = sum([abs(angle[kndex][index]) for kndex in range(1, 6)])
    
    with open('spine.csv', 'r') as csvfile:
        video = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
    with open('spine.csv', 'r') as csvfile:
        spine = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
    for i in range(l):
        for j in range(6):
            video[i].append(video[i].pop(1))
            spine[i].append(spine[i].pop(1))
        for j in range(1, 14, 2):
            video[i][j] *= ratiox
            spine[i][j] *= ratiox
            video[i][j + 1] *= ratioy
            spine[i][j + 1] *= ratioy
    
    #do all the things
    find_sdist(1, l)
    find_bl(0, l)

    for i in range(l):
        direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
    for i in range(1, l):
        turn[i] = cal_turn(direction[i - 1], direction[i])
    turn_avg = runavg(turn, 1, l)

    for i in range(l):
        cal_angle(i)
    for i in range(1, l):
        for j in range(1, 6):
            angle_change[j][i] = angle[j][i] - angle[j][i - 1]

    headtotail_angle_avg = runavg(angle[0], 0, l)
    total_abs_angle_avg = runavg(angle[6], 0, l)
  
    for i in range(1, l):
        headtotail_angle_change[i] = headtotail_angle_avg[i] - headtotail_angle_avg[i - 1]
        total_abs_angle_change[i] = total_abs_angle_avg[i] - total_abs_angle_avg[i - 1]
    
    headtotail_angle_change_avg = runavg(headtotail_angle_change, 1, l)
    total_abs_angle_change_avg = runavg(total_abs_angle_change, 1, l)

    with open('original_spine_distances.csv', 'w') as file:
        header = ['Frame', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 
                  'Distance stdev', 'Distance 1', 'Distance 2', 'Distance 3', 'Distance 4', 'Distance 5', 'Distance 6', 'Distance 7', 'Spine length']
        for word in header:
            file.write(str(word) + ', ')
        file.write('\n')
        for i in range(l):
            file.write(str(i + 1) + ', ')
            for j in range(1, 15):
                file.write(str(spine[i][j]) + ', ')
            for j in range(8):
                file.write(str(sdist[j][i]) + ', ')
            file.write(str(bl[0][i]) + '\n')

    with open('original_spine_turns_and_angles.csv', 'w') as file:
        file.write('Frame' + ', ' + 'Angle 1' + ', ' + 'Angle 2' + ', ' + 'Angle 3' + ', ' + 'Angle 4' + ', ' + 'Angle 5' + ', '
                   + 'Angle 1 change' + ', ' + 'Angle 2 change' + ', ' + 'Angle 3 change' + ', ' + 'Angle 4 change' + ', ' + 'Angle 5 change' + ', '
                   + 'Direction' + ', ' + 'Turn' + ', ' + 'Turn avg' + ', ' 
                   + 'headtotail_angle' + ', ' + 'headtotail_angle_avg' + ', ' + 'headtotail_angle_change' + ', ' + 'headtotail_angle_change_avg' + ', '
                   + 'total_abs_angle' + ', ' + 'total_abs_angle_avg' + ', ' + 'total_abs_angle_change' + ', ' + 'total_abs_angle_change_avg' + '\n')
        for i in range(l):
            file.write(str(i + 1) + ', ')
            for j in range(1, 6):
                file.write(str(angle[j][i]) + ', ')
            for j in range(1, 6):
                file.write(str(angle_change[j][i]) + ', ')
            file.write(str(direction[i]) + ', ' + str(turn[i]) + ', ' + str(turn_avg[i]) + ', ' 
                       + str(angle[0][i]) + ', ' + str(headtotail_angle_avg[i]) + ', '
                       + str(headtotail_angle_change[i]) + ', ' + str(headtotail_angle_change_avg[i]) + ', '
                       + str(angle[6][i]) + ', ' + str(total_abs_angle_avg[i]) + ', '
                       + str(total_abs_angle_change[i]) + ', ' + str(total_abs_angle_change_avg[i]) + '\n')
    
    with open('original_input.csv', 'w') as file:
        for i in range(l):
            for j in range(1, 6):
                file.write(str(angle[j][i]) + ', ')
            file.write(str(spine[i][7]) + ', ' + str(spine[i][8]) + '\n')

    wrong_target = []
    i = 1
    while i < l - 1:
        if Decimal.sqrt((video[i][7] - com[i][0]) ** 2 + (video[i][8] - com[i][1]) ** 2) >= 20:
            j = i + 1
            while Decimal.sqrt((video[j][7] - com[j][0]) ** 2 + (video[j][8] - com[j][1]) ** 2) >= 20:
                j += 1
            for k in range(i, j):
                wrong_target.append(k)
                for ii in range(1, 15):
                    spine[k][ii] = (spine[i - 1][ii] * (j - k) + spine[j][ii] * (k - i + 1)) / (j - i + 1)
            i = j
        i += 1
    
    wrong_angle = [[] for i in range(6)]
    for i in range(l):
        for j in range(1, 6):
            if angle[j][i] >= 1.309:
                wrong_angle[j].append(i)
    
    status = [0]
    spine_cutoff = statistics.median(bl[0]) / Decimal(1.8)
    for i in range(1, l):
        if bl[0][i] < spine_cutoff:
            status.append(8)
        elif sdist[0][i] >= 2:
            status.append(2)
        else:
            status.append(0)
    
    abnormal_spine_count = 0
    for i in range(l):
        if status[i] != 0: 
            abnormal_spine_count += 1
    
    cap = cv2.VideoCapture("ivideo.avi")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cap.isOpened() == True:
        ret, frame = cap.read()
        if ret == True:
            for i in range(7):
                cv2.circle(frame, (int(spine[0][i * 2 + 1] / ratiox), int(spine[0][i * 2 + 2] / ratioy)), 2, (0, i * 42, 255), -1)
            cv2.imwrite('First frame.png', frame)
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    first_frame_reversed = input('Enter 0 if the dots go from red on the head to yellow on the tail, others if reverse:')
    if first_frame_reversed == '0':
        first_frame_reversed = False
    else:
        first_frame_reversed = True
    
    inv_timepts = []
    if first_frame_reversed == True:
        inv_timepts.append(0)
    i = 1
    while i < l - 1:
        if status[i] == 2:
            j = i - 1
            while status[i] == 8:
                j -= 1
            headdistance = Decimal.sqrt((spine[i][1] - spine[j][1]) ** 2 + (spine[i][2] - spine[j][2]) ** 2)
            taildistance = Decimal.sqrt((spine[i][13] - spine[j][13]) ** 2 + (spine[i][14] - spine[j][14]) ** 2)
            alt_headdistance = Decimal.sqrt((spine[i][13] - spine[j][1]) ** 2 + (spine[i][14] - spine[j][2]) ** 2)
            alt_taildistance = Decimal.sqrt((spine[i][1] - spine[j][13]) ** 2 + (spine[i][2] - spine[j][14]) ** 2)
            if alt_headdistance < headdistance and alt_taildistance < taildistance:
                inv_timepts.append(i)
        i += 1
    inv_timepts_len = len(inv_timepts)
    if inv_timepts_len % 2 == 1:
        inv_timepts.append(l)
    
    sdinvcheck = []
    for i in range(inv_timepts_len):
        sdinvcheck.append(sdist[0][inv_timepts[i]])
    
    inv_timepts_len = len(inv_timepts)
    
    with open('error_report.csv', 'w') as file:
        file.write('Number of non-fish recognitions' + ', '
                   + 'Number of frames with abnormally large spine angles (1-5)' + ', ' + '' + ', ' + '' + ', ' + '' + ', ' + '' + ', '
                   + 'Number of fish inversions' + ', '
                   + 'Number of frames with some other abnormalities' + '\n')
        file.write(str(len(wrong_target)) + ', ' + str(len(wrong_angle[1])) + ', ' + str(len(wrong_angle[2])) + ', '
                   + str(len(wrong_angle[3])) + ', ' + str(len(wrong_angle[4])) + ', ' + str(len(wrong_angle[5])) + ', '
                   + str(inv_timepts_len) + ', ' + str(abnormal_spine_count - inv_timepts_len) + '\n')
    
    if spine_inversion:
        
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
            find_sdist(inv_timepts[i], inv_timepts[i + 1])
            if j >= l:
                break
            find_sdist(inv_timepts[i + 1], inv_timepts[i + 1] + 1)
            i += 2
        
        #do all the things
        find_sdist(1, l)
        find_bl(0, l)

        for i in range(l):
            direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
        for i in range(1, l):
            turn[i] = cal_turn(direction[i - 1], direction[i])
        turn_avg = runavg(turn, 1, l)

        for i in range(l):
            cal_angle(i)
        for i in range(1, l):
            for j in range(1, 6):
                angle_change[j][i] = angle[j][i] - angle[j][i - 1]

        headtotail_angle_avg = runavg(angle[0], 0, l)
        total_abs_angle_avg = runavg(angle[6], 0, l)
      
        for i in range(1, l):
            headtotail_angle_change[i] = headtotail_angle_avg[i] - headtotail_angle_avg[i - 1]
            total_abs_angle_change[i] = total_abs_angle_avg[i] - total_abs_angle_avg[i - 1]
        
        headtotail_angle_change_avg = runavg(headtotail_angle_change, 1, l)
        total_abs_angle_change_avg = runavg(total_abs_angle_change, 1, l)
        
    for i in range(1, l):
        if bl[0][i] < spine_cutoff:
            status[i] = 8
        elif sdist[0][i] >= 2:
            status[i] = 2
        else:
            status[i] = 0
    
    if spine_othercorrections:
    
        avg_timepts = [[], []]
        i = 2
        while i < l - 1: # correction by averaging
            if status[i] != 0 or turn[i] > pi / 4:
                j = i + 1
                while j < l and (status[j] != 0 or turn[j] > pi / 4):
                    j += 1
                avg_timepts[0].append(i)
                avg_timepts[1].append(j)
                for k in range(i, j):
                    for ii in range(1, 15):
                        spine[k][ii] = (spine[i - 1][ii] * (j - k) + spine[j][ii] * (k - i + 1)) / (j - i + 1)
                if j >= l:
                    break
                i = j
            i += 1
        
        #do all the things
        find_sdist(1, l)
        find_bl(0, l)

        for i in range(l):
            direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
        for i in range(1, l):
            turn[i] = cal_turn(direction[i - 1], direction[i])
        turn_avg = runavg(turn, 1, l)

        for i in range(l):
            cal_angle(i)
        for i in range(1, l):
            for j in range(1, 6):
                angle_change[j][i] = angle[j][i] - angle[j][i - 1]

        headtotail_angle_avg = runavg(angle[0], 0, l)
        total_abs_angle_avg = runavg(angle[6], 0, l)
      
        for i in range(1, l):
            headtotail_angle_change[i] = headtotail_angle_avg[i] - headtotail_angle_avg[i - 1]
            total_abs_angle_change[i] = total_abs_angle_avg[i] - total_abs_angle_avg[i - 1]
        
        headtotail_angle_change_avg = runavg(headtotail_angle_change, 1, l)
        total_abs_angle_change_avg = runavg(total_abs_angle_change, 1, l)
    
    for i in range(1, l):
        if bl[0][i] < spine_cutoff:
            status[i] = 8
        elif sdist[0][i] >= 2:
            status[i] = 2
        else:
            status[i] = 0
    
    clockwise_time = 0
    clockwise_turn = 0
    anticlockwise_time = 0
    anticlockwise_turn = 0
    for i in range(1, l):
        if turn_avg[i] > 0 and freeze[i] == 0:
            clockwise_turn += abs(turn_avg[i])
            clockwise_time += 1
        if turn_avg[i] < 0 and freeze[i] == 0:
            anticlockwise_turn += abs(turn_avg[i])
            anticlockwise_time += 1
    total_turn = clockwise_turn + anticlockwise_turn
    
    with open('turn_analysis.csv', 'w') as file:
        file.write('Total turn' + ', ' + 'Clockwise turn' + ', ' + 'Anticlockwise turn' + ', ' + 'Turn preference' + ', '
                   + 'Clockwise time' + ', ' + 'Anticlockwise time' + ', ' + 'Turn time preference' + '\n')
        file.write(str(total_turn * 180 / pi) + ', ' + str(clockwise_turn * 180 / pi) + ', '
                   + str(anticlockwise_turn * 180 / pi) + ', ' + str((clockwise_turn - anticlockwise_turn) / (clockwise_turn + anticlockwise_turn)) + ', '
                   + str(Decimal(clockwise_time) / Decimal(fps)) + ', ' + str(Decimal(anticlockwise_time) / Decimal(fps)) + ', '
                   + str((clockwise_time - anticlockwise_time) / (clockwise_time + anticlockwise_time)) + '\n')
    
    if spine_inversion or spine_othercorrections:
    
        with open('converted_spine_distances.csv', 'w') as file:
            header = ['Frame', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 
                      'Distance stdev', 'Distance 1', 'Distance 2', 'Distance 3', 'Distance 4', 'Distance 5', 'Distance 6', 'Distance 7', 'Spine length']
            for word in header:
                file.write(str(word) + ', ')
            file.write('\n')
            for i in range(l):
                file.write(str(i + 1) + ', ')
                for j in range(1, 15):
                    file.write(str(spine[i][j]) + ', ')
                for j in range(8):
                    file.write(str(sdist[j][i]) + ', ')
                file.write(str(bl[0][i]) + '\n')
    
        with open('converted_spine_turns_and_angles.csv', 'w') as file:
            file.write('Frame' + ', ' + 'Angle 1' + ', ' + 'Angle 2' + ', ' + 'Angle 3' + ', ' + 'Angle 4' + ', ' + 'Angle 5' + ', '
                       + 'Angle 1 change' + ', ' + 'Angle 2 change' + ', ' + 'Angle 3 change' + ', ' + 'Angle 4 change' + ', ' + 'Angle 5 change' + ', '
                       + 'Direction' + ', ' + 'Turn' + ', ' + 'Turn avg' + ', ' 
                       + 'headtotail_angle' + ', ' + 'headtotail_angle_avg' + ', ' + 'headtotail_angle_change' + ', ' + 'headtotail_angle_change_avg' + ', '
                       + 'total_abs_angle' + ', ' + 'total_abs_angle_avg' + ', ' + 'total_abs_angle_change' + ', ' + 'total_abs_angle_change_avg' + '\n')
            for i in range(l):
                file.write(str(i + 1) + ', ')
                for j in range(1, 6):
                    file.write(str(angle[j][i]) + ', ')
                for j in range(1, 6):
                    file.write(str(angle_change[j][i]) + ', ')
                file.write(str(direction[i]) + ', ' + str(turn[i]) + ', ' + str(turn_avg[i]) + ', ' 
                           + str(angle[0][i]) + ', ' + str(headtotail_angle_avg[i]) + ', '
                           + str(headtotail_angle_change[i]) + ', ' + str(headtotail_angle_change_avg[i]) + ', '
                           + str(angle[6][i]) + ', ' + str(total_abs_angle_avg[i]) + ', '
                           + str(total_abs_angle_change[i]) + ', ' + str(total_abs_angle_change_avg[i]) + '\n')
        
        with open('converted_input.csv', 'w') as file:
            for i in range(l):
                for j in range(1, 6):
                    file.write(str(angle[j][i]) + ', ')
                file.write(str(spine[i][7]) + ', ' + str(spine[i][8]) + '\n')
    
    if show_video:
        
        output = cv2.VideoWriter("ovideo.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 100, (width, height))

        i = 0
        while cap.isOpened() == True:
            ret, frame = cap.read()
            if ret == True:
                for j in range(7):
                    cv2.circle(frame, (int(spine[i][j * 2 + 1] / ratiox), int(spine[i][j * 2 + 2] / ratioy)), 2, (0, j * 42, 255), -1)
                output.write(frame)
                i += 1
            else:
                break

        output.release()
       
    cap.release()
    cv2.destroyAllWindows()

low_dur = 0
med_dur = 0
hi_dur = 0
for i in range(l):
    if freeze[i] == 0:
        if cdist1[i] >= 5 and cdist1[i] < 20:
            low_dur += 1
        elif cdist1[i] >= 20 and cdist1[i] < 40:
            med_dur += 1
        elif cdist1[i] >= 40:
            hi_dur += 1
low_dur = Decimal(low_dur) / Decimal(fps)
med_dur = Decimal(med_dur) / Decimal(fps)
hi_dur = Decimal(hi_dur) / Decimal(fps)

bend6 = []
bend6_max = []
bend6_pos = []
bend6_dur = []
i = 0
while i < l:
    if total_abs_angle_change_avg[i] >= 0.03 and freeze[i] == 0:
        j = i
        max_total_abs_angle = 0
        while j < l and total_abs_angle_change_avg[j] >= 0.03:
            if total_abs_angle_avg[j] > max_total_abs_angle:
                max_total_abs_angle = total_abs_angle_avg[j]
            j += 1
        if j >= l:
            break
        bend6_pos.append(i)
        bend6.append(total_abs_angle_avg[j - 1] - total_abs_angle_avg[i - 1])
        bend6_max.append(max_total_abs_angle)
        bend6_dur.append(j - i)
        i = j
    i += 1

i = 0
while i < len(bend6):
    if bend6_max[i] < 0.0873:
        bend6_pos.pop(i)
        bend6_max.pop(i)
        bend6.pop(i)
        bend6_dur.pop(i)
        i -= 1
    i += 1

bend6_low_dur = 0
bend6_med_dur = 0
bend6_hi_dur = 0
bend6_low_freq = 0
bend6_med_freq = 0
bend6_hi_freq = 0
bend6_low_bend = 0
bend6_med_bend = 0
bend6_hi_bend = 0
bend6_len = len(bend6)
for i in range(bend6_len):
    if cdist1[bend6_pos[i]] >= 5 and cdist1[bend6_pos[i]] < 20:
        bend6_low_freq += 1
        bend6_low_bend += bend6_max[i]
        bend6_low_dur += bend6_dur[i]
    elif cdist1[bend6_pos[i]] >= 20 and cdist1[bend6_pos[i]] < 40:
        bend6_med_freq += 1
        bend6_med_bend += bend6_max[i]
        bend6_med_dur += bend6_dur[i]
    elif cdist1[bend6_pos[i]] >= 40:
        bend6_hi_freq += 1
        bend6_hi_bend += bend6_max[i]
        bend6_hi_dur += bend6_dur[i]

bend6_low_bend = bend6_low_bend * 180 / pi
bend6_med_bend = bend6_med_bend * 180 / pi
bend6_hi_bend = bend6_hi_bend * 180 / pi
bend6_mean = (bend6_low_bend + bend6_med_bend + bend6_hi_bend) / bend6_len

bend6_graph = np.zeros((1000, l, 3), np.uint8)
for i in range(l):
    cv2.circle(bend6_graph, (i + 1, 1000 - int(round(total_abs_angle_avg[i] * 180))), 1, (255, 255, 255), -1)
for i in range(bend6_len):
    for j in range(bend6_pos[i], bend6_pos[i] + bend6_dur[i]):
        cv2.circle(bend6_graph, (j + 1, 1000 - int(round(total_abs_angle_avg[j] * 180))), 1, (0, 127, 255), -1)
cv2.imwrite('total_abs_angle_graph.png', bend6_graph)

with open('total_abs_angle_analysis.csv', 'w') as file:    
    file.write('Mean tail bend angle' + ', ' + 'Mean low speed tail bend angle' + ', '
               + 'Mean medium speed tail bend angle' + ', ' + 'Mean high speed tail bend angle' + ', '
               + 'Tail beat frequency' + ', ' + 'Low speed tail beat frequency' + ', '
               + 'Medium speed tail beat frequency' + ', ' + 'High speed tail beat frequency' + '\n')
    file.write(str(bend6_mean) + ', ')
    if bend6_low_freq > 0:
        file.write(str(bend6_low_bend / bend6_low_freq) + ', ')
    else:
        file.write('NA' + ', ')
    if bend6_med_freq > 0:
        file.write(str(bend6_med_bend / bend6_med_freq) + ', ')
    else:
        file.write('NA' + ', ')
    if bend6_hi_freq > 0: 
        file.write(str(bend6_hi_bend / bend6_hi_freq) + ', ')
    else:
        file.write('NA' + ', ')
    file.write(str(bend6_len / (60 - total_freeze_time)) + ', ')
    if bend6_low_freq > 0:
        file.write(str(bend6_low_freq / low_dur) + ', ')
    else:
        file.write('NA' + ', ')
    if bend6_med_freq > 0:
        file.write(str(bend6_med_freq / med_dur) + ', ')
    else:
        file.write('NA' + ', ')
    if bend6_hi_freq > 0: 
        file.write(str(bend6_hi_freq / hi_dur) + ', ')
    else:
        file.write('NA' + ', ')
    file.write('\n')

bend0_abs = []
for i in range(l):
    if headtotail_angle_avg[i] > 0:
        bend0_abs.append(headtotail_angle_avg[i])
    else:
        bend0_abs.append(Decimal(0))
bend0_abs_change = [0]
for i in range(1, l):
    bend0_abs_change.append(bend0_abs[i] - bend0_abs[i - 1])
bend0_abs_change_avg = runavg(bend0_abs_change, 1, l)

bend0p = []
bend0p_max = []
bend0p_pos = []
bend0p_dur = []
i = 0
while i < l:
    if bend0_abs_change_avg[i] >= 0.03 and freeze[i] == 0:
        j = i
        max_headtotail_angle = 0
        while j < l and bend0_abs_change_avg[j] >= 0.03:
            if bend0_abs[j] > max_headtotail_angle:
                max_headtotail_angle = bend0_abs[j]
            j += 1
        if j >= l:
            break
        bend0p_pos.append(i)
        bend0p.append(bend0_abs[j - 1] - bend0_abs[i - 1])
        bend0p_max.append(max_headtotail_angle)
        bend0p_dur.append(j - i)
        i = j
    i += 1

i = 0
while i < len(bend0p):
    if bend0p_max[i] < 0.0873:
        bend0p_pos.pop(i)
        bend0p_max.pop(i)
        bend0p.pop(i)
        bend0p_dur.pop(i)
        i -= 1
    i += 1

bend0_abs = []
for i in range(l):
    if headtotail_angle_avg[i] < 0:
        bend0_abs.append(-headtotail_angle_avg[i])
    else:
        bend0_abs.append(0)
bend0_abs_change = [0]
for i in range(1, l):
    bend0_abs_change.append(bend0_abs[i] - bend0_abs[i - 1])
bend0_abs_change_avg = runavg(bend0_abs_change, 1, l)

bend0n = []
bend0n_max = []
bend0n_pos = []
bend0n_dur = []
i = 0
while i < l:
    if bend0_abs_change_avg[i] >= 0.03 and freeze[i] == 0:
        j = i
        max_headtotail_angle = 0
        while j < l and bend0_abs_change_avg[j] >= 0.03:
            if bend0_abs[j] > max_headtotail_angle:
                max_headtotail_angle = bend0_abs[j]
            j += 1
        if j >= l:
            break
        bend0n_pos.append(i)
        bend0n.append(bend0_abs[j - 1] - bend0_abs[i - 1])
        bend0n_max.append(max_headtotail_angle)
        bend0n_dur.append(j - i)
        i = j
    i += 1

i = 0
while i < len(bend0n):
    if bend0n_max[i] < 0.0873:
        bend0n_pos.pop(i)
        bend0n_max.pop(i)
        bend0n.pop(i)
        bend0n_dur.pop(i)
        i -= 1
    i += 1

bend0p_low_dur = 0
bend0p_med_dur = 0
bend0p_hi_dur = 0
bend0p_low_freq = 0
bend0p_med_freq = 0
bend0p_hi_freq = 0
bend0p_low_bend = 0
bend0p_med_bend = 0
bend0p_hi_bend = 0
bend0p_len = len(bend0p)
for i in range(bend0p_len):
    if cdist1[bend0p_pos[i]] >= 5 and cdist1[bend0p_pos[i]] < 20:
        bend0p_low_freq += 1
        bend0p_low_bend += bend0p_max[i]
        bend0p_low_dur += bend0p_dur[i]
    elif cdist1[bend0p_pos[i]] >= 20 and cdist1[bend0p_pos[i]] < 40:
        bend0p_med_freq += 1
        bend0p_med_bend += bend0p_max[i]
        bend0p_med_dur += bend0p_dur[i]
    elif cdist1[bend0p_pos[i]] >= 40:
        bend0p_hi_freq += 1
        bend0p_hi_bend += bend0p_max[i]
        bend0p_hi_dur += bend0p_dur[i]

bend0p_low_bend = bend0p_low_bend * 180 / pi
bend0p_med_bend = bend0p_med_bend * 180 / pi
bend0p_hi_bend = bend0p_hi_bend * 180 / pi
bend0p_bend = bend0p_low_bend + bend0p_med_bend + bend0p_hi_bend

bend0n_low_dur = 0
bend0n_med_dur = 0
bend0n_hi_dur = 0
bend0n_low_freq = 0
bend0n_med_freq = 0
bend0n_hi_freq = 0
bend0n_low_bend = 0
bend0n_med_bend = 0
bend0n_hi_bend = 0
bend0n_len = len(bend0n)
for i in range(bend0n_len):
    if cdist1[bend0n_pos[i]] >= 5 and cdist1[bend0n_pos[i]] < 20:
        bend0n_low_freq += 1
        bend0n_low_bend += bend0n_max[i]
        bend0n_low_dur += bend0n_dur[i]
    elif cdist1[bend0n_pos[i]] >= 20 and cdist1[bend0n_pos[i]] < 40:
        bend0n_med_freq += 1
        bend0n_med_bend += bend0n_max[i]
        bend0n_med_dur += bend0n_dur[i]
    elif cdist1[bend0n_pos[i]] >= 40:
        bend0n_hi_freq += 1
        bend0n_hi_bend += bend0n_max[i]
        bend0n_hi_dur += bend0n_dur[i]

bend0n_low_bend = bend0n_low_bend * 180 / pi
bend0n_med_bend = bend0n_med_bend * 180 / pi
bend0n_hi_bend = bend0n_hi_bend * 180 / pi
bend0n_bend = bend0n_low_bend + bend0n_med_bend + bend0n_hi_bend

bend0_low_bend = bend0p_low_bend + bend0n_low_bend
bend0_med_bend = bend0p_med_bend + bend0n_med_bend
bend0_hi_bend = bend0p_hi_bend + bend0n_hi_bend
bend0_low_freq = bend0p_low_freq + bend0n_low_freq
bend0_med_freq = bend0p_med_freq + bend0n_med_freq
bend0_hi_freq = bend0p_hi_freq + bend0n_hi_freq

bend0_bend_mean = (bend0p_bend + bend0n_bend) / (bend0p_len + bend0n_len)
bend0_low_bend_mean = bend0_low_bend / bend0_low_freq
bend0_med_bend_mean = bend0_med_bend / bend0_med_freq
bend0_hi_bend_mean = bend0_hi_bend / bend0_hi_freq

bend0_graph = np.zeros((1200, l, 3), np.uint8)
for i in range(l):
    cv2.circle(bend0_graph, (i + 1, 600 - int(round(headtotail_angle_avg[i] * 180))), 1, (255, 255, 255), -1)
for i in range(bend0p_len):
    for j in range(bend0p_pos[i], bend0p_pos[i] + bend0p_dur[i]):
        cv2.circle(bend0_graph, (j + 1, 600 - int(round(headtotail_angle_avg[j] * 180))), 1, (0, 255, 0), -1)
for i in range(bend0n_len):
    for j in range(bend0n_pos[i], bend0n_pos[i] + bend0n_dur[i]):
        cv2.circle(bend0_graph, (j + 1, 600 - int(round(headtotail_angle_avg[j] * 180))), 1, (0, 127, 255), -1)
cv2.imwrite('headtotail_angle_graph.png', bend0_graph)

with open('headtotail_angle_analysis.csv', 'w') as file:    
    file.write('Mean tail bend angle' + ', ' + 'Mean low speed tail bend angle' + ', '
               + 'Mean medium speed tail bend angle' + ', ' + 'Mean high speed tail bend angle' + ', '
               + 'Tail beat frequency' + ', ' + 'Low speed tail beat frequency' + ', '
               + 'Medium speed tail beat frequency' + ', ' + 'High speed tail beat frequency' + ', '
               + 'Tail bend angle preference' + ', ' + 'Tail beat frequency preference' + '\n')
    file.write(str(bend0_bend_mean) + ', ')
    if bend0_low_freq > 0:
        file.write(str(bend0_low_bend_mean) + ', ')
    else:
        file.write('NA' + ', ')
    if bend0_med_freq > 0:
        file.write(str(bend0_med_bend_mean) + ', ')
    else:
        file.write('NA' + ', ')
    if bend0_hi_freq > 0: 
        file.write(str(bend0_hi_bend_mean) + ', ')
    else:
        file.write('NA' + ', ')
    file.write(str((bend0_low_freq + bend0_med_freq + bend0_hi_freq) / (60 - total_freeze_time)) + ', ')
    if bend0_low_freq > 0:
        file.write(str(bend0_low_freq / low_dur) + ', ')
    else:
        file.write('NA' + ', ')
    if bend0_med_freq > 0:
        file.write(str(bend0_med_freq / med_dur) + ', ')
    else:
        file.write('NA' + ', ')
    if bend0_hi_freq > 0: 
        file.write(str(bend0_hi_freq / hi_dur) + ', ')
    else:
        file.write('NA' + ', ')
    file.write(str((bend0p_bend - bend0n_bend) / (bend0p_bend + bend0n_bend)) + ', '
               + str((bend0p_len - bend0n_len) / (bend0p_len + bend0n_len)) + '\n')

etime = time.time()
print(etime - stime)
