import csv
import decimal
from decimal import Decimal
import math
import statistics
from statistics import stdev
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time

stime = time.time()

with open('info.csv', 'r') as csvfile: 
    info = [[int(number) for number in row] for row in csv.reader(csvfile)]
    fps = info[0][0]
    cfps = info[0][1]
    fpsr = fps // cfps
    xl = info[0][2] 
    xr = info[0][3]
    yu = info[0][4]
    yd = info[0][5]
    ratiox = Decimal(210) / Decimal(xr - xl)
    ratioy = Decimal(144) / Decimal(yd - yu)
with open('com.csv', 'r') as csvfile:
    com = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
cl = len(com)
for i in range(cl):
    com[i][0] *= ratiox
    com[i][1] *= ratioy
with open('spine.csv', 'r') as csvfile:
    video = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
with open('spine.csv', 'r') as csvfile:
    spine = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
l = len(spine)
for i in range(l):
    for j in range(6):
        video[i].append(video[i].pop(1))
        spine[i].append(spine[i].pop(1))
    for j in range(1, 14, 2):
        video[i][j] *= ratiox
        spine[i][j] *= ratiox
        video[i][j + 1] *= ratioy
        spine[i][j + 1] *= ratioy

xl *= ratiox
xr *= ratiox
yu *= ratioy
yd *= ratioy
distancetowall = [(min([(com[i][0] - xl) / (xr - xl), (xr - com[i][0]) / (xr - xl), (com[i][1] - yu) / (yd - yu), (yd - com[i][1]) / (yd - yu)]) * 2) for i in range(cl)]

dist = [0]
speed = [0]
for i in range(1, cl):
    dist.append(Decimal.sqrt((com[i][0] - com[i - 1][0]) ** 2 + (com[i][1] - com[i - 1][1]) ** 2))
    speed.append(dist[i] * cfps)

freeze = [0 for i in range(cl)] #freezing analysis
for i in range(cfps * 3, cl):
    distance1 = Decimal.sqrt((com[i - cfps * 2][0] - com[i - cfps * 3][0]) ** 2 + (com[i - cfps * 2][1] - com[i - cfps * 3][1]) ** 2)
    distance2 = Decimal.sqrt((com[i - cfps][0] - com[i - cfps * 2][0]) ** 2 + (com[i - cfps][1] - com[i - cfps * 2][1]) ** 2)
    distance3 = Decimal.sqrt((com[i][0] - com[i - cfps][0]) ** 2 + (com[i][1] - com[i - cfps][1]) ** 2)
    if distance1 < 1 and distance2 < 1 and distance3 < 1:
        for j in range(i - cfps * 3 + 1, i + 1):
            freeze[j] = 1
    elif distance1 > 1 and distance2 > 1 and distance3 > 1:
        for j in range(i - cfps * 2 + 1, i + 1):
            freeze[j] = 0
total_freeze_time = Decimal(sum(freeze)) / Decimal(cfps)

total_distance = sum(dist)

max_distance_02s = sum([dist[i] for i in range(1, cfps // 5 + 1)])
distance_02s = max_distance_02s
for i in range(cfps // 5 + 1, cl):
    distance_02s += (dist[i] - dist[i - cfps // 5])
    if distance_02s > max_distance_02s:
        max_distance_02s = distance_02s

max_distance_1s = sum([dist[i] for i in range(1, cfps + 1)])
distance_1s = max_distance_1s
for i in range(cfps + 1, cl):
    distance_1s += (dist[i] - dist[i - cfps])
    if distance_1s > max_distance_1s:
        max_distance_1s = distance_1s

i = 1
while i < l - 1:
    if Decimal.sqrt((video[i][7] - com[i // fpsr][0]) ** 2 + (video[i][8] - com[i // fpsr][1]) ** 2) >= 20:
        j = i + 1
        while Decimal.sqrt((video[j][7] - com[j // fpsr][0]) ** 2 + (video[j][8] - com[j // fpsr][1]) ** 2) >= 20:
            j += 1
        for k in range(i, j):
            for ii in range(1, 15):
                spine[k][ii] = (spine[i - 1][ii] * (j - k) + spine[j][ii] * (k - i + 1)) / (j - i + 1)
        i = j
    i += 1

distance = [[0 for i in range(l)] for j in range(8)]
bl = [[0 for i in range(l)] for j in range(7)]

def find_distances(start, end):
    for index in range(start, end):
        for jndex in range(1, 8):
            distance[jndex][index] = Decimal.sqrt((spine[index][jndex * 2 - 1] - spine[index - 1][jndex * 2 - 1]) ** 2 + (spine[index][jndex * 2] - spine[index - 1][jndex * 2]) ** 2)
        distance[0][index] = stdev([distance[jndex][index] for jndex in range(1, 8)])

def find_bl(start, end):
    for index in range(start, end):
        for jndex in range(3, 14, 2):
            bl[jndex // 2][index] = Decimal.sqrt((spine[index][jndex] - spine[index][jndex - 2]) ** 2 + (spine[index][jndex + 1] - spine[index][jndex - 1]) ** 2)
        bl[0][index] = sum([bl[jndex][index] for jndex in range(1, 7)])

find_distances(1, l)
find_bl(0, l)

status = [0]
spine_cutoff = statistics.median(bl[0]) / Decimal(1.8)
for i in range(1, l):
    if bl[0][i] < spine_cutoff:
        status.append(8)
    elif distance[0][i] >= 2:
        status.append(2)
    else:
        status.append(0)

inv_timepts = [] #find time points for fish inversion
i = 2
while i < l - 1:
    if status[i] == 2:
        j = i - 1
        while status[j] == 8:
            j -= 1
        headdistance = Decimal.sqrt((spine[i][1] - spine[j][1]) ** 2 + (spine[i][2] - spine[j][2]) ** 2)
        taildistance = Decimal.sqrt((spine[i][13] - spine[j][13]) ** 2 + (spine[i][14] - spine[j][14]) ** 2)
        alt_headdistance = Decimal.sqrt((spine[i][13] - spine[j][1]) ** 2 + (spine[i][14] - spine[j][2]) ** 2)
        alt_taildistance = Decimal.sqrt((spine[i][1] - spine[j][13]) ** 2 + (spine[i][2] - spine[j][14]) ** 2)
        if alt_headdistance < headdistance and alt_taildistance < taildistance:
            inv_timepts.append(i)
            print(i)
    i += 1

i = 0 #fish inversion
inv_timepts_len = len(inv_timepts)
if inv_timepts_len % 2 == 1:
    inv_timepts.append(l)
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
    find_distances(inv_timepts[i], inv_timepts[i + 1])
    if j >= l:
        break
    find_distances(inv_timepts[i + 1], inv_timepts[i + 1] + 1)
    i += 2

for i in range(1, l):
    if bl[0][i] < spine_cutoff:
        status[i] = 8
    elif distance[0][i] >= 2:
        status[i] = 2
    else:
        status[i] = 0

i = 2
while i < l - 1: # correction by averaging
    if status[i] != 0:
        j = i + 1
        while j < l and status[j] != 0:
            j += 1
        print(i, j)
        for k in range(i, j):
            for ii in range(1, 15):
                spine[k][ii] = (spine[i - 1][ii] * (j - k) + spine[j][ii] * (k - i + 1)) / (j - i + 1)
        find_distances(i, j)
        find_bl(i, j)
        if j >= l:
            break
        find_distances(j, j + 1)
        i = j
    i += 1

def cal_direction(x1, y1, x2, y2): #caudal (x2, y2) to cranial (x1, y1)
    inclin = math.atan2(y1 - y2, x1 - x2)
    if inclin < 0:
        return (inclin + math.pi * 2)
    else:
        return inclin

direction = [cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4]) for i in range(l)] 

def cal_turn(s1, s2): #from s1 to s2
    if s2 - s1 > math.pi:
        return s2 - s1 - math.pi * 2
    elif s2 - s1 <= -math.pi:
        return s2 - s1 + math.pi * 2
    else:
        return s2 - s1

turn = [0]
for i in range(1, l):
    turn.append(cal_turn(direction[i - 1], direction[i]))

angle = [[0 for i in range(l)] for j in range(7)]
def cal_angle(index):
    for jndex in range(1, 10, 2):
        angle[jndex // 2 + 1][index] = cal_turn(cal_direction(spine[index][jndex], spine[index][jndex + 1], spine[index][jndex + 2], spine[index][jndex + 3]), cal_direction(spine[index][jndex + 2], spine[index][jndex + 3], spine[index][jndex + 4], spine[index][jndex + 5]))
    angle[0][index] = sum([angle[kndex][index] for kndex in range(1, 6)])
    angle[6][index] = sum([abs(angle[kndex][index]) for kndex in range(1, 6)])
for i in range(l):
    cal_angle(i)

for i in range(1, l - 1): #adjust for single frame errors
    if turn[i] >= math.pi / 2 or angle[0][i] - angle[0][i - 1] >= math.pi / 2:
        for j in range(1, 15):
            spine[i][j] = (spine[i - 1][j] + spine[i + 1][j]) / 2
        find_distances(i, i + 2)
        find_bl(i, i + 1)
        direction[i] = cal_direction(spine[i][1], spine[i][2], spine[i][3], spine[i][4])
        turn[i] = cal_turn(direction[i - 1], direction[i])
        turn[i + 1] = cal_turn(direction[i], direction[i + 1])
        cal_angle(i)

angle0avg = [sum([angle[0][i] for i in range(3)]) / 3, sum([angle[0][i] for i in range(4)]) / 4, sum([angle[0][i] for i in range(5)]) / 5]
angle6avg = [sum([angle[6][i] for i in range(3)]) / 3, sum([angle[6][i] for i in range(4)]) / 4, sum([angle[6][i] for i in range(5)]) / 5]
for i in range(3, l - 2):
    angle0avg.append((angle0avg[i - 1] * 5 + angle[0][i + 2] - angle[0][i - 3]) / 5)
    angle6avg.append((angle6avg[i - 1] * 5 + angle[6][i + 2] - angle[6][i - 3]) / 5)
angle0avg.append(sum([angle[0][i] for i in range(l - 4, l)]) / 4)
angle0avg.append(sum([angle[0][i] for i in range(l - 3, l)]) / 3)
angle6avg.append(sum([angle[6][i] for i in range(l - 4, l)]) / 4)
angle6avg.append(sum([angle[6][i] for i in range(l - 3, l)]) / 3)

angle0_change = [0]
angle6_change = [0]
for i in range(1, l):
    angle0_change.append(angle0avg[i] - angle0avg[i - 1])
    angle6_change.append(angle6avg[i] - angle6avg[i - 1])

angle6avg = np.array(angle6avg)
peaks = find_peaks(angle6avg, height = 0.0873, distance = 5)
bend6 = peaks[1]['peak_heights']
bend6_pos = peaks[0]
bend6_len = len(bend6)
low_bend6 = []
med_bend6 = []
hi_bend6 = []
for i in range(bend6_len):
    if speed[bend6_pos[i] // fpsr] >= 5 and speed[bend6_pos[i] // fpsr] < 20:
        low_bend6.append(bend6[i])
    elif speed[bend6_pos[i] // fpsr] >= 20 and speed[bend6_pos[i] // fpsr] < 40:
        med_bend6.append(bend6[i])
    elif speed[bend6_pos[i] // fpsr] >= 40:
        hi_bend6.append(bend6[i])

low_dur = 0
med_dur = 0
hi_dur = 0
for i in range(cl):
    if speed[i] >= 5 and speed[i] < 20:
        low_dur += 1 / cfps
    elif speed[i] >= 20 and speed[i] < 40:
        med_dur += 1 / cfps
    elif speed[i] >= 40:
        hi_dur += 1 / cfps

low_freq = len(low_bend6) / low_dur
med_freq = len(med_bend6) / med_dur
hi_freq = len(hi_bend6) / hi_dur
mean_freq = bend6_len / (low_dur + med_dur + hi_dur)

clockwise_time = 0
clockwise_turn = 0
anticlockwise_time = 0
anticlockwise_turn = 0
for i in range(1, l):
    if turn[i] > 0 and freeze[i // fpsr] == 0:
        clockwise_turn += abs(turn[i])
        clockwise_time += 1
    if turn[i] < 0 and freeze[i // fpsr] == 0:
        anticlockwise_turn += abs(turn[i])
        anticlockwise_time += 1



accel = [0 for i in range(cl)]
for i in range(2, cl):
    accel[i] = speed[i] - speed[i - 1]

move = [0 for i in range(cl)]
max_accel = []
total_accel = []
dur_accel = []
for i in range(1, cl):
    if accel[i] > 20:
        move[i] = 1

i = 0
while i < cl - 1:
    i += 1
    max_acce = 0
    total_acce = 0
    j = i
    if move[i] == 1:
        while j < cl - 1 and move[j] == 1:
            total_acce += accel[j]
            if accel[j] > max_acce:
                max_acce = accel[j]
            j += 1
        max_accel.append(max_acce)
        total_accel.append(total_acce)
        dur_accel.append(j - i)
        i = j

accel_freq = len(dur_accel)

with open('converted.csv', 'w') as file:
    header = ['Frame', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'Distance stdev', 'Spine length', 'Direction', 'Turn', 'Angle0', 'Angle0 change', 'Angle0 avg', 'Angle6', 'Angle6 change', 'Angle6 avg', 'X', 'Y', 'Freeze', 'Speed', 'Acceleration']
    for word in header:
        file.write(str(word) + ', ')
    file.write('\n')

    for i in range(l):
        file.write(str(i + 1) + ', ')
        for j in range(1, 15):
            file.write(str(spine[i][j]) + ', ')
        file.write(str(distance[0][i]) + ', ' + str(bl[0][i]) + ', ' + str(direction[i]) + ', ' + str(turn[i]) + ', ' + str(angle[0][i]) + ', ' + str(angle0_change[i]) + ', ' + str(angle0avg[i]) + ', ' + str(angle[6][i]) + ', ' + str(angle6_change[i]) + ', ' + str(angle6avg[i]) + ', ')
        if i % fpsr == 0:
            file.write(str(com[i // fpsr][0]) + ', ' + str(com[i // fpsr][1]) + ', ' + str(freeze[i // fpsr]) + ', ' + str(speed[i // fpsr]) + ', ' + str(accel[i // fpsr]) + '\n')
        else:
            file.write('0' + ', ' + '0' + ', ' + '0' + ', ' + '0' + ', ' + '0' + '\n')

with open('input.csv', 'w') as file:
    for i in range(l):
        for j in range(1, 6):
            file.write(str(angle[j][i]) + ', ')
        file.write(str(spine[i][7]) + ', ' + str(spine[i][8]) + '\n')

with open('analysis.csv', 'w') as file:    
    file.write('Total distance' + ', ' + str(total_distance) + '\n')
    file.write('Mean speed' + ', ' + str(total_distance * fps / Decimal(l)) + '\n')
    file.write('Max speed' + ', ' + str(max(speed)) + '\n')
    file.write('Max speed in 0.2s' + ', ' + str(max_distance_02s * 5) + '\n')
    file.write('Max speed in 1s' + ', ' + str(max_distance_1s) + '\n')
    file.write('Total freeze time' + ', ' + str(total_freeze_time) + '\n')
    file.write('% of time freezing' + ', ' + str(total_freeze_time * fps / Decimal(l)) + '\n')
    file.write('Mean active speed' + ', ' + str(total_distance / Decimal(180 - total_freeze_time)) + '\n')
    file.write('Clockwise turn' + ', ' + str(clockwise_turn * 180 / math.pi) + '\n')
    file.write('Anticlockwise turn' + ', ' + str(anticlockwise_turn * 180 / math.pi) + '\n')
    file.write('Turn preference' + ', ' + str((clockwise_turn - anticlockwise_turn) / (clockwise_turn + anticlockwise_turn)) + '\n')
    file.write('Clockwise time' + ', ' + str(Decimal(clockwise_time) / Decimal(100)) + '\n')
    file.write('Anticlockwise time' + ', ' + str(Decimal(anticlockwise_time) / Decimal(100)) + '\n')
    file.write('Turn time preference' + ', ' + str((clockwise_time - anticlockwise_time) / (clockwise_time + anticlockwise_time)) + '\n')
    file.write('Thigmotaxis' + ', ' + str(Decimal(sum(distancetowall)) / Decimal(cl)) + '\n')
    file.write('Mean peak acceleration' + ', ' + str(sum(max_accel) / accel_freq) + '\n')
    file.write('Mean acceleration' + ', ' + str(sum(total_accel) / accel_freq) + '\n')
    file.write('Mean acceleration duration' + ', ' + str(sum(dur_accel) / cfps / accel_freq) + '\n')
    file.write('Mean tail bend' + ', ' + str((sum(low_bend6) + sum(med_bend6) + sum(hi_bend6)) / (len(low_bend6) + len(med_bend6) + len(hi_bend6))) + '\n')
    file.write('Tail bend frequency' + ', ' + str(mean_freq) + '\n')
    if len(low_bend6) > 0:
        file.write('Mean low speed tail bend' + ', ' + str(sum(low_bend6) / len(low_bend6)) + '\n')
        file.write('low speed tail bend frequency' + ', ' + str(low_freq) + '\n')
    if len(med_bend6) > 0:
        file.write('Mean medium speed tail bend' + ', ' + str(sum(med_bend6) / len(med_bend6)) + '\n')
        file.write('Medium speed tail bend frequency' + ', ' + str(med_freq) + '\n')
    if len(hi_bend6) > 0: 
        file.write('Mean high speed tail bend' + ', ' + str(sum(hi_bend6) / len(hi_bend6)) + '\n')
        file.write('High speed tail bend frequency' + ', ' + str(hi_freq) + '\n')

etime = time.time()
print(etime - stime)
