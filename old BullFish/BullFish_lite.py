import csv
import decimal
from decimal import Decimal
import math
import time


stime = time.time()

with open('info.csv', 'r') as csvfile: 
    info = [[int(number) for number in row] for row in csv.reader(csvfile)]
    fps = info[0][0]
    afps = info[0][1]
    step = fps // afps
    xl = info[0][2] 
    xr = info[0][3]
    yu = info[0][4]
    yd = info[0][5]
    ratiox = Decimal(210) / Decimal(xr - xl)
    ratioy = Decimal(144) / Decimal(yd - yu)
with open('com.csv', 'r') as csvfile:
    com = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]
l = len(com)
for i in range(l):
    com[i][0] *= ratiox
    com[i][1] *= ratioy
acom = [com[i] for i in range(0, l, step)]
al = len(acom)

xl *= ratiox
xr *= ratiox
yu *= ratioy
yd *= ratioy
distancetowall = [(min([(com[i][0] - xl) / (xr - xl), (xr - com[i][0]) / (xr - xl), (com[i][1] - yu) / (yd - yu), (yd - com[i][1]) / (yd - yu)]) * 2) for i in range(l)]

adist = [0]
speed = [0]
for i in range(1, al):
    adist.append(Decimal.sqrt((acom[i][0] - acom[i - 1][0]) ** 2 + (acom[i][1] - acom[i - 1][1]) ** 2))
for i in range(1, l):
    speed.append(Decimal.sqrt((com[i][0] - com[i - 1][0]) ** 2 + (com[i][1] - com[i - 1][1]) ** 2) * fps)

speed_avg = [0, sum(speed[1:4]) / 3, sum(speed[1:5]) / 4, sum(speed[1:6]) / 5]
for i in range(4, l - 2):
    speed_avg.append((speed_avg[i - 1] * 5 + speed[i + 2] - speed[i - 3]) / 5)
speed_avg.append(sum(speed[(l - 4):l]) / 4)
speed_avg.append(sum(speed[(l - 3):l]) / 3)

freeze = [0 for i in range(l)] #freezing analysis
for i in range(fps * 3, l):
    distance1 = Decimal.sqrt((com[i - fps * 2][0] - com[i - fps * 3][0]) ** 2 + (com[i - fps * 2][1] - com[i - fps * 3][1]) ** 2)
    distance2 = Decimal.sqrt((com[i - fps][0] - com[i - fps * 2][0]) ** 2 + (com[i - fps][1] - com[i - fps * 2][1]) ** 2)
    distance3 = Decimal.sqrt((com[i][0] - com[i - fps][0]) ** 2 + (com[i][1] - com[i - fps][1]) ** 2)
    if distance1 < 1 and distance2 < 1 and distance3 < 1:
        for j in range(i - fps * 3 + 1, i + 1):
            freeze[j] = 1
    elif distance1 > 1 and distance2 > 1 and distance3 > 1:
        for j in range(i - fps * 2 + 1, i + 1):
            freeze[j] = 0
total_freeze_time = Decimal(sum(freeze)) / Decimal(fps)

total_distance = sum(adist)

max_distance_02s = sum([adist[i] for i in range(1, afps // 5 + 1)])
distance_02s = max_distance_02s
for i in range(afps // 5 + 1, l // step):
    distance_02s += (adist[i] - adist[i - afps // 5])
    if distance_02s > max_distance_02s:
        max_distance_02s = distance_02s

max_distance_1s = sum([adist[i] for i in range(1, afps + 1)])
distance_1s = max_distance_1s
for i in range(afps + 1, l // step):
    distance_1s += (adist[i] - adist[i - afps])
    if distance_1s > max_distance_1s:
        max_distance_1s = distance_1s

def cal_direction(x1, y1, x2, y2): #(x2, y2) to (x1, y1)
    inclin = math.atan2(y1 - y2, x1 - x2)
    if inclin < 0:
        return (inclin + math.pi * 2)
    else:
        return inclin

direction = [0]
for i in range(1, al):
    direction.append(cal_direction(acom[i][0], acom[i][1], acom[i - 1][0], acom[i - 1][1])) 

def cal_turn(s1, s2): #from s1 to s2
    if s2 - s1 > math.pi:
        return s2 - s1 - math.pi * 2
    elif s2 - s1 <= -math.pi:
        return s2 - s1 + math.pi * 2
    else:
        return s2 - s1

turn = [0, 0]
for i in range(2, al):
    turn.append(cal_turn(direction[i - 1], direction[i]))
'''
i = 2
while i < al - 1: # correction by averaging
    if abs(turn[i]) >= math.pi / 4:
        j = i + 1
        while j < al and abs(turn[j]) >= math.pi / 4:
            j += 1
        print(i, j)
        for k in range(i, j):
            acom[k][0] = (acom[i - 1][0] * (j - k) + acom[j][0] * (k - i + 1)) / (j - i + 1)
            acom[k][1] = (acom[i - 1][1] * (j - k) + acom[j][1] * (k - i + 1)) / (j - i + 1)
        if j >= al:
            break
        i = j
    i += 1
for i in range(1, al):
    direction[i] = cal_direction(acom[i][0], acom[i][1], acom[i - 1][0], acom[i - 1][1])
for i in range(2, al):
    turn[i] = cal_turn(direction[i - 1], direction[i])
'''
turn_avg = [0, 0, sum(turn[2:5]) / 3, sum(turn[2:6]) / 4, sum(turn[2:7]) / 5]
for i in range(5, al - 2):
    turn_avg.append((turn_avg[i - 1] * 5 + turn[i + 2] - turn[i - 3]) / 5)
turn_avg.append(sum(turn[(al - 4):al]) / 4)
turn_avg.append(sum(turn[(al - 3):al]) / 3)

clockwise_time = 0
clockwise_turn = 0
anticlockwise_time = 0
anticlockwise_turn = 0
for i in range(2, al):
    if turn_avg[i] > 0 and freeze[i] == 0:
        clockwise_turn += abs(turn_avg[i])
        clockwise_time += 1
    if turn_avg[i] < 0 and freeze[i] == 0:
        anticlockwise_turn += abs(turn_avg[i])
        anticlockwise_time += 1
total_turn = clockwise_turn + anticlockwise_turn

accel = [0 for i in range(l)]
for i in range(2, l):
    accel[i] = (speed_avg[i] - speed_avg[i - 1]) * fps

accel_avg = [0, 0, sum(accel[2:5]) / 3, sum(accel[2:6]) / 4, sum(accel[2:7]) / 5]
for i in range(5, l - 2):
    accel_avg.append((accel_avg[i - 1] * 5 + accel[i + 2] - accel[i - 3]) / 5)
accel_avg.append(sum(accel[(l - 4):l]) / 4)
accel_avg.append(sum(accel[(l - 3):l]) / 3)

move = [0 for i in range(l)]
max_accel = []
mean_accel = []
dur_accel = []
for i in range(2, l):
    if accel_avg[i] > 100:
        move[i] = 1

i = 0
while i < l - 1:
    i += 1
    max_accel_t = 0
    total_accel_t = 0
    j = i
    if move[i] == 1:
        while j < l - 1 and move[j] == 1:
            total_accel_t += accel_avg[j]
            if accel_avg[j] > max_accel_t:
                max_accel_t = accel[j]
            j += 1
        max_accel.append(max_accel_t)
        dur_accel.append(j - i)
        mean_accel.append(total_accel_t / (j - i))
        i = j

accel_freq = len(dur_accel)

with open('raw_1.csv', 'w') as file:
    header = ['Frame', 'X', 'Y', 'Freeze', 'Speed', 'Acceleration']
    for word in header:
        file.write(str(word) + ', ')
    file.write('\n')
    for i in range(l):
        file.write(str(i + 1) + ', ' + str(com[i][0]) + ', ' + str(com[i][1]) + ', ' + str(freeze[i]) + ', ' + str(speed_avg[i]) + ', ' + str(accel_avg[i]) + '\n')

with open('raw_2.csv', 'w') as file:
    header = ['Frame', 'X', 'Y', 'Freeze', 'Direction', 'Turn', 'Turn avg']
    for word in header:
        file.write(str(word) + ', ')
    file.write('\n')
    for i in range(al):
        file.write(str(i * step + 1) + ', ' + str(acom[i][0]) + ', ' + str(acom[i][1]) + ', ' + str(freeze[i * step]) + ', ' + str(direction[i] * 180 / math.pi) + ', ' + str(turn[i] * 180 / math.pi) + ', ' + str(turn_avg[i] * 180 / math.pi) + '\n')

with open('analysis_lite.csv', 'w') as file:    
    file.write('Total distance' + ', ' + str(total_distance) + '\n')
    file.write('Mean speed' + ', ' + str(total_distance * fps / Decimal(l)) + '\n')
    file.write('Max speed' + ', ' + str(max(speed)) + '\n')
    file.write('Max speed in 0.2s' + ', ' + str(max_distance_02s * 5) + '\n')
    file.write('Max speed in 1s' + ', ' + str(max_distance_1s) + '\n')
    file.write('Total freeze time' + ', ' + str(total_freeze_time) + '\n')
    file.write('% of time freezing' + ', ' + str(total_freeze_time * fps / Decimal(l)) + '\n')
    file.write('Mean active speed' + ', ' + str(total_distance / Decimal(l // fps - total_freeze_time)) + '\n')
    file.write('Total turn' + ', ' + str(total_turn * 180 / math.pi) + '\n')
    file.write('Clockwise turn' + ', ' + str(clockwise_turn * 180 / math.pi) + '\n')
    file.write('Anticlockwise turn' + ', ' + str(anticlockwise_turn * 180 / math.pi) + '\n')
    file.write('Turn preference' + ', ' + str((clockwise_turn - anticlockwise_turn) / (clockwise_turn + anticlockwise_turn)) + '\n')
    file.write('Clockwise time' + ', ' + str(Decimal(clockwise_time) / Decimal(afps)) + '\n')
    file.write('Anticlockwise time' + ', ' + str(Decimal(anticlockwise_time) / Decimal(afps)) + '\n')
    file.write('Turn time preference' + ', ' + str((clockwise_time - anticlockwise_time) / (clockwise_time + anticlockwise_time)) + '\n')
    file.write('Thigmotaxis' + ', ' + str(Decimal(sum(distancetowall)) / Decimal(l)) + '\n')
    file.write('Mean peak acceleration' + ', ' + str(sum(max_accel) / accel_freq) + '\n')
    file.write('Max peak acceleration' + ', ' + str(max(max_accel)) + '\n')
    file.write('Mean acceleration' + ', ' + str(sum(mean_accel) / accel_freq) + '\n')
    file.write('Max acceleration' + ', ' + str(max(mean_accel)) + '\n')
    file.write('Mean acceleration duration' + ', ' + str(sum(dur_accel) / fps / accel_freq) + '\n')
    file.write('Number of accelerations per minute' + ', ' + str(accel_freq / (l // fps) * 60) + '\n')

etime = time.time()
print(etime - stime)
