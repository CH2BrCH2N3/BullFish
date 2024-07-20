import os
import csv
import decimal
from decimal import Decimal
import math
import cv2 as cv
import numpy as np
import time

stime = time.time()

pi = Decimal(math.pi)

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

def cal_bias(p, n):
    return (p - n) / (p + n)

with open('settings.csv', 'r') as csvfile:
    settings = {row[0]: row[1] for row in csv.reader(csvfile)}
    tank_length = Decimal(settings['tank length (mm)'])
    tank_width = Decimal(settings['tank width (mm)'])
    accel_analysis = bool(int(settings['acceleration analysis (0/1)']))
    if accel_analysis:
        accel_cutoff = Decimal(settings['acceleration cutoff (mm/s2)'])
        accel_min_dur = int(settings['minimum duration of an acceleration (frame(s))'])
        accel_min_total = Decimal(settings['minimum speed change of an acceleration (mm/s)'])
        accel_min_max = Decimal(settings['minimum maximum speed slope of an acceleration (mm/s2)'])
        accel_min_mean = Decimal(settings['minimum mean speed slope of an acceleration (mm/s2)'])
        plot_vtgraph = bool(int(settings['plot vt graph (0/1)']))
    spine_analysis = bool(int(settings['angle and turn analysis (0/1)']))
    if spine_analysis:
        spine_inversion = bool(int(settings['spine inversion (0/1)']))
        spine_averaging = bool(int(settings['spine averaging (0/1)']))
        show_video = bool(int(settings['show video (0/1)']))
        turn_cutoff = Decimal(settings['turn cutoff (deg/s)']) * pi / Decimal(180)
        turn_min_dur = int(settings['minimum duration of a turn (frame(s))'])
        turn_min_amp = Decimal(settings['minimum direction change of a turn (deg)']) * pi / Decimal(180)
        turn_min_max = Decimal(settings['minimum maximum angular velocity of a turn (deg/s)']) * pi / Decimal(180)
        turn_min_mean = Decimal(settings['minimum mean angular velocity of a turn (deg/s)']) * pi / Decimal(180)
        plot_turngraph = bool(int(settings['plot turn graph (0/1)']))
        angle_cutoff = Decimal(settings['spine angular velocity cutoff (deg/s)']) * pi / Decimal(180)
        angle_min_dur = int(settings['minimum duration of a tail beat (frame(s))'])
        angle_min_amp = Decimal(settings['minimum tail bend amplitude of a tail beat (deg)']) * pi / Decimal(180)
        angle_min_max = Decimal(settings['minimum maximum angular velocity of a tail beat (deg/s)']) * pi / Decimal(180)
        angle_min_mean = Decimal(settings['minimum mean angular velocity of a tail beat (deg/s)']) * pi / Decimal(180)
        plot_anglegraph = bool(int(settings['plot spine angle graph (0/1)']))

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
        cap = cv.VideoCapture(videoname + '_t.avi')
        fps = int(cap.get(cv.CAP_PROP_FPS))
        cap.release()

with open(videoname + '_com.csv', 'r') as csvfile:
    com = [[Decimal(number) for number in frame] for frame in csv.reader(csvfile)]

l = len(com)
for i in range(l):
    com[i][0] *= ratiox
    com[i][1] *= ratioy

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
for i in range(2, l):
    if freeze[i] - freeze[i - 1] == 1:
        freeze_count += 1

total_distance = sum(cdist)

max_distance_1s = sum([cdist[j] for j in range(1, fps + 1)])
cdist1 = [max_distance_1s for i in range(fps + 1)]
for i in range(fps + 1, l):
    cdist1.append(cdist1[i - 1] + cdist[i] - cdist[i - fps])
    if cdist1[i] > max_distance_1s:
        max_distance_1s = cdist1[i]

analysis = {
    'Total distance (mm)': total_distance,
    'Mean speed (mm/s)': total_distance / Decimal(l // fps),
    'Mean active speed (mm/s)': total_distance / Decimal(l // fps - total_freeze_time),
    'Max speed (mm/s)': max(speed_avg),
    'Max speed in 1s (mm/s)': max_distance_1s,
    '% of time freezing': total_freeze_time * fps / Decimal(l),
    'Number of freezing episodes per min': Decimal(freeze_count) / Decimal(l // fps) * 60
}

class peak_data:
    def __init__(self, pos, length, height, maxslope, meanslope, curspeed, freeze):
        self.pos = pos
        self.length = length
        self.height = height
        self.maxslope = maxslope
        self.meanslope = meanslope
        self.curspeed = curspeed
        self.freeze = freeze

class gen_peaks:
    
    def __init__(self, inputlist, cutoff, criteria):
        
        self.b = inputlist
        self.peaks = []
        i = 1
        while i < l:
            if inputlist[i] > cutoff:
                j = i
                max_slope = 0
                total_change = 0
                while j < l and inputlist[j] > cutoff:
                    total_change += inputlist[j]
                    if inputlist[j] > max_slope:
                        max_slope = inputlist[j]
                    j += 1
                if j >= l:
                    break
                total_change /= Decimal(fps)
                self.peaks.append(peak_data(i, j - i, total_change, max_slope, Decimal(total_change) / Decimal(j - i) * Decimal(fps), cdist1[i], freeze[i]))
                i = j
            i += 1
        
        i = 0
        while i < len(self.peaks):
            if (self.peaks[i].length < criteria.length
                or self.peaks[i].height < criteria.height
                or self.peaks[i].maxslope < criteria.maxslope
                or self.peaks[i].meanslope < criteria.meanslope
                or self.peaks[i].curspeed < criteria.curspeed
                or self.peaks[i].freeze > criteria.freeze):
                self.peaks.pop(i)
                i -= 1
            i += 1
        
        self.freq = len(self.peaks)
        self.length_mean = sum([self.peaks[i].length for i in range(self.freq)]) / Decimal(self.freq)
        self.height_sum = sum([self.peaks[i].height for i in range(self.freq)])
        self.height_mean = self.height_sum / Decimal(self.freq)
        self.maxslope_mean = sum([self.peaks[i].maxslope for i in range(self.freq)]) / Decimal(self.freq)
        self.meanslope_mean = sum([self.peaks[i].meanslope for i in range(self.freq)]) / Decimal(self.freq)

    def write_peaks(self, name, header):
        with open(name, 'w') as csvfile:
            for word in header:
                csvfile.write(word + ', ')
            csvfile.write('\n')
            for i in range(self.freq):
                data = [str(self.peaks[i].pos), str(self.peaks[i].length),
                        str(self.peaks[i].height), str(self.peaks[i].maxslope),
                        str(self.peaks[i].meanslope), str(self.peaks[i].curspeed),
                        str(self.peaks[i].freeze)]
                for datum in data:
                    csvfile.write(datum + ', ')
                csvfile.write('\n')

class splitpn:
    def __init__(self, inputlist):
        self.b = inputlist
        self.p = []
        self.n = []
        l = len(inputlist)
        for i in range(l):
            if inputlist[i] > 0:
                self.p.append(Decimal(inputlist[i]))
                self.n.append(Decimal(0))
            else:
                self.p.append(Decimal(0))
                self.n.append(Decimal(abs(inputlist[i])))

if accel_analysis:
    
    accel = [0 for i in range(l)]
    for i in range(2, l):
        accel[i] = (speed_avg[i] - speed_avg[i - 1]) * fps
    
    accel_avg = runavg(accel, 2, l)
    
    accel_criteria = peak_data(0, accel_min_dur, accel_min_total, accel_min_max, accel_min_mean, 0, 1)
    accel_avg = gen_peaks(accel_avg, accel_cutoff, accel_criteria)
    
    header = ['Position (frame)', 'Duration (frame(s))', 'Speed change (mm/s)', 'Maximum accleration (mm/s2)',
              'Mean acceleration (mm/s2)', 'Current speed (mm/s)', 'Freeze (0/1)']
    accel_avg.write_peaks(videoname + '_accel_peaks_data.csv', header)
    
    analysis.update({
        'Number of accelerations per sec': Decimal(accel_avg.freq) / Decimal(l // fps),
        'Mean acceleration duration (s)': Decimal(accel_avg.length_mean) / Decimal(fps),
        'Mean peak acceleration (mm/s2)': accel_avg.maxslope_mean,
        'Mean acceleration (mm/s2)': accel_avg.meanslope_mean,
    })
    
    accel_criteria.freeze = 0
    accel_avg_temp = gen_peaks(accel_avg.b, accel_cutoff, accel_criteria)

    analysis.update({
        'Number of accelerations per sec during active mvt': Decimal(accel_avg_temp.freq) / Decimal(l // fps - total_freeze_time),
        'Mean acceleration duration during active mvt (s)': Decimal(accel_avg_temp.length_mean) / Decimal(fps),
        'Mean peak acceleration during active mvt (mm/s2)': accel_avg_temp.maxslope_mean,
        'Mean acceleration during active mvt (mm/s2)': accel_avg_temp.meanslope_mean
    })

    if plot_vtgraph:
        vt_graph = np.zeros((1000, l, 3), np.uint8)
        for i in range(l):
            cv.circle(vt_graph, (i, 1000 - int(round(speed_avg[i]))), 1, (255, 255, 255), -1)
        for i in range(accel_avg.freq):
            for j in range(accel_avg.peaks[i].pos, accel_avg.peaks[i].pos + accel_avg.peaks[i].length):
                cv.circle(vt_graph, (j, 1000 - int(round(speed_avg[j]))), 1, (0, 127, 255), -1)
        cv.imwrite(videoname + '_vt_graph.png', vt_graph)

with open(videoname + '_kinematics_data.csv', 'w') as csvfile:
    if accel_analysis:
        csvfile.write('X' + ', ' + 'Y' + ', ' + 'Distance' + ', ' + 'Freeze' + ', '
                   + 'Speed' + ', ' + 'Speed avg' + ', ' + 'Accel' + ', ' + 'Accel avg' + '\n')
        for i in range(l):
            csvfile.write(str(com[i][0]) + ', ' + str(com[i][1]) + ', ' + str(cdist[i]) + ', ' + str(freeze[i]) + ', '
                       + str(speed[i]) + ', ' + str(speed_avg[i]) + ', ' + str(accel[i]) + ', ' + str(accel_avg.b[i]) + '\n')
    else:
        csvfile.write('X' + ', ' + 'Y' + ', ' + 'Distance' + ', ' + 'Freeze' + ', ' + 'Speed' + '\n')
        for i in range(l):
            csvfile.write(str(com[i][0]) + ', ' + str(com[i][1]) + ', ' + str(cdist[i]) + ', ' + str(freeze[i]) + ', ' + str(speed[i]) + '\n')

#file.write('Thigmotaxis' + ', ' + str(Decimal(sum(distancetowall)) / Decimal(l)) + '\n')

if spine_analysis:
    
    turn_avg = [0 for i in range(l)]
    headtotail_angle_avg = [0 for i in range(l)]
    headtotail_angle_change_avg = [0 for i in range(l)]
    total_abs_angle_avg = [0 for i in range(l)]
    total_abs_angle_change_avg = [0 for i in range(l)]
    
    with open(videoname + '_converted_spine_turns_and_angles.csv', 'r') as csvfile:
        data = [[number for number in frame] for frame in csv.reader(csvfile)]
        data.pop(0)
        for i in range(l):
            turn_avg[i] = Decimal(data[i][3])
            headtotail_angle_avg[i] = Decimal(data[i][15])
            headtotail_angle_change_avg[i] = Decimal(data[i][17])
            total_abs_angle_avg[i] = Decimal(data[i][19])
            total_abs_angle_change_avg[i] = Decimal(data[i][21])
   
    turn_criteria = peak_data(0, turn_min_dur, turn_min_amp, turn_min_max, turn_min_mean, 0, 0)
    turn_avg = splitpn(turn_avg)
    turn_avg.p = gen_peaks(turn_avg.p, turn_cutoff, turn_criteria)
    turn_avg.n = gen_peaks(turn_avg.n, turn_cutoff, turn_criteria)
    
    header = ['Position (frame)', 'Duration (frame(s))', 'Direction change (rad)', 'Maximum angular velocity (rad/s)',
              'Mean angular velocity (rad/s)', 'Current speed (mm/s)', 'Freeze (0/1)']
    turn_avg.p.write_peaks(videoname + '_turn_clockwise_peaks_data.csv', header)
    turn_avg.n.write_peaks(videoname + '_turn_anticlockwise_peaks_data.csv', header)
    
    if plot_turngraph:
        turn_graph = np.zeros((2000, l, 3), np.uint8)
        for i in range(l):
            cv.circle(turn_graph, (i, 1000 - int(round(turn_avg.b[i] * 10))), 1, (255, 255, 255), -1)
        for i in range(turn_avg.p.freq):
            for j in range(turn_avg.p.peaks[i].pos, turn_avg.p.peaks[i].pos + turn_avg.p.peaks[i].length):
                cv.circle(turn_graph, (j, 1000 - int(round(turn_avg.b[j] * 10))), 1, (0, 255, 0), -1)
        for i in range(turn_avg.n.freq):
            for j in range(turn_avg.n.peaks[i].pos, turn_avg.n.peaks[i].pos + turn_avg.n.peaks[i].length):
                cv.circle(turn_graph, (j, 1000 - int(round(turn_avg.b[j] * 10))), 1, (0, 127, 255), -1)
        cv.imwrite(videoname + '_turn_graph.png', turn_graph)
    
    turn_avg_p_freq = Decimal(turn_avg.p.freq) / Decimal(l // fps - total_freeze_time) * 60
    turn_avg_n_freq = Decimal(turn_avg.n.freq) / Decimal(l // fps - total_freeze_time) * 60
    turn_avg_p_amp_total = Decimal(turn_avg.p.height_sum) * 180 / pi / Decimal(l // fps - total_freeze_time) * 60
    turn_avg_n_amp_total = Decimal(turn_avg.n.height_sum) * 180 / pi / Decimal(l // fps - total_freeze_time) * 60
    
    analysis.update({
        'Number of clockwise turn per min during active mvt': turn_avg_p_freq,
        'Number of anticlockwise turn per min during active mvt': turn_avg_n_freq,
        'Turn number preference (positive for clockwise)': cal_bias(turn_avg_p_freq, turn_avg_n_freq),
        'Total clockwise turn per min during active mvt': turn_avg_p_amp_total,
        'Total anticlockwise turn per min during active mvt': turn_avg_n_amp_total,
        'Turn amplitude preference (positive for clockwise)': cal_bias(turn_avg_p_amp_total, turn_avg_n_amp_total),
        'Mean clockwise angular velocity (deg/s)': Decimal(turn_avg.p.meanslope_mean) * 180 / pi,
        'Mean anticlockwise angular velocity (deg/s)': Decimal(turn_avg.n.meanslope_mean) * 180 / pi,
        'Turn angular velocity preference (positive for clockwise)': cal_bias(turn_avg.p.meanslope_mean, turn_avg.n.meanslope_mean),
        'Meandering (deg/mm)': Decimal(turn_avg.p.height_sum + turn_avg.n.height_sum) * 180 / pi / Decimal(total_distance)
    })
    
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
    
    angle_criteria = peak_data(0, angle_min_dur, angle_min_amp, angle_min_max, angle_min_mean, 0, 1)
    total_abs_angle_change_avg = gen_peaks(total_abs_angle_change_avg, angle_cutoff, angle_criteria)
    
    header = ['Position (frame)', 'Duration (frame(s))', 'Tail bend amplitude (rad)', 'Maximum angular velocity (rad/s)',
              'Mean angular velocity (rad/s)', 'Current speed (mm/s)', 'Freeze (0/1)']
    total_abs_angle_change_avg.write_peaks(videoname + '_total_abs_angle_peaks_data.csv', header)
    
    bend6_low_dur = 0
    bend6_med_dur = 0
    bend6_hi_dur = 0
    bend6_low_freq = 0
    bend6_med_freq = 0
    bend6_hi_freq = 0
    bend6_low_bend = 0
    bend6_med_bend = 0
    bend6_hi_bend = 0
    for i in range(total_abs_angle_change_avg.freq):
        if total_abs_angle_change_avg.peaks[i].curspeed >= 5 and total_abs_angle_change_avg.peaks[i].curspeed < 20:
            bend6_low_freq += 1
            bend6_low_bend += total_abs_angle_change_avg.peaks[i].height
            bend6_low_dur += total_abs_angle_change_avg.peaks[i].length
        elif total_abs_angle_change_avg.peaks[i].curspeed >= 20 and total_abs_angle_change_avg.peaks[i].curspeed < 40:
            bend6_med_freq += 1
            bend6_med_bend += total_abs_angle_change_avg.peaks[i].height
            bend6_med_dur += total_abs_angle_change_avg.peaks[i].length
        elif total_abs_angle_change_avg.peaks[i].curspeed >= 40:
            bend6_hi_freq += 1
            bend6_hi_bend += total_abs_angle_change_avg.peaks[i].height
            bend6_hi_dur += total_abs_angle_change_avg.peaks[i].length
    
    bend6_low_bend = bend6_low_bend * 180 / pi
    bend6_med_bend = bend6_med_bend * 180 / pi
    bend6_hi_bend = bend6_hi_bend * 180 / pi
    bend6_mean = (bend6_low_bend + bend6_med_bend + bend6_hi_bend) / Decimal(total_abs_angle_change_avg.freq)
    
    analysis.update({
        '[total abs angle] Tail beat frequency (Hz)': Decimal(total_abs_angle_change_avg.freq) / Decimal(l // fps - total_freeze_time),
        '[total abs angle] Low speed tail beat frequency (Hz)': Decimal(bend6_low_freq) / low_dur,
        '[total abs angle] Medium speed tail beat frequency (Hz)': Decimal(bend6_med_freq) / med_dur,
        '[total abs angle] High speed tail beat frequency (Hz)': Decimal(bend6_hi_freq) / hi_dur,
        '[total abs angle] Mean tail bend amplitude (deg)': bend6_mean,
        '[total abs angle] Mean low speed tail bend amplitude (deg)': bend6_low_bend / Decimal(bend6_low_freq) if bend6_low_freq != 0 else 0,
        '[total abs angle] Mean medium speed tail bend amplitude (deg)': bend6_med_bend / Decimal(bend6_med_freq) if bend6_med_freq != 0 else 0,
        '[total abs angle] Mean high speed tail bend amplitude (deg)': bend6_hi_bend / Decimal(bend6_hi_freq) if bend6_hi_freq != 0 else 0
    })
    
    headtotail_angle_avg = splitpn(headtotail_angle_avg)
    headtotail_angle_change_p = [0]
    headtotail_angle_change_n = [0]
    for i in range(1, l):
        headtotail_angle_change_p.append((headtotail_angle_avg.p[i] - headtotail_angle_avg.p[i - 1]) * fps)
        headtotail_angle_change_n.append((headtotail_angle_avg.n[i] - headtotail_angle_avg.n[i - 1]) * fps)
    headtotail_angle_change_avg_p = runavg(headtotail_angle_change_p, 1, l)
    headtotail_angle_change_avg_n = runavg(headtotail_angle_change_n, 1, l)
    
    headtotail_angle_change_avg_p = gen_peaks(headtotail_angle_change_avg_p, angle_cutoff, angle_criteria)
    headtotail_angle_change_avg_n = gen_peaks(headtotail_angle_change_avg_n, angle_cutoff, angle_criteria)
    
    headtotail_angle_change_avg_p.write_peaks(videoname + '_headtotail_angle_clockwise_peaks_data.csv', header)
    headtotail_angle_change_avg_n.write_peaks(videoname + '_headtotail_angle_anticlockwise_peaks_data.csv', header)
    
    bend0p_low_dur = 0
    bend0p_med_dur = 0
    bend0p_hi_dur = 0
    bend0p_low_freq = 0
    bend0p_med_freq = 0
    bend0p_hi_freq = 0
    bend0p_low_bend = 0
    bend0p_med_bend = 0
    bend0p_hi_bend = 0
    for i in range(headtotail_angle_change_avg_p.freq):
        if headtotail_angle_change_avg_p.peaks[i].curspeed >= 5 and headtotail_angle_change_avg_p.peaks[i].curspeed < 20:
            bend0p_low_freq += 1
            bend0p_low_bend += headtotail_angle_change_avg_p.peaks[i].height
            bend0p_low_dur += headtotail_angle_change_avg_p.peaks[i].length
        elif headtotail_angle_change_avg_p.peaks[i].curspeed >= 20 and headtotail_angle_change_avg_p.peaks[i].curspeed < 40:
            bend0p_med_freq += 1
            bend0p_med_bend += headtotail_angle_change_avg_p.peaks[i].height
            bend0p_med_dur += headtotail_angle_change_avg_p.peaks[i].length
        elif headtotail_angle_change_avg_p.peaks[i].curspeed >= 40:
            bend0p_hi_freq += 1
            bend0p_hi_bend += headtotail_angle_change_avg_p.peaks[i].height
            bend0p_hi_dur += headtotail_angle_change_avg_p.peaks[i].length
    
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
    for i in range(headtotail_angle_change_avg_n.freq):
        if headtotail_angle_change_avg_n.peaks[i].curspeed >= 5 and headtotail_angle_change_avg_n.peaks[i].curspeed < 20:
            bend0n_low_freq += 1
            bend0n_low_bend += headtotail_angle_change_avg_n.peaks[i].height
            bend0n_low_dur += headtotail_angle_change_avg_n.peaks[i].length
        elif headtotail_angle_change_avg_n.peaks[i].curspeed >= 20 and headtotail_angle_change_avg_n.peaks[i].curspeed < 40:
            bend0n_med_freq += 1
            bend0n_med_bend += headtotail_angle_change_avg_n.peaks[i].height
            bend0n_med_dur += headtotail_angle_change_avg_n.peaks[i].length
        elif headtotail_angle_change_avg_n.peaks[i].curspeed >= 40:
            bend0n_hi_freq += 1
            bend0n_hi_bend += headtotail_angle_change_avg_n.peaks[i].height
            bend0n_hi_dur += headtotail_angle_change_avg_n.peaks[i].length
    
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
    
    bend0_mean = (bend0p_bend + bend0n_bend) / Decimal(headtotail_angle_change_avg_p.freq + headtotail_angle_change_avg_n.freq)
    
    if plot_anglegraph:
    
        bend6_graph = np.zeros((2000, l, 3), np.uint8)
        for i in range(l):
            cv.circle(bend6_graph, (i + 1, 1000 - int(round(total_abs_angle_avg[i] * 10))), 1, (255, 255, 255), -1)
        for i in range(total_abs_angle_change_avg.freq):
            for j in range(total_abs_angle_change_avg.peaks[i].pos, total_abs_angle_change_avg.peaks[i].pos + total_abs_angle_change_avg.peaks[i].length):
                cv.circle(bend6_graph, (j + 1, 1000 - int(round(total_abs_angle_avg[j] * 10))), 1, (0, 127, 255), -1)
        cv.imwrite(videoname + '_total_abs_angle_graph.png', bend6_graph)
        
        bend0_graph = np.zeros((2000, l, 3), np.uint8)
        for i in range(l):
            cv.circle(bend0_graph, (i + 1, 1000 - int(round(headtotail_angle_avg.b[i] * 10))), 1, (255, 255, 255), -1)
        for i in range(headtotail_angle_change_avg_p.freq):
            for j in range(headtotail_angle_change_avg_p.peaks[i].pos, headtotail_angle_change_avg_p.peaks[i].pos + headtotail_angle_change_avg_p.peaks[i].length):
                cv.circle(bend0_graph, (j + 1, 1000 - int(round(headtotail_angle_avg.b[j] * 10))), 1, (0, 255, 0), -1)
        for i in range(headtotail_angle_change_avg_n.freq):
            for j in range(headtotail_angle_change_avg_n.peaks[i].pos, headtotail_angle_change_avg_n.peaks[i].pos + headtotail_angle_change_avg_n.peaks[i].length):
                cv.circle(bend0_graph, (j + 1, 1000 - int(round(headtotail_angle_avg.b[j] * 10))), 1, (0, 127, 255), -1)
        cv.imwrite(videoname + '_headtotail_angle_graph.png', bend0_graph)
    
    analysis.update({
        '[head-to-tail] Tail beat frequency (Hz)': Decimal(headtotail_angle_change_avg_p.freq + headtotail_angle_change_avg_n.freq) / Decimal(l // fps - total_freeze_time),
        '[head-to-tail] Low speed tail beat frequency (Hz)': Decimal(bend0_low_freq) / low_dur,
        '[head-to-tail] Medium speed tail beat frequency (Hz)': Decimal(bend0_med_freq) / med_dur,
        '[head-to-tail] High speed tail beat frequency (Hz)': Decimal(bend0_hi_freq) / hi_dur,
        '[head-to-tail] Mean tail bend amplitude (deg)': bend0_mean,
        '[head-to-tail] Mean low speed tail bend amplitude (deg)': bend0_low_bend / Decimal(bend0_low_freq) if bend0_low_freq != 0 else 0,
        '[head-to-tail] Mean medium speed tail bend amplitude (deg)': bend0_med_bend / Decimal(bend0_med_freq) if bend0_med_freq != 0 else 0,
        '[head-to-tail] Mean high speed tail bend amplitude (deg)': bend0_hi_bend / Decimal(bend0_hi_freq) if bend0_hi_freq != 0 else 0,
        'Tail bend amplitude preference (positive for clockwise)': cal_bias(bend0p_bend, bend0n_bend),
        'Tail beat frequency preference (positive for clockwise)': cal_bias(headtotail_angle_change_avg_p.freq, headtotail_angle_change_avg_n.freq)
    })

with open(videoname + '_analysis.csv', 'w') as csvfile:
    for key in analysis:
        csvfile.write(key + ', ' + str(analysis[key]) + '\n')

cv.destroyAllWindows()
etime = time.time()
print('Runtime: ' + str(etime - stime))
