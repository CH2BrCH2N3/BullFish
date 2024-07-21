import os
import csv
import matplotlib.pyplot as plt
import math
from decimal import Decimal
from statistics import median
pi = Decimal(math.pi)

import cv2 as cv
import numpy as np

if not os.path.exists('bullfish_analysis_settings.csv'):
    with open('bullfish_analysis_settings.csv', 'w') as f:
        headers = ['tank_x', 'tank_y', 'accel_cutoff', 'accel_min_dur', 'accel_min_maxima', 'accel_min_auc',
                   'spine_analysis', 'turn_cutoff', 'turn_min_dur', 'turn_min_maxima', 'turn_min_auc', 'angle_cutoff',
                   'angle_min_dur', 'angle_min_maxima', 'angle_min_auc', 'amplitude_cutoff', 'amplitude_min_dur',
                   'amplitude_min_maxima', 'amplitude_min_auc']
        for word in headers:
            f.write(word + '\n')
    print('Set settings first')
    from sys import exit
    exit()
else:
    with open('bullfish_analysis_settings.csv', 'r') as f:
        settings = {row[0]: row[1] for row in csv.reader(f)}
    tank_x = Decimal(settings['tank_x'])
    tank_y = Decimal(settings['tank_y'])
    accel_cutoff = Decimal(settings['accel_cutoff'])
    accel_min_dur = Decimal(settings['accel_min_dur'])
    accel_min_maxima = Decimal(settings['accel_min_maxima'])
    accel_min_auc = Decimal(settings['accel_min_auc'])
    spine_analysis = bool(int(settings['spine_analysis']))
    if spine_analysis:
        turn_cutoff = Decimal(settings['turn_cutoff']) * pi / 180
        turn_min_dur = Decimal(settings['turn_min_dur'])
        turn_min_maxima = Decimal(settings['turn_min_maxima']) * pi / 180
        turn_min_auc = Decimal(settings['turn_min_auc']) * pi / 180
        angle_cutoff = Decimal(settings['angle_cutoff']) * pi / 180
        angle_min_dur = Decimal(settings['angle_min_dur'])
        angle_min_maxima = Decimal(settings['angle_min_maxima']) * pi / 180
        angle_min_auc = Decimal(settings['angle_min_auc']) * pi / 180
        amplitude_cutoff = Decimal(settings['amplitude_cutoff'])
        amplitude_min_dur = Decimal(settings['amplitude_min_dur'])
        amplitude_min_maxima = Decimal(settings['amplitude_min_maxima'])
        amplitude_min_auc = Decimal(settings['amplitude_min_auc'])

def pyth(point1, point2):
    return Decimal.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def cal_direction(point1, point2): #from point1 to point2
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return Decimal(0)
    else:
        return Decimal(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))

def cal_direction_change(s1, s2): #from s1 to s2
    direction_change = Decimal(s2) - Decimal(s1)
    if direction_change > pi:
        return direction_change - pi * 2
    elif direction_change <= -pi:
        return direction_change + pi * 2
    else:
        return direction_change

def cal_preference(p, n):
    try:
        return (p - n) / (p + n)
    except:
        print('cal_preference Error')
        return 0

class peak_data:
    def __init__(self, pos, length, maxima, auc):
        self.pos = pos
        self.length = length
        self.maxima = maxima
        self.auc = auc

class peak_analysis:
    
    def __init__(self, inputlist):
        self.original_list = inputlist
        self.list = inputlist
    
    def downsampling(self, factor):
        list_len = len(self.list)
        for i in range(0, list_len - factor, factor):
            for j in range(i + 1, i + factor):
                self.list[j] = (self.list[i] * (i + factor - j) + self.list[i + factor] * (j - i)) / factor
        for i in range(list_len - factor + 1, list_len - 1):
            self.list[i] = (self.list[list_len - factor] * (list_len - 1 - i) + self.list[list_len - 1] * (i - (list_len - factor))) / (list_len - 1 - (list_len - factor))
    
    def runavg(self, start, end, window):

        temp_list = self.original_list[start:end]
        self.list = [0 for i in range(end)]
        for i in range(window // 2):
            temp_list.insert(0, self.original_list[start])
        for i in range(window // 2):
            temp_list.append(self.original_list[end - 1])
        for i in range(start, end):
            self.list[i] = sum(temp_list[(i - start):(i - start + window)]) / window
            
        self.positive_list = [(item if item > 0 else 0) for item in self.list]
        self.negative_list = [(-item if item < 0 else 0) for item in self.list]
    
    def find_peaks(self, positive_cutoff, negative_cutoff, criteria):
        
        self.positive_peaks = []
        i = 1
        while i < l:
            if self.positive_list[i] > positive_cutoff:
                j = i
                maxima = 0
                auc = 0
                while j < l and self.positive_list[j] > positive_cutoff:
                    auc += self.positive_list[j]
                    if self.positive_list[j] > maxima:
                        maxima = self.positive_list[j]
                    j += 1
                if j >= l:
                    break
                auc = auc / fps
                self.positive_peaks.append(peak_data(i, j - i, maxima, auc))
                i = j
            i += 1
        
        i = 0
        while i < len(self.positive_peaks):
            if (self.positive_peaks[i].length < criteria.length
                or self.positive_peaks[i].maxima < criteria.maxima
                or self.positive_peaks[i].auc < criteria.auc):
                self.positive_peaks.pop(i)
                i -= 1
            i += 1
        
        self.positive_count = len(self.positive_peaks)
        self.positive_length_sum = sum((self.positive_peaks[i].length for i in range(self.positive_count)))
        self.positive_length_mean = self.positive_length_sum / Decimal(self.positive_count) if self.positive_count > 0 else 0
        self.positive_length_max = max((self.positive_peaks[i].length for i in range(self.positive_count))) if self.positive_count > 0 else 0
        self.positive_maxima_sum = sum((self.positive_peaks[i].maxima for i in range(self.positive_count)))
        self.positive_maxima_mean = self.positive_maxima_sum / Decimal(self.positive_count) if self.positive_count > 0 else 0
        self.positive_maxima_max = max((self.positive_peaks[i].maxima for i in range(self.positive_count))) if self.positive_count > 0 else 0
        self.positive_auc_sum = sum((self.positive_peaks[i].auc for i in range(self.positive_count)))
        self.positive_auc_mean = self.positive_auc_sum / Decimal(self.positive_count) if self.positive_count > 0 else 0
        self.positive_auc_max = max((self.positive_peaks[i].auc for i in range(self.positive_count))) if self.positive_count > 0 else 0
        
        self.negative_peaks = []
        i = 1
        while i < l:
            if self.negative_list[i] > negative_cutoff:
                j = i
                maxima = 0
                auc = 0
                while j < l and self.negative_list[j] > negative_cutoff:
                    auc += self.negative_list[j]
                    if self.negative_list[j] > maxima:
                        maxima = self.negative_list[j]
                    j += 1
                if j >= l:
                    break
                auc = auc / fps
                self.negative_peaks.append(peak_data(i, j - i, maxima, auc))
                i = j
            i += 1
        
        i = 0
        while i < len(self.negative_peaks):
            if (self.negative_peaks[i].length < criteria.length
                or self.negative_peaks[i].maxima < criteria.maxima
                or self.negative_peaks[i].auc < criteria.auc):
                self.negative_peaks.pop(i)
                i -= 1
            i += 1
        
        self.negative_count = len(self.negative_peaks)
        self.negative_length_sum = sum((self.negative_peaks[i].length for i in range(self.negative_count)))
        self.negative_length_mean = self.negative_length_sum / Decimal(self.negative_count) if self.negative_count > 0 else 0
        self.negative_length_max = max((self.negative_peaks[i].length for i in range(self.negative_count))) if self.negative_count > 0 else 0
        self.negative_maxima_sum = sum((self.negative_peaks[i].maxima for i in range(self.negative_count)))
        self.negative_maxima_mean = self.negative_maxima_sum / Decimal(self.negative_count) if self.negative_count > 0 else 0
        self.negative_maxima_max = max((self.negative_peaks[i].maxima for i in range(self.negative_count))) if self.negative_count > 0 else 0
        self.negative_auc_sum = sum((self.negative_peaks[i].auc for i in range(self.negative_count)))
        self.negative_auc_mean = self.negative_auc_sum / Decimal(self.negative_count) if self.negative_count > 0 else 0
        self.negative_auc_max = max((self.negative_peaks[i].auc for i in range(self.negative_count))) if self.negative_count > 0 else 0
        
        self.count = self.positive_count + self.negative_count
        self.length_sum = self.positive_length_sum + self.negative_length_sum
        self.length_mean = self.length_sum / Decimal(self.count) if self.count > 0 else 0
        self.length_max = max(self.positive_length_max, self.negative_length_max)
        self.maxima_sum = self.positive_maxima_sum + self.negative_maxima_sum
        self.maxima_mean = self.maxima_sum / Decimal(self.count) if self.count > 0 else 0
        self.maxima_max = max(self.positive_maxima_max, self.negative_maxima_max)
        self.auc_sum = self.positive_auc_sum + self.negative_auc_sum
        self.auc_mean = self.auc_sum / Decimal(self.count) if self.count > 0 else 0
        self.auc_max = max(self.positive_auc_max, self.negative_auc_max)
        
        self.peaks = [0 for i in range(self.count)]
        i = 0
        pi = 0
        ni = 0
        while i < self.count:
            if pi >= self.positive_count:
                self.peaks[i] = self.negative_peaks[ni]
                self.peaks[i].maxima = -self.peaks[i].maxima
                self.peaks[i].auc = -self.peaks[i].auc
                ni += 1
                i += 1
                continue
            if ni >= self.negative_count:
                self.peaks[i] = self.positive_peaks[pi]
                pi += 1
                i += 1
                continue
            if self.positive_peaks[pi].pos < self.negative_peaks[ni].pos:
                self.peaks[i] = self.positive_peaks[pi]
                pi += 1
                i += 1
            else:
                self.peaks[i] = self.negative_peaks[ni]
                self.peaks[i].maxima = -self.peaks[i].maxima
                self.peaks[i].auc = -self.peaks[i].auc
                ni += 1
                i += 1
        
        '''
        self.count_a = 0
        self.length_sum_a = 0
        self.height_sum_a = 0
        self.maxslope_sum_a = 0
        for i in range(self.count):
            if self.peaks[i].freeze == 0:
                self.count_a += 1
                self.length_sum_a += self.peaks[i].length
                self.height_sum_a += self.peaks[i].height
                self.maxslope_sum_a += self.peaks[i].maxslope
        self.length_mean_a = self.length_sum_a / Decimal(self.count_a) if self.count_a > 0 else 0
        self.height_mean_a = self.height_sum_a / Decimal(self.count_a) if self.count_a > 0 else 0
        self.maxslope_mean_a = self.maxslope_sum_a / Decimal(self.count_a) if self.count_a > 0 else 0
        '''
    def write_peaks(self, name):
        with open(name, 'w') as f:
            for word in ['pos', 'length', 'maxima', 'auc']:
                f.write(word + ', ')
            f.write('\n')
            for i in range(self.count):
                data = [str(self.peaks[i].pos), str(self.peaks[i].length),
                        str(self.peaks[i].maxima), str(self.peaks[i].auc)]
                for datum in data:
                    f.write(datum + ', ')
                f.write('\n')
    
    def plot_peaks(self, axes, list_toplot, color_list, color_positive_peak, color_negative_peak, factor=1):
        for item in list_toplot:
            item *= Decimal(factor)
        if color_list != None:
            axes.plot(list_toplot, color=color_list)
        if color_positive_peak != None:
            for peak in self.positive_peaks:
                axes.plot([i for i in range(peak.pos, peak.pos + peak.length)], list_toplot[peak.pos:(peak.pos + peak.length)], color=color_positive_peak)
        if color_negative_peak != None:
            for peak in self.negative_peaks:
                axes.plot([i for i in range(peak.pos, peak.pos + peak.length)], list_toplot[peak.pos:(peak.pos + peak.length)], color=color_negative_peak)
        
for file in os.listdir('.'):
     
    filename = os.fsdecode(file)
    filename_split = os.path.splitext(filename)
    supported_formats = ['.avi', '.mp4']
    if filename_split[1] not in supported_formats:
        continue
    videoname = filename_split[0]
    path = './' + videoname
    if not os.path.isfile(path + '/' + videoname + '_metadata.csv'):
        print('Metadata missing for ' + videoname)
        continue
    print('\nProcessing ' + filename)
    with open(path + '/' + videoname + '_metadata.csv', 'r') as f:
        metadata = {row[0]: row[1] for row in csv.reader(f)}
        fps = Decimal(metadata['fps'])
        video_start = int(metadata['video_start'])
        video_end = int(metadata['video_end'])
        swimarea_tlx = int(metadata['swimarea_tlx'])
        swimarea_x = int(metadata['swimarea_x'])
        swimarea_tly = int(metadata['swimarea_tly'])
        swimarea_y = int(metadata['swimarea_y'])
    l = video_end - video_start
    if swimarea_x > swimarea_y:
        ratio = tank_x / Decimal(swimarea_x)
    else:
        ratio = tank_y / Decimal(swimarea_y)
    with open(path + '/' + videoname + '_fishlength.csv', 'r') as f:
        fish_lengths = [cell for cell in csv.reader(f)]
    fish_length = median([Decimal(length[0]) for length in fish_lengths])
    
    #annotated = cv.VideoWriter(path + '/' + videoname + '_aa.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (200, 700), 0)
    
    with open(path + '/' + videoname + '_cen.csv', 'r') as f:
        cen = [[cell for cell in row] for row in csv.reader(f)]
        cen.pop(0)
        for i in range(l):
            cen[i] = (Decimal(cen[i][0]) * ratio, Decimal(cen[i][1]) * ratio)
    
    cdists = [0 for i in range(l)]
    speeds = [0 for i in range(l)]
    for i in range(1, l):
        cdists[i] = pyth(cen[i], cen[i - 1])
        speeds[i] = cdists[i] * fps
    total_distance = sum(cdists)
    total_time = Decimal(l) / fps
    speed_avg = total_distance / total_time
    
    speeds = peak_analysis(speeds)
    speeds.runavg(1, l, 5)
    max_speed = max(speeds.list)

    freeze = [0 for i in range(l)]
    for i in range(int(fps * 3), l):
        cdist1 = pyth(cen[int(i - fps * 2)], cen[int(i - fps * 3)])
        cdist2 = pyth(cen[int(i - fps)], cen[int(i - fps * 2)])
        cdist3 = pyth(cen[i], cen[int(i - fps)])
        if cdist1 < 1 and cdist2 < 1 and cdist3 < 1:
            for j in range(int(i - fps * 3 + 1), i + 1):
                freeze[j] = 1
        elif cdist1 > 1 and cdist2 > 1 and cdist3 > 1:
            for j in range(int(i - fps * 2 + 1), i + 1):
                freeze[j] = 0
    
    total_freeze_time = Decimal(sum(freeze)) / fps
    freeze_percent = total_freeze_time / total_time * 100
    active_time = total_time - total_freeze_time
    active_speed = total_distance / active_time
    freeze_count = 0
    for i in range(2, l):
        if freeze[i] - freeze[i - 1] == 1:
            freeze_count += 1
    freeze_freq = Decimal(freeze_count) / total_time * 60
    
    cdist1s = [0 for i in range(l)]
    for i in range(1, l):
        start = max(1, round(i + 1 - fps / 2))
        end = min(l, round(i + 1 + fps / 2))
        cdist1s[i] = sum((cdists[j] for j in range(start, end))) * fps / Decimal(end - start)
    max_distance_1s = max(cdist1s)
    
    analysis = {
        'speed_avg': speed_avg,
        'max_speed': max_speed,
        'max_distance_1s': max_distance_1s,
        'active_speed': active_speed,
        'freeze_percent': freeze_percent,
        'freeze_count': freeze_count,
        'freeze_freq': freeze_freq
    }
    
    accels = [0 for i in range(l)]
    for i in range(2, l):
        accels[i] = (speeds.list[i] - speeds.list[i - 1]) * fps
    
    accel_criteria = peak_data(0, accel_min_dur * fps, accel_min_maxima, accel_min_auc)
    accels = peak_analysis(accels)
    accels.runavg(2, l, 5)
    accels.find_peaks(accel_cutoff, 99999999, accel_criteria)
    accels.write_peaks(path + '/' + videoname + '_accel_peaks_data.csv')
    
    accels_freq = Decimal(accels.positive_count) / total_time
    accels_mean_dur = Decimal(accels.positive_length_mean) / fps
    accels_mean_maxima = accels.positive_maxima_mean
    accels_mean_auc = accels.positive_auc_mean
    accels_max_dur = Decimal(accels.positive_length_max) / fps
    accels_max_maxima = accels.positive_maxima_max
    accels_max_auc = accels.positive_auc_max
    analysis.update({
        'accels_freq': accels_freq,
        'accels_mean_dur': accels_mean_dur,
        'accels_mean_maxima': accels_mean_maxima,
        'accels_mean_auc': accels_mean_auc,
        'accels_max_dur': accels_max_dur,
        'accels_max_maxima': accels_max_maxima,
        'accels_max_auc': accels_max_auc
    })
        
    if spine_analysis:
        
        spine_lens = [0 for i in range(l)]
        with open(path + '/' + videoname + '_spine.csv', 'r') as f:
            spines = [[cell for cell in row] for row in csv.reader(f)]
            spines.pop(0)
            for i in range(l):
                spine_lens[i] = int(spines[i][0])
                spine_temp = []
                for j in range(1, spine_lens[i] + 1):
                    spine_temp.append([Decimal(spines[i][j * 2 - 1]), Decimal(spines[i][j * 2])])
                spines[i] = spine_temp
        directions = [0 for i in range(l)]
        turns = [0 for i in range(l)]
        with open(path + '/' + videoname + '_direction.csv', 'r') as f:
            direction_temp = [[cell for cell in row] for row in csv.reader(f)]
            direction_temp.pop(0)
            for i in range(l):
                directions[i] = Decimal(direction_temp[i][0])
                turns[i] = Decimal(direction_temp[i][1])
        tails = [0 for i in range(l)]
        with open(path + '/' + videoname + '_headtail.csv', 'r') as f:
            headtail = [[cell for cell in row] for row in csv.reader(f)]
            headtail.pop(0)
            for i in range(l):
                tails[i] = [Decimal(headtail[i][2]), Decimal(headtail[i][3])]
        
        tail_dists = [0 for i in range(l)]
        for i in range(1, l):
            tail_dists[i] = pyth(tails[i], tails[i - 1])
        
        amplitudes = [0 for i in range(l)]
        for i in range(l):
            if spines[i][spine_lens[i] - 1][0] == spines[i][spine_lens[i] - 2][0]:
                amplitudes[i] = abs(tails[i][0] - spines[i][spine_lens[i] - 1][0]) * ratio
            else:
                m = (spines[i][spine_lens[i] - 1][1] - spines[i][spine_lens[i] - 2][1]) / (spines[i][spine_lens[i] - 1][0] - spines[i][spine_lens[i] - 2][0])
                c = spines[i][spine_lens[i] - 1][1] - m * spines[i][spine_lens[i] - 1][0]
                amplitudes[i] = abs(m * tails[i][0] - tails[i][1] + c) / Decimal.sqrt(m ** 2 + 1) * ratio
            
        spine_angles = [[] for i in range(l)]
        trunk_angles = [0 for i in range(l)]
        tail_angles = [0 for i in range(l)]
        trunk_curvs = [0 for i in range(l)]
        total_curvs = [0 for i in range(l)]
        spine_angles_filtered = [[] for i in range(l)]
        trunk_curvs_filtered = [0 for i in range(l)]
        total_curvs_filtered = [0 for i in range(l)]
        
        for i in range(l):
            
            spine_dirs = []
            for j in range(1, spine_lens[i]):
                spine_dirs.append(cal_direction(spines[i][j - 1], spines[i][j]))
            for j in range(2, spine_lens[i]):
                spine_angles[i].append(cal_direction_change(spine_dirs[j - 1], spine_dirs[j - 2]))
                trunk_angles[i] += spine_angles[i][j - 2]
            
            head_len = max(2, spine_lens[i] // 3)
            for j in range(spine_lens[i] - head_len - 2):
                tail_angles[i] += spine_angles[i][j]
            tail_dir = cal_direction(tails[i], spines[i][0])
            tail_angles[i] += cal_direction_change(spine_dirs[0], tail_dir)
            directiono = cal_direction(spines[i][spine_lens[i] - head_len], spines[i][spine_lens[i] - 1])
            tail_angles[i] += cal_direction_change(directiono, spine_dirs[spine_lens[i] - head_len - 2])
            if tail_angles[i] < 0:
                amplitudes[i] = -amplitudes[i]
            
            for j in range(spine_lens[i] - 2):
                trunk_curvs[i] += abs(spine_angles[i][j])
            if trunk_curvs[i] < 1:
                spine_dirs = [cal_direction(spines[i][0], spines[i][1])]
                if spine_lens[i] % 2 == 1:
                    spine_dirs.append(cal_direction(spines[i][1], spines[i][2]))
                    for j in range(4, spine_lens[i], 2):
                        spine_dirs.append(cal_direction(spines[i][j - 2], spines[i][j]))
                else:
                    for j in range(3, spine_lens[i], 2):
                        spine_dirs.append(cal_direction(spines[i][j - 2], spines[i][j]))
                for j in range(1, (spine_lens[i] + 1) // 2):
                    spine_angles_filtered[i].append(cal_direction_change(spine_dirs[j], spine_dirs[j - 1]))
                    trunk_curvs_filtered[i] += abs(spine_angles_filtered[i][j - 1])
            else:
                spine_angles_filtered[i] = list(spine_angles[i])
                trunk_curvs_filtered[i] = trunk_curvs[i]
            total_curvs_filtered[i] = trunk_curvs_filtered[i] + abs(cal_direction_change(spine_dirs[0], tail_dir))
        
        trunk_angles = peak_analysis(trunk_angles)
        trunk_angles.runavg(0, l, 5)
        trunk_positive_angles_change = [0 for i in range(l)]
        trunk_negative_angles_change = [0 for i in range(l)]
        tail_angles = peak_analysis(tail_angles)
        tail_angles.runavg(0, l, 5)
        tail_positive_angles_change = [0 for i in range(l)]
        tail_negative_angles_change = [0 for i in range(l)]
        amplitudes = peak_analysis(amplitudes)
        amplitudes.runavg(0, l, 5)
        positive_amplitudes_change = [0 for i in range(l)]
        negative_amplitudes_change = [0 for i in range(l)]
        trunk_curvs_filtered = peak_analysis(trunk_curvs_filtered)
        trunk_curvs_filtered.runavg(0, l, 5)
        trunk_curvs_change = [0 for i in range(l)]
        total_curvs_filtered = peak_analysis(total_curvs_filtered)
        total_curvs_filtered.runavg(0, l, 5)
        total_curvs_change = [0 for i in range(l)]
        for i in range(1, l):
            trunk_positive_angles_change[i] = (trunk_angles.positive_list[i] - trunk_angles.positive_list[i - 1]) * fps
            trunk_negative_angles_change[i] = (trunk_angles.negative_list[i] - trunk_angles.negative_list[i - 1]) * fps
            tail_positive_angles_change[i] = (tail_angles.positive_list[i] - tail_angles.positive_list[i - 1]) * fps
            tail_negative_angles_change[i] = (tail_angles.negative_list[i] - tail_angles.negative_list[i - 1]) * fps
            positive_amplitudes_change[i] = (amplitudes.positive_list[i] - amplitudes.positive_list[i - 1]) * fps
            negative_amplitudes_change[i] = (amplitudes.negative_list[i] - amplitudes.negative_list[i - 1]) * fps
            trunk_curvs_change[i] = (trunk_curvs_filtered.list[i] - trunk_curvs_filtered.list[i - 1]) * fps
            total_curvs_change[i] = (total_curvs_filtered.list[i] - total_curvs_filtered.list[i - 1]) * fps
        
        turn_criteria = peak_data(0, turn_min_dur * fps, turn_min_maxima, turn_min_auc)
        turns = peak_analysis(turns)
        turns.runavg(1, l, 5)
        turns.find_peaks(turn_cutoff, turn_cutoff, turn_criteria)
        turns.write_peaks(path + '/' + videoname + '_turn_peaks_data.csv')
        
        turn_per_min = Decimal(turns.count) / total_time * 60
        right_turn_per_min = Decimal(turns.positive_count) / total_time * 60
        left_turn_per_min = Decimal(turns.negative_count) / total_time * 60
        turn_count_preference = cal_preference(right_turn_per_min, left_turn_per_min)
        turn_angle_mean = turns.auc_mean
        right_turn_angle_mean = turns.positive_auc_mean
        left_turn_angle_mean = turns.negative_auc_mean
        turn_angle_preference = cal_preference(right_turn_angle_mean, left_turn_angle_mean)
        turn_speed_mean = turns.maxima_mean
        right_turn_speed_mean = turns.positive_maxima_mean
        left_turn_speed_mean = turns.negative_maxima_mean
        turn_speed_preference = cal_preference(right_turn_speed_mean, left_turn_speed_mean)
        meandering = turns.auc_sum / total_distance
        analysis.update({
            'turn_per_min': turn_per_min,
            'right_turn_per_min': right_turn_per_min,
            'left_turn_per_min': left_turn_per_min,
            'turn_count_preference': turn_count_preference,
            'turn_angle_mean': turn_angle_mean,
            'right_turn_angle_mean': right_turn_angle_mean,
            'left_turn_angle_mean': left_turn_angle_mean,
            'turn_angle_preference': turn_angle_preference,
            'turn_speed_mean': turn_speed_mean,
            'right_turn_speed_mean': right_turn_speed_mean,
            'left_turn_speed_mean': left_turn_speed_mean,
            'turn_speed_preference': turn_speed_preference,
            'meandering': meandering
        })
        
        angle_criteria = peak_data(0, angle_min_dur * fps, angle_min_maxima, angle_min_auc)
        trunk_positive_angles_change = peak_analysis(trunk_positive_angles_change)
        trunk_positive_angles_change.runavg(1, l, 5)
        trunk_negative_angles_change = peak_analysis(trunk_negative_angles_change)
        trunk_negative_angles_change.runavg(1, l, 5)
        trunk_positive_angles_change.find_peaks(angle_cutoff, angle_cutoff, angle_criteria)
        trunk_negative_angles_change.find_peaks(angle_cutoff, angle_cutoff, angle_criteria)
        #trunk_angles.write_peaks(path + '/' + videoname + '_trunk_angles_peaks_data.csv')
        tail_positive_angles_change = peak_analysis(tail_positive_angles_change)
        tail_positive_angles_change.runavg(1, l, 5)
        tail_negative_angles_change = peak_analysis(tail_negative_angles_change)
        tail_negative_angles_change.runavg(1, l, 5)
        tail_positive_angles_change.find_peaks(angle_cutoff, angle_cutoff, angle_criteria)
        tail_negative_angles_change.find_peaks(angle_cutoff, angle_cutoff, angle_criteria)
        #tail_angles.write_peaks(path + '/' + videoname + '_tail_angles_peaks_data.csv')
        trunk_curvs_change = peak_analysis(trunk_curvs_change)
        trunk_curvs_change.runavg(1, l, 5)
        trunk_curvs_change.find_peaks(angle_cutoff, angle_cutoff, angle_criteria)
        #trunk_curvs_change.write_peaks(path + '/' + videoname + '_trunk_curvs_peaks_data.csv')
        total_curvs_change = peak_analysis(total_curvs_change)
        total_curvs_change.runavg(1, l, 5)
        total_curvs_change.find_peaks(angle_cutoff, angle_cutoff, angle_criteria)
        total_curvs_change.write_peaks(path + '/' + videoname + '_total_curvs_peaks_data.csv')
        amplitude_criteria = peak_data(0, amplitude_min_dur * fps, amplitude_min_maxima, amplitude_min_auc)
        positive_amplitudes_change = peak_analysis(positive_amplitudes_change)
        positive_amplitudes_change.runavg(1, l, 5)
        negative_amplitudes_change = peak_analysis(negative_amplitudes_change)
        negative_amplitudes_change.runavg(1, l, 5)
        positive_amplitudes_change.find_peaks(amplitude_cutoff, amplitude_cutoff, amplitude_criteria)
        negative_amplitudes_change.find_peaks(amplitude_cutoff, amplitude_cutoff, amplitude_criteria)
        
        total_stride = trunk_curvs_change.positive_count
        stride_lengths = [0 for i in range(total_stride)]
        stride_current_speeds = [0 for i in range(total_stride)]
        stride_speeds = [0 for i in range(total_stride)]
        stride_accels = [0 for i in range(total_stride)]
        j = 0
        for i in range(0, total_stride - 1):
            start = trunk_curvs_change.positive_peaks[i].pos
            end = trunk_curvs_change.positive_peaks[i + 1].pos
            stride_lengths[i] = pyth(cen[start], cen[end])
            while j < accels.positive_count:
                if accels.positive_peaks[j].pos > start:
                    if accels.positive_peaks[j].pos < end:
                        stride_current_speeds[i] = cdist1s[start]
                        stride_speeds[i] = accels.positive_peaks[j].auc
                        stride_accels[i] = accels.positive_peaks[j].maxima
                    break
                else:
                    j += 1
        with open(path + '/' + videoname + '_trunk_curvs_peaks_data.csv', 'w') as f:
            for word in ['pos', 'length', 'maxima', 'auc', 'stride_lengths',
                         'stride_current_speeds', 'stride_speeds', 'stride_accels']:
                f.write(word + ', ')
            f.write('\n')
            for i in range(total_stride):
                data = [trunk_curvs_change.positive_peaks[i].pos, trunk_curvs_change.positive_peaks[i].length,
                        trunk_curvs_change.positive_peaks[i].maxima, trunk_curvs_change.positive_peaks[i].auc,
                        stride_lengths[i], stride_current_speeds[i],
                        stride_speeds[i], stride_accels[i]]
                for datum in data:
                    f.write(str(datum) + ', ')
                f.write('\n')
                
        stride_per_min = Decimal(total_stride) / total_time * 60
        stride_length_sum = 0
        stride_length_count = 0
        stride_speed_sum = 0
        stride_speed_count = 0
        stride_accel_sum = 0
        stride_accel_count = 0
        for i in range(total_stride):
            if stride_lengths[i] > 0:
                stride_length_sum += stride_lengths[i]
                stride_length_count += 1
            if stride_speeds[i] > 0:
                stride_speed_sum += stride_speeds[i]
                stride_speed_count += 1
            if stride_accels[i] > 0:
                stride_accel_sum += stride_accels[i]
                stride_accel_count += 1
        stride_length_mean = stride_length_sum / Decimal(stride_length_count)
        stride_speed_mean = stride_speed_sum / Decimal(stride_speed_count)
        stride_accel_mean = stride_accel_sum / Decimal(stride_accel_count)
        analysis.update({
            'stride_per_min': stride_per_min,
            'stride_length_mean': stride_length_mean,
            'stride_speed_mean': stride_speed_mean,
            'stride_accel_mean': stride_accel_mean
        })
        
        fig, ax = plt.subplots()
        trunk_positive_angles_change.plot_peaks(ax, trunk_angles.list, 'y', 'b', None)
        trunk_negative_angles_change.plot_peaks(ax, trunk_angles.list, None, 'r', None)
        tail_positive_angles_change.plot_peaks(ax, tail_angles.list, 'y', 'b', None)
        tail_negative_angles_change.plot_peaks(ax, tail_angles.list, None, 'r', None)
        plt.show()
        
        fig, ax = plt.subplots()
        trunk_curvs_change.plot_peaks(ax, trunk_curvs_filtered.list, 'y', 'b', None)
        total_curvs_change.plot_peaks(ax, total_curvs_filtered.list, 'y', 'b', None)
        plt.show()
        '''
        fig, (total_curvs_filtered_ax, tail_angles_ax, turn_ax, accel_ax) = plt.subplots(4, sharex=True)
        tail_positive_angles_change.plot_peaks(tail_angles_ax, tail_angles.list, 'y', 'b', None)
        tail_negative_angles_change.plot_peaks(tail_angles_ax, tail_angles.list, None, 'r', None)
        turns.plot_peaks(turn_ax, turns.list, 'y', 'b', 'r')
        accels.plot_peaks(accel_ax, speeds.list, 'y', 'b', 'r')
        plt.show()
        '''
        plt.figure()
        scatter_x = [peak.auc for peak in trunk_curvs_change.positive_peaks]
        scatter_y = stride_lengths
        '''
        scatter_x = []
        scatter_y = []
        for peaki in positive_amplitudes_change.positive_peaks:
            for peakj in accels.positive_peaks:
                if peakj.pos >= peaki.pos and peakj.pos < peaki.pos + peaki.length:
                    scatter_x.append(abs(peaki.auc))
                    scatter_y.append(peakj.auc)
                    continue
        for peaki in negative_amplitudes_change.positive_peaks:
            for peakj in accels.positive_peaks:
                if peakj.pos >= peaki.pos and peakj.pos < peaki.pos + peaki.length:
                    scatter_x.append(abs(peaki.auc))
                    scatter_y.append(peakj.auc)
                    continue
        '''
        plt.scatter(scatter_x, scatter_y)
        plt.show()
        '''
        for i in range(l):
            image = np.zeros((700,200), np.uint8)
            cv.line(image, (1, 350), (199, 350), 255, 1)
            filtered_len = len(spine_angles_filtered[i])
            for j in range(1, filtered_len):
                cv.line(image, (round((j - 1) / filtered_len * 190 + 5), round(spine_angles_filtered[i][j - 1] * 100) + 350), (round(j / filtered_len * 190 + 5), round(spine_angles_filtered[i][j] * 100) + 350), 255, 2)
            annotated.write(image)
            print('\r' + str(i), end = '')
        annotated.release()
        '''
        '''
        for i in range(100):
            spine = [spine_angles[i][j] for j in range(spine_lens[i] - 2)]
            tail_angle = cal_direction_change(cal_direction(spines[i][0], spines[i][1]), cal_direction(tails[i], spines[i][0]))
            #spine.insert(0, tail_angle)
            ax.plot(spine)
        '''
        '''
        plt.figure('correlation')
        
        turn_avg_p.plot_peaks(turn_avg.b, 'b', None)
        tail_angles_change_avg_p.plot_peaks(tail_angles_avg.b, 'g', None)
        plt.show()
        '''
    '''
    plt.figure('accel')
    accel_avg.plot_peaks(accel_avg.b, 'y', 'r')
    accel_avg.plot_peaks(speed_avg, 'b', 'r')
    plt.show()
    '''
    with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
        for key in analysis:
            f.write(key + ', ' + str(analysis[key]) + '\n')
    print('Analysis of ' + videoname + ' complete.')
