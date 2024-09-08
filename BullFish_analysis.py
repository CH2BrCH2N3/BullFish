import os
import csv
import scipy.signal
import matplotlib.pyplot as plt
import math
from decimal import Decimal
from statistics import median
pi = Decimal(math.pi)

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
    
    def __init__(self, start=0, length=0, end=0, peakpos=0, uplength=0,
                 height=0, upheight=0, auc=0, maxslope=0, maxslopepos=0):
        self.start = start
        self.length = length
        self.end = end
        self.peakpos = peakpos
        self.uplength = uplength
        self.height = height
        self.upheight = upheight
        self.auc = auc
        self.maxslope = maxslope
        self.maxslopepos = maxslopepos

class peak_analysis:
    
    def __init__(self, inputlist, window=0, start=0, end=0):
        
        self.original_list = inputlist
        
        if window > 2:
            temp_list = self.original_list[start:end]
            self.list = [0 for i in range(end)]
            for i in range(window // 2):
                temp_list.insert(0, self.original_list[start])
            for i in range(window // 2):
                temp_list.append(self.original_list[end - 1])
            for i in range(start, end):
                self.list[i] = sum(temp_list[(i - start):(i - start + window)]) / window
        else:
            self.list = list(inputlist)
        
        i = 0
        while i < end:
            if self.list[i] == 0:
                self.list[i] = self.list[start]
                i += 1
            else:
                break
        
        self.p_list = [(item if item > 0 else 0) for item in self.list]
        self.n_list = [(-item if item < 0 else 0) for item in self.list]
        
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
    
    with open(path + '/' + videoname + '_cen.csv', 'r') as f:
        cen = [[cell for cell in row] for row in csv.reader(f)]
        cen.pop(0)
        for i in range(l):
            cen[i] = (Decimal(cen[i][0]) * ratio, Decimal(cen[i][1]) * ratio)
    
    sampling = 2
    cen_dists = [0 for i in range(l)]
    speeds = [0 for i in range(l)]
    for i in range(sampling, l, sampling):
        cen_dists[i] = pyth(cen[i], cen[i - sampling]) / Decimal(sampling)
        speeds[i] = cen_dists[i] * fps
    for i in range(0, sampling):
        cen_dists[i] = cen_dists[sampling]
    for i in range(sampling * 2, l, sampling):
        for j in range(i - sampling + 1, i):
            cen_dists[j] = (cen_dists[i - sampling] * (i - j) + cen_dists[i] * (j - (i - sampling))) / Decimal(sampling)
            speeds[j] = cen_dists[j] * fps
    total_distance = sum(cen_dists)
    total_time = Decimal(l) / fps
    speed_avg = total_distance / total_time
    
    speeds = peak_analysis(speeds, window=5, start=1, end=l)
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
        cdist1s[i] = sum((cen_dists[j] for j in range(start, end))) * fps / Decimal(end - start)
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
    for i in range(1, l):
        accels[i] = (speeds.list[i] - speeds.list[i - 1]) * fps
    accels[0] = accels[1]
    accels = peak_analysis(accels, window=5, start=1, end=l)
    accels_peaks, _ = scipy.signal.find_peaks(accels.p_list, prominence=100)
    
    fig, ax = plt.subplots()
    ax.plot(accels.p_list)
    scatter_x = accels_peaks
    scatter_y = [accels.p_list[i] for i in accels_peaks]
    ax.scatter(scatter_x, scatter_y, marker='x')
    
    accels_peak_count = len(accels_peaks)
    speeds_data = [peak_data() for i in range(accels_peak_count)]
    i = 0
    while i < accels_peak_count:
        j = int(accels_peaks[i])
        while j > 0:
            slope = speeds.list[j] - speeds.list[j - 1]
            if slope < -0.1:
                speeds_data[i].start = j
                break
            j -= 1
        j = int(accels_peaks[i])
        while j < l - 1:
            slope = speeds.list[j + 1] - speeds.list[j]
            if slope < -0.2:
                speeds_data[i].end = j
                speeds_data[i].length = j - speeds_data[i].start
                break
            j += 1
        speeds_data[i].height = speeds.list[speeds_data[i].end]
        speeds_data[i].upheight = speeds_data[i].height - speeds.list[speeds_data[i].start]
        for j in range(speeds_data[i].start, speeds_data[i].end):
            slope = speeds.list[j + 1] - speeds.list[j]
            if slope > speeds_data[i].maxslope:
                speeds_data[i].maxslope = slope
                speeds_data[i].maxslopepos = j
        i += 1
    accels_count = len(speeds_data)
    
    fig, ax = plt.subplots()
    ax.plot(speeds.list)
    for i in range(accels_count):
        x = [j for j in range(speeds_data[i].start, speeds_data[i].end + 1)]
        y = [speeds.list[j] for j in range(speeds_data[i].start, speeds_data[i].end + 1)]
        ax.plot(x, y, c='r')
    scatter_x = [speeds_data[i].maxslopepos for i in range(accels_count)]
    scatter_y = [speeds.list[speeds_data[i].maxslopepos] for i in range(accels_count)]
    ax.scatter(scatter_x, scatter_y, c='y', marker='o')
    '''
    ke = [0 for i in range(accels_count)]
    for i in range(accels_count):
        ke[i] = speeds_data[i].height ** 2 - speeds.list[speeds_data[i].start] ** 2
    '''
    accels_per_min = Decimal(accels_count) * 60 / total_time
    total_speed_change = 0
    total_accel_dur = 0
    mean_speed_change = 0
    mean_peak_accel = 0
    mean_accel = 0
    mean_accel_dur = 0
    max_speed_change = 0
    max_peak_accel = 0
    max_accel = 0
    max_accel_dur = 0
    if accels_count > 0:
        total_speed_change = sum([speeds_data[i].upheight for i in range(accels_count)])
        total_accel_dur = sum([(Decimal(speeds_data[i].length) / fps) for i in range(accels_count)])
        mean_speed_change = total_speed_change / Decimal(accels_count)
        mean_peak_accel = sum([(speeds_data[i].maxslope * fps) for i in range(accels_count)]) / Decimal(accels_count)
        mean_accel = sum([(speeds_data[i].upheight * fps / Decimal(speeds_data[i].length)) for i in range(accels_count)]) / Decimal(accels_count)
        mean_accel_dur = total_accel_dur / Decimal(accels_count)
        max_speed_change = max([speeds_data[i].upheight for i in range(accels_count)])
        max_peak_accel = max([(speeds_data[i].maxslope * fps) for i in range(accels_count)])
        max_accel = max([(speeds_data[i].upheight * fps / Decimal(speeds_data[i].length)) for i in range(accels_count)])
        max_accel_dur = max([(Decimal(speeds_data[i].length) / fps) for i in range(accels_count)])
    analysis.update({
        'accels_per_min': accels_per_min,
        'total_speed_change': total_speed_change,
        'total_accel_dur': total_accel_dur,
        'mean_speed_change': mean_speed_change,
        'mean_peak_accel': mean_peak_accel,
        'mean_accel': mean_accel,
        'mean_accel_dur': mean_accel_dur,
        'max_speed_change': max_speed_change,
        'max_peak_accel': max_peak_accel,
        'max_accel': max_accel,
        'max_accel_dur': max_accel_dur
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
        
        directions_free = [0 for i in range(l)]
        directions_free[0] = directions[0]
        for i in range(1, l):
            directions_free[i] = directions_free[i - 1] + turns[i] / fps
        directions_free = peak_analysis(directions_free, window=3, start=0, end=l)
        turns_original = list(turns)
        turns = [0 for i in range(l)]
        for i in range(1, l):
            turns[i] = (directions_free.list[i] - directions_free.list[i - 1]) * fps
        turns = peak_analysis(turns, window=3, start=1, end=l)
        
        turns_p_peaks, _ = scipy.signal.find_peaks(turns.p_list, prominence=2)
        turns_p_count = len(turns_p_peaks)
        turns_p_data = [peak_data() for i in range(turns_p_count)]
        i = 0
        while i < turns_p_count:
            j = int(turns_p_peaks[i])
            turns_p_data[i].peakpos = j
            turns_p_data[i].height = turns.p_list[j]
            while j > 0:
                slope = turns.p_list[j] - turns.p_list[j - 1]
                if turns.p_list[j] > 2 and slope < -1:
                    break
                elif turns.p_list[j] <= 2 and slope < 0.1:
                    break
                j -= 1
            turns_p_data[i].start = j
            turns_p_data[i].uplength = turns_p_peaks[i] - j
            turns_p_data[i].upheight = turns_p_data[i].height - turns.p_list[j]
            j = int(turns_p_peaks[i])
            while j < l - 1:
                slope = turns.p_list[j + 1] - turns.p_list[j]
                if turns.p_list[j] > 2 and slope > 1:
                    break
                elif turns.p_list[j] <= 2 and slope > -0.1:
                    break
                j += 1
            turns_p_data[i].end = j
            turns_p_data[i].length = j - turns_p_data[i].start
            for j in range(turns_p_data[i].start, turns_p_peaks[i]):
                slope = turns.p_list[j + 1] - turns.p_list[j]
                if slope > turns_p_data[i].maxslope:
                    turns_p_data[i].maxslope = slope
                    turns_p_data[i].maxslopepos = j
            for j in range(turns_p_data[i].start, turns_p_data[i].end + 1):
                turns_p_data[i].auc += turns.p_list[j]
            i += 1
        
        turns_n_peaks, _ = scipy.signal.find_peaks(turns.n_list, prominence=2)
        turns_n_count = len(turns_n_peaks)
        turns_n_data = [peak_data() for i in range(turns_n_count)]
        i = 0
        while i < turns_n_count:
            j = int(turns_n_peaks[i])
            turns_n_data[i].peakpos = j
            turns_n_data[i].height = turns.n_list[j]
            while j > 0:
                slope = turns.n_list[j] - turns.n_list[j - 1]
                if turns.n_list[j] > 2 and slope < -1:
                    break
                elif turns.n_list[j] <= 2 and slope < 0.1:
                    break
                j -= 1
            turns_n_data[i].start = j
            turns_n_data[i].uplength = turns_n_peaks[i] - j
            turns_n_data[i].upheight = turns_n_data[i].height - turns.n_list[j]
            j = int(turns_n_peaks[i])
            while j < l - 1:
                slope = turns.n_list[j + 1] - turns.n_list[j]
                if turns.n_list[j] > 2 and slope > 1:
                    break
                elif turns.n_list[j] <= 2 and slope > -0.1:
                    break
                j += 1
            turns_n_data[i].end = j
            turns_n_data[i].length = j - turns_n_data[i].start
            for j in range(turns_n_data[i].start, turns_n_peaks[i]):
                slope = turns.n_list[j + 1] - turns.n_list[j]
                if slope > turns_n_data[i].maxslope:
                    turns_n_data[i].maxslope = slope
                    turns_n_data[i].maxslopepos = j
            for j in range(turns_n_data[i].start, turns_n_data[i].end + 1):
                turns_n_data[i].auc += turns.n_list[j]
            i += 1
        '''
        for i in range(turns_p_count):
            if turns_p_data[i].auc > 17:
                turns_p_data[i].function = {'significant': True}
            else:
                turns_p_data[i].function = {'significant': False}
        for i in range(turns_n_count):
            if turns_n_data[i].auc > 17:
                turns_n_data[i].function = {'significant': True}
            else:
                turns_n_data[i].function = {'significant': False}
        '''
        turns_data = []
        i = 0
        p_i = 0
        n_i = 0
        while p_i < turns_p_count or n_i < turns_n_count:
            choose_p = True
            if p_i >= turns_p_count:
                choose_p = False
            elif n_i < turns_n_count:
                if turns_p_data[p_i].start > turns_n_data[n_i].start:
                    choose_p = False
            if choose_p:
                turns_data.append(turns_p_data[p_i])
                p_i += 1
            else:
                turns_data.append(turns_n_data[n_i])
                '''
                turns_data[i].height = -turns_data[i].height
                turns_data[i].upheight = -turns_data[i].upheight
                turns_data[i].auc = -turns_data[i].auc
                turns_data[i].maxslope = -turns_data[i].maxslope
                '''
                n_i += 1
            i += 1
        turns_count = len(turns_data)
        
        turn_count_per_min = Decimal(turns_count) * 60 / total_time
        right_turn_count_per_min = Decimal(turns_p_count) * 60 / total_time
        left_turn_count_per_min = Decimal(turns_n_count) * 60 / total_time
        turn_count_preference = cal_preference(turns_p_count, turns_n_count)
        total_turn_angle_per_min = sum([abs(turns_data[i].auc / fps * 180 / pi) for i in range(turns_count)]) * 60 / total_time
        total_right_turn_angle_per_min = sum([(turns_p_data[i].auc / fps * 180 / pi) for i in range(turns_p_count)]) * 60 / total_time
        total_left_turn_angle_per_min = sum([(turns_n_data[i].auc / fps * 180 / pi) for i in range(turns_n_count)]) * 60 / total_time
        total_turn_angle_preference = cal_preference(total_right_turn_angle_per_min, total_left_turn_angle_per_min)
        mean_turn_angle = total_turn_angle_per_min * total_time / 60 / Decimal(turns_count)
        mean_right_turn_angle = total_right_turn_angle_per_min * total_time / 60 / Decimal(turns_p_count)
        mean_left_turn_angle = total_left_turn_angle_per_min * total_time / 60 / Decimal(turns_n_count)
        mean_turn_angle_preference = cal_preference(mean_right_turn_angle, mean_left_turn_angle)
        total_turn_dur = sum([(Decimal(turns_data[i].length) / fps) for i in range(turns_count)])
        total_right_turn_dur = sum([(Decimal(turns_p_data[i].length) / fps) for i in range(turns_p_count)])
        total_left_turn_dur = sum([(Decimal(turns_n_data[i].length) / fps) for i in range(turns_n_count)])
        total_turn_dur_preference = cal_preference(total_right_turn_dur, total_left_turn_dur)
        mean_turn_dur = total_turn_dur / Decimal(turns_count)
        mean_right_turn_dur = total_right_turn_dur / Decimal(turns_p_count)
        mean_left_turn_dur = total_left_turn_dur / Decimal(turns_n_count)
        mean_turn_dur_preference = cal_preference(mean_right_turn_dur, mean_left_turn_dur)
        mean_turn_velocity = sum([abs(turns_data[i].height) for i in range(turns_count)]) / Decimal(turns_count)
        mean_right_turn_velocity = sum([turns_p_data[i].height for i in range(turns_p_count)]) / Decimal(turns_p_count)
        mean_left_turn_velocity = sum([turns_n_data[i].height for i in range(turns_n_count)]) / Decimal(turns_n_count)
        mean_turn_velocity_preference = cal_preference(mean_right_turn_velocity, mean_left_turn_velocity)
        analysis.update({
            'turn_count_per_min': turn_count_per_min,
            'right_turn_count_per_min': right_turn_count_per_min,
            'left_turn_count_per_min': left_turn_count_per_min,
            'turn_count_preference': turn_count_preference,
            'total_turn_angle_per_min': total_turn_angle_per_min,
            'total_right_turn_angle_per_min': total_right_turn_angle_per_min,
            'total_left_turn_angle_per_min': total_left_turn_angle_per_min,
            'total_turn_angle_preference': total_turn_angle_preference,
            'mean_turn_angle': mean_turn_angle,
            'mean_right_turn_angle': mean_right_turn_angle,
            'mean_left_turn_angle': mean_left_turn_angle,
            'mean_turn_angle_preference': mean_turn_angle_preference,
            'total_turn_dur': total_turn_dur,
            'total_right_turn_dur': total_right_turn_dur,
            'total_left_turn_dur': total_left_turn_dur,
            'total_turn_dur_preference': total_turn_dur_preference,
            'mean_turn_dur': mean_turn_dur,
            'mean_right_turn_dur': mean_right_turn_dur,
            'mean_left_turn_dur': mean_left_turn_dur,
            'mean_turn_dur_preference': mean_turn_dur_preference,
            'mean_turn_velocity': mean_turn_velocity,
            'mean_right_turn_velocity': mean_right_turn_velocity,
            'mean_left_turn_velocity': mean_left_turn_velocity,
            'mean_turn_velocity_preference': mean_turn_velocity_preference
        })
        
        trunk_angles = peak_analysis(trunk_angles, window=5, start=0, end=l)
        trunk_angles_ddt = [0 for i in range(l)]
        for i in range(1, l):
            trunk_angles_ddt[i] = trunk_angles.list[i] - trunk_angles.list[i - 1]
        trunk_angles_ddt = peak_analysis(trunk_angles_ddt, window=5, start=1, end=l)
        trunk_angles_p_data = []
        trunk_angles_ddt_p_peaks, _ = scipy.signal.find_peaks(trunk_angles_ddt.p_list, prominence=0.05)
        trunk_angles_ddt_p_count = len(trunk_angles_ddt_p_peaks)
        i = 0
        while i < trunk_angles_ddt_p_count:
            current_peak_data = peak_data()
            j = int(trunk_angles_ddt_p_peaks[i])
            while j > 0:
                slope = trunk_angles.list[j] - trunk_angles.list[j - 1]
                if slope < 0:
                    break
                j -= 1
            current_peak_data.start = j
            j = int(trunk_angles_ddt_p_peaks[i])
            while j < l - 1:
                slope = trunk_angles.list[j + 1] - trunk_angles.list[j]
                if slope < 0:
                    break
                j += 1
            current_peak_data.peakpos = j
            current_peak_data.uplength = j - current_peak_data.start
            current_peak_data.height = trunk_angles.list[j]
            current_peak_data.upheight = trunk_angles.list[j] - trunk_angles.list[current_peak_data.start]
            if current_peak_data.upheight < 0.05:
                i += 1
                continue
            trunk_angles_p_data.append(current_peak_data)
            i += 1
        trunk_angles_p_count = len(trunk_angles_p_data)
        trunk_angles_n_peaks, _ = scipy.signal.find_peaks(trunk_angles.n_list, prominence=0.05)
        trunk_angles_n_data = []
        trunk_angles_ddt_n_peaks, _ = scipy.signal.find_peaks(trunk_angles_ddt.n_list, prominence=0.05)
        trunk_angles_ddt_n_count = len(trunk_angles_ddt_n_peaks)
        i = 0
        while i < trunk_angles_ddt_n_count:
            current_peak_data = peak_data()
            j = int(trunk_angles_ddt_n_peaks[i])
            while j > 0:
                slope = trunk_angles.list[j] - trunk_angles.list[j - 1]
                if slope > 0:
                    break
                j -= 1
            current_peak_data.start = j
            j = int(trunk_angles_ddt_n_peaks[i])
            while j < l - 1:
                slope = trunk_angles.list[j + 1] - trunk_angles.list[j]
                if slope > 0:
                    break
                j += 1
            current_peak_data.peakpos = j
            current_peak_data.uplength = j - current_peak_data.start
            current_peak_data.height = trunk_angles.list[j]
            current_peak_data.upheight = trunk_angles.list[j] - trunk_angles.list[current_peak_data.start]
            if current_peak_data.upheight > -0.05:
                i += 1
                continue
            trunk_angles_n_data.append(current_peak_data)
            i += 1
        trunk_angles_n_count = len(trunk_angles_n_data)
        
        trunk_angles_data = []
        i = 0
        p_i = 0
        n_i = 0
        while p_i < trunk_angles_p_count or n_i < trunk_angles_n_count:
            choose_p = True
            if p_i >= trunk_angles_p_count:
                choose_p = False
            elif n_i < trunk_angles_n_count:
                if trunk_angles_p_data[p_i].start > trunk_angles_n_data[n_i].start:
                    choose_p = False
            if choose_p:
                trunk_angles_data.append(trunk_angles_p_data[p_i])
                p_i += 1
            else:
                trunk_angles_data.append(trunk_angles_n_data[n_i])
                trunk_angles_data[i].height = -trunk_angles_data[i].height
                trunk_angles_data[i].upheight = -trunk_angles_data[i].upheight
                n_i += 1
            i += 1
        trunk_angles_count = len(trunk_angles_data)
        i = 0
        while i < trunk_angles_count - 1:
            if trunk_angles_data[i].end > trunk_angles_data[i + 1].start:
                trunk_angles_data[i].end = trunk_angles_data[i + 1].start
            i += 1
        '''
        trunk_bend_count_per_min = Decimal(trunk_angles_count) * 60 / total_time
        right_trunk_bend_count_per_min = Decimal(trunk_angles_p_count) * 60 / total_time
        left_trunk_bend_count_per_min = Decimal(trunk_angles_n_count) * 60 / total_time
        trunk_bend_count_preference = cal_preference(trunk_angles_p_count, trunk_angles_n_count)
        total_trunk_bend_angle_per_min = sum([trunk_angles_data[i].upheight])
        '''
        tail_angles = peak_analysis(tail_angles, window=5, start=0, end=l)
        tail_angles_ddt = [0 for i in range(l)]
        for i in range(1, l):
            tail_angles_ddt[i] = tail_angles.list[i] - tail_angles.list[i - 1]
        tail_angles_ddt = peak_analysis(tail_angles_ddt, window=5, start=1, end=l)
        tail_angles_p_data = []
        tail_angles_ddt_p_peaks, _ = scipy.signal.find_peaks(tail_angles_ddt.p_list, prominence=0.05)
        tail_angles_ddt_p_count = len(tail_angles_ddt_p_peaks)
        i = 0
        while i < tail_angles_ddt_p_count:
            current_peak_data = peak_data()
            j = int(tail_angles_ddt_p_peaks[i])
            while j > 0:
                slope = tail_angles.list[j] - tail_angles.list[j - 1]
                if slope < 0:
                    break
                j -= 1
            current_peak_data.start = j
            j = int(tail_angles_ddt_p_peaks[i])
            while j < l - 1:
                slope = tail_angles.list[j + 1] - tail_angles.list[j]
                if slope < 0:
                    break
                j += 1
            current_peak_data.peakpos = j
            current_peak_data.uplength = j - current_peak_data.start
            current_peak_data.height = tail_angles.list[j]
            current_peak_data.upheight = tail_angles.list[j] - tail_angles.list[current_peak_data.start]
            if current_peak_data.upheight < 0.05:
                i += 1
                continue
            tail_angles_p_data.append(current_peak_data)
            i += 1
        tail_angles_p_count = len(tail_angles_p_data)
        tail_angles_n_peaks, _ = scipy.signal.find_peaks(tail_angles.n_list, prominence=0.05)
        tail_angles_n_data = []
        tail_angles_ddt_n_peaks, _ = scipy.signal.find_peaks(tail_angles_ddt.n_list, prominence=0.05)
        tail_angles_ddt_n_count = len(tail_angles_ddt_n_peaks)
        i = 0
        while i < tail_angles_ddt_n_count:
            current_peak_data = peak_data()
            j = int(tail_angles_ddt_n_peaks[i])
            while j > 0:
                slope = tail_angles.list[j] - tail_angles.list[j - 1]
                if slope > 0:
                    break
                j -= 1
            current_peak_data.start = j
            j = int(tail_angles_ddt_n_peaks[i])
            while j < l - 1:
                slope = tail_angles.list[j + 1] - tail_angles.list[j]
                if slope > 0:
                    break
                j += 1
            current_peak_data.peakpos = j
            current_peak_data.uplength = j - current_peak_data.start
            current_peak_data.height = tail_angles.list[j]
            current_peak_data.upheight = tail_angles.list[j] - tail_angles.list[current_peak_data.start]
            if current_peak_data.upheight > -0.05:
                i += 1
                continue
            tail_angles_n_data.append(current_peak_data)
            i += 1
        tail_angles_n_count = len(tail_angles_n_data)
        
        tail_angles_data = []
        i = 0
        p_i = 0
        n_i = 0
        while p_i < tail_angles_p_count or n_i < tail_angles_n_count:
            choose_p = True
            if p_i >= tail_angles_p_count:
                choose_p = False
            elif n_i < tail_angles_n_count:
                if tail_angles_p_data[p_i].start > tail_angles_n_data[n_i].start:
                    choose_p = False
            if choose_p:
                tail_angles_data.append(tail_angles_p_data[p_i])
                p_i += 1
            else:
                tail_angles_data.append(tail_angles_n_data[n_i])
                tail_angles_data[i].height = -tail_angles_data[i].height
                tail_angles_data[i].upheight = -tail_angles_data[i].upheight
                n_i += 1
            i += 1
        tail_angles_count = len(tail_angles_data)
        i = 0
        while i < tail_angles_count - 1:
            if tail_angles_data[i].end > tail_angles_data[i + 1].start:
                tail_angles_data[i].end = tail_angles_data[i + 1].start
            i += 1
        
        amplitudes = peak_analysis(amplitudes, window=5, start=0, end=l)
        amplitudes_ddt = [0 for i in range(l)]
        for i in range(1, l):
            amplitudes_ddt[i] = amplitudes.list[i] - amplitudes.list[i - 1]
        amplitudes_ddt = peak_analysis(amplitudes_ddt, window=5, start=1, end=l)
        amplitudes_p_data = []
        amplitudes_ddt_p_peaks, _ = scipy.signal.find_peaks(amplitudes_ddt.p_list, prominence=0.5)
        amplitudes_ddt_p_count = len(amplitudes_ddt_p_peaks)
        i = 0
        while i < amplitudes_ddt_p_count:
            current_peak_data = peak_data()
            j = int(amplitudes_ddt_p_peaks[i])
            while j > 0:
                slope = amplitudes.list[j] - amplitudes.list[j - 1]
                if slope < 0:
                    break
                j -= 1
            current_peak_data.start = j
            j = int(amplitudes_ddt_p_peaks[i])
            while j < l - 1:
                slope = amplitudes.list[j + 1] - amplitudes.list[j]
                if slope < 0:
                    break
                j += 1
            current_peak_data.peakpos = j
            current_peak_data.uplength = j - current_peak_data.start
            current_peak_data.height = amplitudes.list[j]
            current_peak_data.upheight = amplitudes.list[j] - amplitudes.list[current_peak_data.start]
            if current_peak_data.upheight < 0.5:
                i += 1
                continue
            amplitudes_p_data.append(current_peak_data)
            i += 1
        amplitudes_p_count = len(amplitudes_p_data)
        amplitudes_n_peaks, _ = scipy.signal.find_peaks(amplitudes.n_list, prominence=0.5)
        amplitudes_n_data = []
        amplitudes_ddt_n_peaks, _ = scipy.signal.find_peaks(amplitudes_ddt.n_list, prominence=0.5)
        amplitudes_ddt_n_count = len(amplitudes_ddt_n_peaks)
        i = 0
        while i < amplitudes_ddt_n_count:
            current_peak_data = peak_data()
            j = int(amplitudes_ddt_n_peaks[i])
            while j > 0:
                slope = amplitudes.list[j] - amplitudes.list[j - 1]
                if slope > 0:
                    break
                j -= 1
            current_peak_data.start = j
            j = int(amplitudes_ddt_n_peaks[i])
            while j < l - 1:
                slope = amplitudes.list[j + 1] - amplitudes.list[j]
                if slope > 0:
                    break
                j += 1
            current_peak_data.peakpos = j
            current_peak_data.uplength = j - current_peak_data.start
            current_peak_data.height = amplitudes.list[j]
            current_peak_data.upheight = amplitudes.list[j] - amplitudes.list[current_peak_data.start]
            if current_peak_data.upheight > -0.5:
                i += 1
                continue
            amplitudes_n_data.append(current_peak_data)
            i += 1
        amplitudes_n_count = len(amplitudes_n_data)
        
        amplitudes_data = []
        p_i = 0
        n_i = 0
        while p_i < amplitudes_p_count or n_i < amplitudes_n_count:
            choose_p = True
            if p_i >= amplitudes_p_count:
                choose_p = False
            elif n_i < amplitudes_n_count:
                if amplitudes_p_data[p_i].start > amplitudes_n_data[n_i].start:
                    choose_p = False
            if choose_p:
                amplitudes_data.append(amplitudes_p_data[p_i])
                p_i += 1
            else:
                amplitudes_data.append(amplitudes_n_data[n_i])
                n_i += 1
        amplitudes_count = len(amplitudes_data)
        i = 0
        while i < amplitudes_count - 1:
            if amplitudes_data[i].end > amplitudes_data[i + 1].start:
                amplitudes_data[i].end = amplitudes_data[i + 1].start
            i += 1
        
        bursts = []
        i = 0
        accel_i = 0
        while i < trunk_angles_count - 1:
            pair = False
            i_data = trunk_angles_data[i]
            next_data = trunk_angles_data[i + 1]
            i_up = trunk_angles.list[i_data.peakpos] > trunk_angles.list[i_data.start]
            next_up = trunk_angles.list[next_data.peakpos] > trunk_angles.list[next_data.start]
            if (i_up == True and next_up == False) or (i_up == False and next_up == True):
                if next_data.start - i_data.peakpos <= 1:
                    pair = True
            if pair:
                while accel_i < accels_count:
                    if speeds_data[accel_i].maxslopepos >= i_data.start:
                        if speeds_data[accel_i].maxslopepos < next_data.peakpos:
                            #trunk_angles_data[i].function = {'pair': i + 1}
                            
                            bursts.append({'i': i,
                                           'next': i + 1,
                                           'accel': speeds_data[accel_i]})
                            accel_i += 1
                        break
                    accel_i += 1
            i += 1
        
        
        trunk_curvs_filtered = peak_analysis(trunk_curvs_filtered, window=5, start=0, end=l)
        trunk_curvs_peaks, _ = scipy.signal.find_peaks(trunk_curvs_filtered.list, prominence=0.05)
        trunk_curvs_count = len(trunk_curvs_peaks)
        trunk_curvs_data = [peak_data() for i in range(trunk_curvs_count)]
        i = 0
        while i < trunk_curvs_count:
            j = int(trunk_curvs_peaks[i])
            trunk_curvs_data[i].peakpos = j
            trunk_curvs_data[i].height = trunk_curvs_filtered.list[j]
            while j > 0:
                slope = trunk_curvs_filtered.list[j] - trunk_curvs_filtered.list[j - 1]
                if slope < 0:
                    break
                j -= 1
            trunk_curvs_data[i].start = j
            trunk_curvs_data[i].upheight = trunk_curvs_data[i].height - trunk_curvs_filtered.list[j]
            trunk_curvs_data[i].uplength = trunk_curvs_peaks[i] - j
            j = int(trunk_curvs_peaks[i])
            while j < l - 1:
                slope = trunk_curvs_filtered.list[j + 1] - trunk_curvs_filtered.list[j]
                if slope > 0:
                    break
                j += 1
            trunk_curvs_data[i].end = j
            trunk_curvs_data[i].length = j - trunk_curvs_data[i].start
            for j in range(trunk_curvs_data[i].start, trunk_curvs_data[i].end):
                slope = trunk_curvs_filtered.list[j + 1] - trunk_curvs_filtered.list[j]
                if slope > trunk_curvs_data[i].maxslope:
                    trunk_curvs_data[i].maxslope = slope
                    trunk_curvs_data[i].maxslopepos = j
            i += 1
        i = 0
        while i < trunk_curvs_count:
            if trunk_curvs_data[i].upheight < 0.05:
                trunk_curvs_data.pop(i)
                i -= 1
                trunk_curvs_count -= 1
            i += 1
        
        trunk_curv_count_per_min = trunk_curvs_count * 60 / total_time
        total_trunk_curv_angle_per_min = sum([(trunk_curvs_data[i].upheight * 180 / pi) for i in range(trunk_curvs_count)]) * 60 / total_time
        mean_trunk_curv_angle = total_trunk_curv_angle_per_min * total_time / 60 / Decimal(trunk_curvs_count)
        analysis.update({
            'trunk_curv_count_per_min': trunk_curv_count_per_min,
            'total_trunk_curv_angle_per_min': total_trunk_curv_angle_per_min,
            'mean_trunk_curv_angle': mean_trunk_curv_angle
        })
        
        total_curvs_filtered = peak_analysis(total_curvs_filtered, window=5, start=0, end=l)
        total_curvs_peaks, _ = scipy.signal.find_peaks(total_curvs_filtered.list, prominence=0.05)
        total_curvs_count = len(total_curvs_peaks)
        total_curvs_data = [peak_data() for i in range(total_curvs_count)]
        i = 0
        while i < total_curvs_count:
            j = int(total_curvs_peaks[i])
            total_curvs_data[i].peakpos = j
            total_curvs_data[i].height = total_curvs_filtered.list[j]
            while j > 0:
                slope = total_curvs_filtered.list[j] - total_curvs_filtered.list[j - 1]
                if slope < 0:
                    break
                j -= 1
            total_curvs_data[i].start = j
            total_curvs_data[i].upheight = total_curvs_data[i].height - total_curvs_filtered.list[j]
            total_curvs_data[i].uplength = total_curvs_peaks[i] - j
            j = int(total_curvs_peaks[i])
            while j < l - 1:
                slope = total_curvs_filtered.list[j + 1] - total_curvs_filtered.list[j]
                if slope > 0:
                    break
                j += 1
            total_curvs_data[i].end = j
            total_curvs_data[i].length = j - total_curvs_data[i].start
            for j in range(total_curvs_data[i].start, total_curvs_data[i].end):
                slope = total_curvs_filtered.list[j + 1] - total_curvs_filtered.list[j]
                if slope > total_curvs_data[i].maxslope:
                    total_curvs_data[i].maxslope = slope
                    total_curvs_data[i].maxslopepos = j
            i += 1
        i = 0
        while i < total_curvs_count:
            if total_curvs_data[i].upheight < 0.05:
                total_curvs_data.pop(i)
                i -= 1
                total_curvs_count -= 1
            i += 1
        
        #find change list
        #find nearest right or left peak
        #median slope
        
        '''
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
                data = [trunk_curvs_change.positive_peaks[i].pos,
                        trunk_curvs_change.positive_peaks[i].length,
                        trunk_curvs_change.positive_peaks[i].maxima,
                        trunk_curvs_change.positive_peaks[i].auc,
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
        
        fig, (total_curvs_filtered_ax, tail_angles_ax, turn_ax, accel_ax) = plt.subplots(4, sharex=True)
        tail_positive_angles_change.plot_peaks(tail_angles_ax, tail_angles.list, 'y', 'b', None)
        tail_negative_angles_change.plot_peaks(tail_angles_ax, tail_angles.list, None, 'r', None)
        turns.plot_peaks(turn_ax, turns.list, 'y', 'b', 'r')
        accels.plot_peaks(accel_ax, speeds.list, 'y', 'b', 'r')
        plt.show()
        
        plt.figure()
        scatter_x = [peak.auc for peak in trunk_curvs_change.positive_peaks]
        scatter_y = stride_lengths
        
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
        
        plt.scatter(scatter_x, scatter_y)
        plt.show()
        
        plt.figure('correlation')
        '''
        
        fig, ax = plt.subplots()
        #ax.plot(turns.original_list, c='y')
        ax.plot(turns.list)
        scatter_x = [turns_p_data[i].peakpos for i in range(turns_p_count)]
        scatter_y = [turns.list[turns_p_data[i].peakpos] for i in range(turns_p_count)]
        ax.scatter(scatter_x, scatter_y, c='b', marker='o')
        for i in range(turns_p_count):
            y = [turns.list[j] for j in range(turns_p_data[i].start, turns_p_data[i].end + 1)]
            x = [j for j in range(turns_p_data[i].start, turns_p_data[i].end + 1)]
            ax.plot(x, y, c='b')
        scatter_x = [turns_n_data[i].peakpos for i in range(turns_n_count)]
        scatter_y = [turns.list[turns_n_data[i].peakpos] for i in range(turns_n_count)]
        ax.scatter(scatter_x, scatter_y, c='r', marker='o')
        for i in range(turns_n_count):
            y = [turns.list[j] for j in range(turns_n_data[i].start, turns_n_data[i].end + 1)]
            x = [j for j in range(turns_n_data[i].start, turns_n_data[i].end + 1)]
            ax.plot(x, y, c='r')
        
        fig, ax = plt.subplots()
        ax.plot(tail_angles.list)
        ax.plot(tail_angles_ddt.list, c='y')
        for i in range(tail_angles_p_count):
            ax.plot([j for j in range(tail_angles_p_data[i].start, tail_angles_p_data[i].peakpos + 1)],
                    [tail_angles.list[j] for j in range(tail_angles_p_data[i].start, tail_angles_p_data[i].peakpos + 1)],
                     c='b')
        for i in range(tail_angles_n_count):
            ax.plot([j for j in range(tail_angles_n_data[i].start, tail_angles_n_data[i].peakpos + 1)],
                    [tail_angles.list[j] for j in range(tail_angles_n_data[i].start, tail_angles_n_data[i].peakpos + 1)],
                     c='r')
        
        fig, ax = plt.subplots()
        ax.plot(trunk_angles.list)
        ax.plot(trunk_angles_ddt.list, c='y')
        for i in range(trunk_angles_p_count):
            ax.plot([j for j in range(trunk_angles_p_data[i].start, trunk_angles_p_data[i].peakpos + 1)],
                    [trunk_angles.list[j] for j in range(trunk_angles_p_data[i].start, trunk_angles_p_data[i].peakpos + 1)],
                     c='b')
        for i in range(trunk_angles_n_count):
            ax.plot([j for j in range(trunk_angles_n_data[i].start, trunk_angles_n_data[i].peakpos + 1)],
                    [trunk_angles.list[j] for j in range(trunk_angles_n_data[i].start, trunk_angles_n_data[i].peakpos + 1)],
                     c='r')
        
        fig, ax = plt.subplots()
        ax.plot(amplitudes.list)
        ax.plot(amplitudes_ddt.list, c='y')
        for i in range(amplitudes_p_count):
            ax.plot([j for j in range(amplitudes_p_data[i].start, amplitudes_p_data[i].peakpos + 1)],
                    [amplitudes.list[j] for j in range(amplitudes_p_data[i].start, amplitudes_p_data[i].peakpos + 1)],
                     c='b')
        for i in range(amplitudes_n_count):
            ax.plot([j for j in range(amplitudes_n_data[i].start, amplitudes_n_data[i].peakpos + 1)],
                    [amplitudes.list[j] for j in range(amplitudes_n_data[i].start, amplitudes_n_data[i].peakpos + 1)],
                     c='r')
        
        fig, ax = plt.subplots()
        ax.plot(trunk_angles.list)
        ax.plot([(i / 10) for i in turns.list], c='g')
        ax.plot([(i / 20) for i in speeds.list], c='r')
        
        fig, ax = plt.subplots()
        ax.plot(trunk_curvs_filtered.list)
        for i in range(trunk_curvs_count):
            ax.plot([j for j in range(trunk_curvs_data[i].start, trunk_curvs_data[i].end + 1)],
                    [trunk_curvs_filtered.list[j] for j in range(trunk_curvs_data[i].start, trunk_curvs_data[i].end + 1)],
                    c='r')
        scatter_x = [trunk_curvs_data[i].peakpos for i in range(trunk_curvs_count)]
        scatter_y = [trunk_curvs_filtered.list[trunk_curvs_data[i].peakpos] for i in range(trunk_curvs_count)]
        ax.scatter(scatter_x, scatter_y, c='r', marker='o')
        
        fig, ax = plt.subplots()
        ax.plot(total_curvs_filtered.list)
        for i in range(total_curvs_count):
            ax.plot([j for j in range(total_curvs_data[i].start, total_curvs_data[i].end + 1)],
                    [total_curvs_filtered.list[j] for j in range(total_curvs_data[i].start, total_curvs_data[i].end + 1)],
                    c='r')
        scatter_x = [total_curvs_data[i].peakpos for i in range(total_curvs_count)]
        scatter_y = [total_curvs_filtered.list[total_curvs_data[i].peakpos] for i in range(total_curvs_count)]
        ax.scatter(scatter_x, scatter_y, c='r', marker='o')
        
        with open(path + '/' + videoname + '_turns.csv', 'w', newline='') as f:
            fieldnames = ['start', 'length', 'end', 'peakpos', 'uplength', 'height',
                          'upheight', 'auc', 'maxslope', 'maxslopepos']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(turns_count):
                writer.writerow({'start': turns_data[i].start,
                                 'length': turns_data[i].length,
                                 'end': turns_data[i].end,
                                 'peakpos': turns_data[i].peakpos,
                                 'uplength': turns_data[i].uplength,
                                 'height': turns_data[i].height,
                                 'upheight': turns_data[i].upheight,
                                 'auc': turns_data[i].auc,
                                 'maxslope': turns_data[i].maxslope,
                                 'maxslopepos': turns_data[i].maxslopepos})
        
        with open(path + '/' + videoname + '_trunk_angles.csv', 'w', newline='') as f:
            fieldnames = ['start', 'peakpos', 'uplength', 'height', 'upheight']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(trunk_angles_count):
                writer.writerow({'start': trunk_angles_data[i].start,
                                 'peakpos': trunk_angles_data[i].peakpos,
                                 'uplength': trunk_angles_data[i].uplength,
                                 'height': trunk_angles_data[i].height,
                                 'upheight': trunk_angles_data[i].upheight})
        
        with open(path + '/' + videoname + '_tail_angles.csv', 'w', newline='') as f:
            fieldnames = ['start', 'peakpos', 'uplength', 'height', 'upheight']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(tail_angles_count):
                writer.writerow({'start': tail_angles_data[i].start,
                                 'peakpos': tail_angles_data[i].peakpos,
                                 'uplength': tail_angles_data[i].uplength,
                                 'height': tail_angles_data[i].height,
                                 'upheight': tail_angles_data[i].upheight})
        
        with open(path + '/' + videoname + '_amplitudes.csv', 'w', newline='') as f:
            fieldnames = ['start', 'peakpos', 'uplength', 'height', 'upheight']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(amplitudes_count):
                writer.writerow({'start': amplitudes_data[i].start,
                                 'peakpos': amplitudes_data[i].peakpos,
                                 'uplength': amplitudes_data[i].uplength,
                                 'height': amplitudes_data[i].height,
                                 'upheight': amplitudes_data[i].upheight})
                
        with open(path + '/' + videoname + '_trunk_curvs.csv', 'w', newline='') as f:
            fieldnames = ['start', 'length', 'end', 'peakpos', 'uplength', 'height',
                          'upheight', 'auc', 'maxslope', 'maxslopepos']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(trunk_curvs_count):
                writer.writerow({'start': trunk_curvs_data[i].start,
                                 'length': trunk_curvs_data[i].length,
                                 'end': trunk_curvs_data[i].end,
                                 'peakpos': trunk_curvs_data[i].peakpos,
                                 'uplength': trunk_curvs_data[i].uplength,
                                 'height': trunk_curvs_data[i].height,
                                 'upheight': trunk_curvs_data[i].upheight,
                                 'auc': trunk_curvs_data[i].auc,
                                 'maxslope': trunk_curvs_data[i].maxslope,
                                 'maxslopepos': trunk_curvs_data[i].maxslopepos})
        
        with open(path + '/' + videoname + '_total_curvs.csv', 'w', newline='') as f:
            fieldnames = ['start', 'length', 'end', 'peakpos', 'uplength', 'height',
                          'upheight', 'auc', 'maxslope', 'maxslopepos']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(total_curvs_count):
                writer.writerow({'start': total_curvs_data[i].start,
                                 'length': total_curvs_data[i].length,
                                 'end': total_curvs_data[i].end,
                                 'peakpos': total_curvs_data[i].peakpos,
                                 'uplength': total_curvs_data[i].uplength,
                                 'height': total_curvs_data[i].height,
                                 'upheight': total_curvs_data[i].upheight,
                                 'auc': total_curvs_data[i].auc,
                                 'maxslope': total_curvs_data[i].maxslope,
                                 'maxslopepos': total_curvs_data[i].maxslopepos})
        
    with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
        for key in analysis:
            f.write(key + ', ' + str(analysis[key]) + '\n')
    print('Analysis of ' + videoname + ' complete.')
    
