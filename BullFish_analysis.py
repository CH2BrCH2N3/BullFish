import os
import csv
import matplotlib.pyplot as plt
import math
from sys import exit
from scipy.signal import find_peaks
from copy import copy
from decimal import Decimal
from statistics import median
pi = Decimal(math.pi)

if not os.path.exists('bullfish_analysis_settings.csv'):
    with open('bullfish_analysis_settings.csv', 'w') as f:
        headers = ['tank_x', 'tank_y', 'plot_figure', 'export', 'sampling', 'accel_cutoff',
                   'min_accel_dur', 'min_max_accel', 'min_speed_change', 'spine_analysis',
                   'turn_cutoff', 'min_turn_dur', 'min_max_turn_velocity',
                   'min_turn_angle', 'bend_cutoff', 'min_bend_dur', 'min_bend_speed',
                   'min_bend_angle', 'amplitude_cutoff', 'min_amplitude_dur',
                   'min_amplitude_dt', 'min_amplitude']
        for word in headers:
            f.write(word + '\n')
    print('Set settings first')
    exit()
else:
    with open('bullfish_analysis_settings.csv', 'r') as f:
        settings = {row[0]: row[1] for row in csv.reader(f)}
    tank_x = Decimal(settings['tank_x'])
    tank_y = Decimal(settings['tank_y'])
    plot_figure = bool(int(settings['plot_figure']))
    export = bool(int(settings['export']))
    sampling = int(settings['sampling'])
    accel_cutoff = Decimal(settings['accel_cutoff'])
    min_accel_dur = Decimal(settings['min_accel_dur'])
    min_max_accel = Decimal(settings['min_max_accel'])
    min_speed_change = Decimal(settings['min_speed_change'])
    spine_analysis = bool(int(settings['spine_analysis']))
    if spine_analysis:
        turn_cutoff = Decimal(settings['turn_cutoff']) * pi / 180
        min_turn_dur = Decimal(settings['min_turn_dur'])
        min_max_turn_velocity = Decimal(settings['min_max_turn_velocity']) * pi / 180
        min_turn_angle = Decimal(settings['min_turn_angle']) * pi / 180
        bend_cutoff = Decimal(settings['bend_cutoff']) * pi / 180
        min_bend_dur = Decimal(settings['min_bend_dur'])
        min_bend_speed = Decimal(settings['min_bend_speed']) * pi / 180
        min_bend_angle = Decimal(settings['min_bend_angle']) * pi / 180
        amplitude_cutoff = Decimal(settings['amplitude_cutoff'])
        min_amplitude_dur = Decimal(settings['min_amplitude_dur'])
        min_amplitude_dt = Decimal(settings['min_amplitude_dt'])
        min_amplitude = Decimal(settings['min_amplitude'])

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

def per_min(count):
    return Decimal(count) * 60 / total_time

def cal_preference(p, n):
    try:
        return (p - n) / (p + n)
    except:
        print('cal_preference Error')
        return 0

class peak_data:
    def __init__(self, start=0, startpos=0, length=0, end=0, endpos=0, peakpos=0,
                 uplength=0, height=0, upheight=0, change=0, auc=0, meanslope=0,
                 maxslope=0, maxslopepos=0):
        self.start = start
        self.startpos = startpos
        self.length = length
        self.end = end
        self.endpos = endpos
        self.peakpos = peakpos
        self.uplength = uplength
        self.height = height
        self.upheight = upheight
        self.change = change
        self.auc = auc
        self.meanslope = meanslope
        self.maxslope = maxslope
        self.maxslopepos = maxslopepos

class list_set:
    
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

# identify local maxima in list1 to characterize peaks in list2
def get_peaks(list1, list2, prominence, criteria_f, criteria_b, criteria_peak):
    peaks, _ = find_peaks(list1, prominence=prominence)
    peaks = [int(peak) for peak in peaks]
    data = []
    l = len(list2)
    for peak in peaks:
        j = peak
        while j > 0:
            slope = list2[j] - list2[j - 1]
            if criteria_f(slope, list2[j]):
                break
            j -= 1
        startpos = j
        start = list2[startpos]
        j = peak
        while j < l - 1:
            slope = list2[j + 1] - list2[j]
            if criteria_b(slope, list2[j]):
                break
            j += 1
        endpos = j
        end = list2[endpos]
        length = endpos - startpos
        if list1 == list2:
            peakpos = peak
            uplength = peakpos - startpos
            height = list2[peakpos]
            upheight = height - list2[startpos]
            change = list2[endpos] - list2[startpos]
            auc = sum(list2[j] for j in range(startpos, endpos + 1))
            maxslope = 0
            maxslopepos = 0
            for j in range(startpos + 1, peakpos + 1):
                slope = list2[j] - list2[j - 1]
                if maxslope < slope:
                    maxslope = slope
                    maxslopepos = j
            datum = peak_data(start=start, startpos=startpos, length=length, end=end,
                              endpos=endpos, peakpos=peakpos, uplength=uplength,
                              height=height, upheight=upheight, change=change, auc=auc,
                              maxslope=maxslope, maxslopepos=maxslopepos)
        else:
            height = list2[endpos]
            change = height - list2[startpos]
            meanslope = change / Decimal(length) if length > 0 else 0
            maxslopepos = peak
            maxslope = list1[peak]
            datum = peak_data(start=start, startpos=startpos, length=length, end=end,
                              endpos=endpos, height=height, change=change,
                              meanslope=meanslope, maxslope=maxslope,
                              maxslopepos=maxslopepos)
        if criteria_peak(datum):
            data.append(datum)
    return data

# merge positive and negative peaks into one list
def merge(data_p, data_n):
    data = []
    len_p = len(data_p)
    len_n = len(data_n)
    i = 0
    i_p = 0
    i_n = 0
    while i_p < len_p or i_n < len_n:
        choose_p = True
        if i_p >= len_p:
            choose_p = False
        elif i_n < len_n:
            if data_p[i_p].startpos > data_n[i_n].startpos:
                choose_p = False
        if choose_p:
            data.append(copy(data_p[i_p]))
            i_p += 1
        else:
            data.append(copy(data_n[i_n]))
            if data[i].uplength != 0:
                data[i].height = -data[i].height
                data[i].upheight = -data[i].upheight
                data[i].change = -data[i].change
                data[i].maxslope = -data[i].maxslope
            i_n += 1
        i += 1
    return data

def find_steps(data, dists):
    data_count = len(data)
    steps = [() for i in range(data_count)]
    for i in range(data_count - 1):
        step = sum(dists[data[i].startpos:data[i + 1].startpos])
        steps[i] = (data[i], step)
    step = sum(dists[data[data_count - 1].startpos:(len(dists) - 1)])
    steps[data_count - 1] = (data[data_count - 1], step)
    sum_step = sum([item[1] for item in steps])
    return (steps, sum_step)

def correlate(xylist, sortby, compute, portion):
    count = len(xylist)
    outputs = [0 for i in range(portion)]
    xylist.sort(key=sortby)
    for i in range(portion):
        start = round(i * count / portion)
        end = round((i + 1) * count / portion)
        outputs[i] = sum([compute(xylist[j]) for j in range(start, end)]) / Decimal(end - start)
    return outputs

def export_data(data, path):
    if data == []:
        print('There are no data to export for ' + path)
        return
    with open(path, 'w', newline='') as f:
        if data[0].uplength != 0:
            fieldnames = ['start', 'startpos', 'length', 'end', 'endpos', 'peakpos',
                          'uplength', 'height', 'upheight', 'change', 'auc',
                          'maxslope', 'maxslopepos']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for datum in data:
                writer.writerow({'start': datum.start,
                                 'startpos': datum.startpos,
                                 'length': datum.length,
                                 'end': datum.end,
                                 'endpos': datum.endpos,
                                 'peakpos': datum.peakpos,
                                 'uplength': datum.uplength,
                                 'height': datum.height,
                                 'upheight': datum.upheight,
                                 'change': datum.change,
                                 'auc': datum.auc,
                                 'maxslope': datum.maxslope,
                                 'maxslopepos': datum.maxslopepos})
        else:
            fieldnames = ['start', 'startpos', 'length', 'end', 'endpos', 'height',
                          'change', 'meanslope', 'maxslope', 'maxslopepos']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for datum in data:
                writer.writerow({'start': datum.start,
                                 'startpos': datum.startpos,
                                 'length': datum.length,
                                 'end': datum.end,
                                 'endpos': datum.endpos,
                                 'height': datum.height,
                                 'change': datum.change,
                                 'meanslope': datum.meanslope,
                                 'maxslope': datum.maxslope,
                                 'maxslopepos': datum.maxslopepos})

analyses = []

# search for videos in the current directory
for file in os.listdir('.'):
    
    # check if the file is a video in the supported formats
    filename = os.fsdecode(file)
    filename_split = os.path.splitext(filename)
    supported_formats = ['.avi', '.mp4']
    if filename_split[1] not in supported_formats:
        continue
    
    # check if the video has been tracked
    videoname = filename_split[0]
    path = './' + videoname
    if not os.path.isfile(path + '/' + videoname + '_metadata.csv'):
        print('Metadata missing for ' + videoname)
        continue
    
    # load metadata
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
    
    # load essential tracking data
    with open(path + '/' + videoname + '_fishlength.csv', 'r') as f:
        fish_lengths = [cell for cell in csv.reader(f)]
    fish_length = median([Decimal(length[0]) for length in fish_lengths])
    
    with open(path + '/' + videoname + '_cen.csv', 'r') as f:
        cen = [[cell for cell in row] for row in csv.reader(f)]
        cen.pop(0)
        for i in range(l):
            cen[i] = (Decimal(cen[i][0]) * ratio, Decimal(cen[i][1]) * ratio)
    
    # obtain a list of speed at each frame
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
    
    # obtain total distance travelled and average speed
    total_distance = sum(cen_dists)
    total_time = Decimal(l) / fps
    speed_avg = total_distance / total_time
    
    # determine whether the fish is freezing for each frame
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
    
    # obtain parameters related to freezing
    total_freeze_time = Decimal(sum(freeze)) / fps
    freeze_percent = total_freeze_time / total_time * 100
    active_time = total_time - total_freeze_time
    active_speed = total_distance / active_time
    freeze_count = 0
    for i in range(2, l):
        if freeze[i] - freeze[i - 1] == 1:
            freeze_count += 1
    freeze_freq = per_min(freeze_count)
    
    # obtain a list of speeds with running average and obtain max speed
    speeds = list_set(speeds, window=5, start=1, end=l)
    max_speed = max(speeds.list)
    
    # obtain a list of speed measured over 1s
    cdist1s = [0 for i in range(l)]
    for i in range(1, l):
        start = max(1, round(i + 1 - fps / 2))
        end = min(l, round(i + 1 + fps / 2))
        cdist1s[i] = sum([cen_dists[j] for j in range(start, end)]) * fps / Decimal(end - start)
    max_distance_1s = max(cdist1s)
    
    # output basic kinematics to analysis dictionary
    analysis = {
        'videoname': videoname,
        'total_time': total_time,
        'total_distance': total_distance,
        'speed_avg': speed_avg,
        'max_speed': max_speed,
        'max_distance_1s': max_distance_1s,
        'active_speed': active_speed,
        'freeze_percent': freeze_percent,
        'freeze_count': freeze_count,
        'freeze_freq': freeze_freq
    }
    
    # obtain a list of acceleration over each frame, with running average
    # the unit of acceleration here is mm/s/frame
    accels = [0 for i in range(l)]
    for i in range(1, l):
        accels[i] = speeds.list[i] - speeds.list[i - 1]
    accels[0] = accels[1]
    accels = list_set(accels, window=5, start=1, end=l)
    
    # get the peaks of acceleration
    def accels_criteria_f(slope, lst):
        if slope < -0.1:
            return True
        else:
            return False
    def accels_criteria_b(slope, lst):
        if slope < -0.2:
            return True
        else:
            return False
    def accels_criteria_peak(datum):
        if datum.length >= int(min_accel_dur * fps) and datum.maxslope * fps >= min_max_accel and datum.change >= min_speed_change:
            return True
        else:
            return False
    accels_data = get_peaks(accels.p_list, speeds.list, accel_cutoff / fps,
                            accels_criteria_f, accels_criteria_b, accels_criteria_peak)
    accels_count = len(accels_data)
    
    if accels_count == 0:
        print('No detectable movement')
        with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
            for key in analysis:
                f.write(key + ', ' + str(analysis[key]) + '\n')
        print('Analysis of ' + videoname + ' complete.')
        analyses.append(analysis)
        continue
    '''
    ke = [0 for i in range(accels_count)]
    for i in range(accels_count):
        ke[i] = speeds_data[i].height ** 2 - speeds.list[speeds_data[i].startpos] ** 2
    '''
    if plot_figure:
        fig, ax = plt.subplots()
        ax.plot([i * fps for i in accels.p_list])
        ax.plot(speeds.list, c='b')
        for datum in accels_data:
            x = [i for i in range(datum.startpos, datum.endpos + 1)]
            y = [speeds.list[i] for i in range(datum.startpos, datum.endpos + 1)]
            ax.plot(x, y, c='r')
        x = [datum.maxslopepos for datum in accels_data]
        y = [speeds.list[datum.maxslopepos] for datum in accels_data]
        ax.scatter(x, y, c='y', marker='o')
    
    accels_per_min = per_min(accels_count)
    total_speed_change = sum([datum.change for datum in accels_data])
    total_accel_dur = Decimal(sum([datum.length for datum in accels_data])) / fps
    mean_speed_change = total_speed_change / Decimal(accels_count)
    mean_peak_accel = sum([datum.maxslope for datum in accels_data]) * fps / Decimal(accels_count)
    mean_accel = sum([(datum.meanslope) for datum in accels_data]) * fps / Decimal(accels_count)
    mean_accel_dur = total_accel_dur / Decimal(accels_count)
    max_speed_change = max([datum.change for datum in accels_data])
    max_peak_accel = max([datum.maxslope for datum in accels_data]) * fps
    max_accel = max([(datum.meanslope) for datum in accels_data]) * fps
    max_accel_dur = Decimal(max([datum.length for datum in accels_data])) / fps
    
    accels_steps, total_accel_step = find_steps(accels_data, cen_dists)
    mean_accel_step = total_accel_step / Decimal(accels_count)
    
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
        'max_accel_dur': max_accel_dur,
        'mean_accel_step': mean_accel_step
    })
    
    if export:
        export_data(accels_data, path + '/' + videoname + '_accels.csv')
    
    if spine_analysis:
        
        # load midline points data
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
            
        # calculate tail bend angles, amplitudes, and body curvatures
        spine_angles = [[] for i in range(l)]
        trunk_angles = [0 for i in range(l)]
        tail_angles = [0 for i in range(l)]
        trunk_curvs = [0 for i in range(l)]
        total_curvs = [0 for i in range(l)]
        spine_angles_filtered = [[] for i in range(l)]
        trunk_curvs_filtered = [0 for i in range(l)]
        total_curvs_filtered = [0 for i in range(l)]
        
        for i in range(l):
            
            # calculate direction from one midline point to another, caudal to cranial
            spine_dirs = []
            for j in range(1, spine_lens[i]):
                spine_dirs.append(cal_direction(spines[i][j - 1], spines[i][j]))
            # calculate tail bend angles. left is +, right is -
            for j in range(2, spine_lens[i]):
                spine_angles[i].append(cal_direction_change(spine_dirs[j - 1], spine_dirs[j - 2]))
                trunk_angles[i] += spine_angles[i][j - 2]
            
            # calculate amplitude. left is +, right is -
            head_len = max(2, spine_lens[i] // 3)
            for j in range(spine_lens[i] - head_len - 2):
                tail_angles[i] += spine_angles[i][j]
            tail_dir = cal_direction(tails[i], spines[i][0])
            tail_angles[i] += cal_direction_change(spine_dirs[0], tail_dir)
            directiono = cal_direction(spines[i][spine_lens[i] - head_len], spines[i][spine_lens[i] - 1])
            tail_angles[i] += cal_direction_change(directiono, spine_dirs[spine_lens[i] - head_len - 2])
            if tail_angles[i] < 0:
                amplitudes[i] = -amplitudes[i]
            
            # calculate body curvature
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
        
        # directions_free is a list of special running average of direction of locomotion
        # turns is derived from directions_free. unit is rad/frame
        # turning left is -, turning right is +
        directions_free = [0 for i in range(l)]
        directions_free[0] = directions[0]
        for i in range(1, l):
            directions_free[i] = directions_free[i - 1] + turns[i] / fps
        directions_free = list_set(directions_free, window=3, start=0, end=l)
        turns_original = list(turns)
        turns = [0 for i in range(l)]
        for i in range(1, l):
            turns[i] = directions_free.list[i] - directions_free.list[i - 1]
        turns = list_set(turns, window=3, start=1, end=l)
        
        def turns_criteria_f(slope, lst):
            if lst > 0.02 and slope < -0.01:
                return True
            elif lst <= 0.02 and slope < 0.001:
                return True
            else:
                return False
        def turns_criteria_b(slope, lst):
            if lst > 0.02 and slope > 0.01:
                return True
            elif lst <= 0.02 and slope > -0.001:
                return True
            else:
                return False
        def turns_criteria_peak(datum):
            if datum.length >= int(min_turn_dur * fps) and datum.height * fps >= min_max_turn_velocity and datum.auc >= min_turn_angle:
                return True
            else:
                return False
        turns_p_data = get_peaks(turns.p_list, turns.p_list, turn_cutoff / fps,
                                 turns_criteria_f, turns_criteria_b, turns_criteria_peak)
        turns_p_count = len(turns_p_data)
        turns_n_data = get_peaks(turns.n_list, turns.n_list, turn_cutoff / fps,
                                 turns_criteria_f, turns_criteria_b, turns_criteria_peak)
        turns_n_count = len(turns_n_data)
        
        if plot_figure:
            fig, ax = plt.subplots()
            #ax.plot(turns.original_list, c='y')
            ax.plot(turns.list)
            scatter_x = [datum.peakpos for datum in turns_p_data]
            scatter_y = [turns.list[datum.peakpos] for datum in turns_p_data]
            ax.scatter(scatter_x, scatter_y, c='b', marker='o')
            for datum in turns_p_data:
                y = [turns.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            x = [datum.peakpos for datum in turns_n_data]
            y = [turns.list[datum.peakpos] for datum in turns_n_data]
            ax.scatter(x, y, c='r', marker='o')
            for datum in turns_n_data:
                y = [turns.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
        
        turns_data = merge(turns_p_data, turns_n_data)
        turns_count = len(turns_data)
        
        turn_count_per_min = per_min(turns_count)
        left_turn_count_per_min = per_min(turns_n_count)
        right_turn_count_per_min = per_min(turns_p_count)
        turn_count_preference = cal_preference(turns_p_count, turns_n_count) # larger if prefer right
        
        total_turn_angle = sum([abs(datum.auc) for datum in turns_data]) * 180 / pi
        total_left_turn_angle = sum([datum.auc for datum in turns_n_data]) * 180 / pi
        total_right_turn_angle = sum([datum.auc for datum in turns_p_data]) * 180 / pi
        
        total_turn_angle_per_min = per_min(total_turn_angle)
        total_left_turn_angle_per_min = per_min(total_left_turn_angle)
        total_right_turn_angle_per_min = per_min(total_right_turn_angle)
        total_turn_angle_preference = cal_preference(total_right_turn_angle, total_left_turn_angle)
        
        mean_turn_angle = total_turn_angle / Decimal(turns_count)
        mean_right_turn_angle = total_right_turn_angle / Decimal(turns_p_count)
        mean_left_turn_angle = total_left_turn_angle / Decimal(turns_n_count)
        mean_turn_angle_preference = cal_preference(mean_right_turn_angle, mean_left_turn_angle)
        
        total_turn_dur = sum([(Decimal(datum.length) / fps) for datum in turns_data])
        total_left_turn_dur = sum([(Decimal(datum.length) / fps) for datum in turns_n_data])
        total_right_turn_dur = sum([(Decimal(datum.length) / fps) for datum in turns_p_data])
        total_turn_dur_preference = cal_preference(total_right_turn_dur, total_left_turn_dur)
        
        mean_turn_dur = total_turn_dur / Decimal(turns_count)
        mean_right_turn_dur = total_right_turn_dur / Decimal(turns_p_count)
        mean_left_turn_dur = total_left_turn_dur / Decimal(turns_n_count)
        mean_turn_dur_preference = cal_preference(mean_right_turn_dur, mean_left_turn_dur)
        
        mean_turn_velocity = sum([abs(datum.height) for datum in turns_data]) * fps * 180 / pi / Decimal(turns_count)
        mean_left_turn_velocity = sum([datum.height for datum in turns_n_data]) * fps * 180 / pi / Decimal(turns_n_count)
        mean_right_turn_velocity = sum([datum.height for datum in turns_p_data]) * fps * 180 / pi / Decimal(turns_p_count)
        mean_turn_velocity_preference = cal_preference(mean_right_turn_velocity, mean_left_turn_velocity)
        
        meandering = total_turn_angle / total_distance
        
        turns_steps, sum_turn_step = find_steps(turns_data, cen_dists)
        mean_turn_step = sum_turn_step / Decimal(turns_count)
        turns_p_steps, sum_turn_p_step = find_steps(turns_p_data, cen_dists)
        mean_turn_right_step = sum_turn_p_step / Decimal(turns_p_count)
        turns_n_steps, sum_turn_n_step = find_steps(turns_n_data, cen_dists)
        mean_turn_right_step = sum_turn_n_step / Decimal(turns_n_count)
        
        def choose_auc(item):
            return item[0].auc
        def choose_y(item):
            return item[1]
        turns_steps_correlation = correlate(turns_steps, choose_auc, choose_y, 3)
        mean_step_low_turn_angle = turns_steps_correlation[0]
        mean_step_mid_turn_angle = turns_steps_correlation[1]
        mean_step_hi_turn_angle = turns_steps_correlation[2]
        turns_steps_correlation = correlate(turns_steps, choose_y, choose_auc, 3)
        mean_turn_angle_low_step = turns_steps_correlation[0]
        mean_turn_angle_mid_step = turns_steps_correlation[1]
        mean_turn_angle_hi_step = turns_steps_correlation[2]
        
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
            'mean_turn_velocity_preference': mean_turn_velocity_preference,
            
            'meandering': meandering
        })
        
        # unit is rad/frame. turning left is +, turning right is -
        trunk_angles = list_set(trunk_angles, window=5, start=0, end=l)
        trunk_angles_ddt = [0 for i in range(l)]
        for i in range(1, l):
            trunk_angles_ddt[i] = trunk_angles.list[i] - trunk_angles.list[i - 1]
        trunk_angles_ddt = list_set(trunk_angles_ddt, window=5, start=1, end=l)
        
        def trunk_angles_p_criteria_f(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def trunk_angles_p_criteria_b(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def trunk_angles_p_criteria_peak(datum):
            if datum.length >= int(min_bend_dur * fps) and datum.maxslope * fps >= min_bend_speed and datum.change >= min_bend_angle:
                return True
            else:
                return False
        trunk_angles_p_data = get_peaks(trunk_angles_ddt.p_list, trunk_angles.list,
                                        bend_cutoff / fps,
                                        trunk_angles_p_criteria_f,
                                        trunk_angles_p_criteria_b,
                                        trunk_angles_p_criteria_peak)
        trunk_angles_p_count = len(trunk_angles_p_data)
        def trunk_angles_n_criteria_f(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def trunk_angles_n_criteria_b(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def trunk_angles_n_criteria_peak(datum):
            if datum.length >= int(min_bend_dur * fps) and abs(datum.maxslope) * fps >= min_bend_speed and abs(datum.change) >= min_bend_angle:
                return True
            else:
                return False
        trunk_angles_n_data = get_peaks(trunk_angles_ddt.n_list, trunk_angles.list,
                                        bend_cutoff / fps,
                                        trunk_angles_n_criteria_f,
                                        trunk_angles_n_criteria_b,
                                        trunk_angles_n_criteria_peak)
        trunk_angles_n_count = len(trunk_angles_n_data)
        
        trunk_angles_data = merge(trunk_angles_p_data, trunk_angles_n_data)
        trunk_angles_count = len(trunk_angles_data)
        
        # classify tail bends into uni and bi
        trunk_angles_uni_data = []
        trunk_angles_uni_left_data = []
        trunk_angles_uni_right_data = []
        trunk_angles_bi_data = []
        trunk_angles_bi_left_data = []
        trunk_angles_bi_right_data = []
        trunk_angles_rec_data = []
        trunk_angles_rec_left_data = []
        trunk_angles_rec_right_data = []
        for datum in trunk_angles_data:
            first_third = (datum.start * 2 + datum.end) / 3
            second_third = (datum.start + datum.end * 2) / 3
            if datum.end > datum.start:
                if first_third > 0:
                    trunk_angles_uni_data.append(copy(datum))
                    trunk_angles_uni_left_data.append(copy(datum))
                elif second_third > 0:
                    trunk_angles_bi_data.append(copy(datum))
                    trunk_angles_bi_left_data.append(copy(datum))
                else:
                    trunk_angles_rec_data.append(copy(datum))
                    trunk_angles_rec_left_data.append(copy(datum))
            elif datum.start > datum.end:
                if first_third < 0:
                    trunk_angles_uni_data.append(copy(datum))
                    trunk_angles_uni_right_data.append(copy(datum))
                elif second_third < 0:
                    trunk_angles_bi_data.append(copy(datum))
                    trunk_angles_bi_right_data.append(copy(datum))
                else:
                    trunk_angles_rec_data.append(copy(datum))
                    trunk_angles_rec_right_data.append(copy(datum))
        trunk_angles_uni_count = len(trunk_angles_uni_data)
        trunk_angles_uni_left_count = len(trunk_angles_uni_left_data)
        trunk_angles_uni_right_count = len(trunk_angles_uni_right_data)
        trunk_angles_bi_count = len(trunk_angles_bi_data)
        trunk_angles_bi_left_count = len(trunk_angles_bi_left_data)
        trunk_angles_bi_right_count = len(trunk_angles_bi_right_data)
        trunk_angles_rec_count = len(trunk_angles_rec_data)
        trunk_angles_rec_left_count = len(trunk_angles_rec_left_data)
        trunk_angles_rec_right_count = len(trunk_angles_rec_right_data)
        
        if plot_figure:
            
            fig, ax = plt.subplots()
            ax.plot(trunk_angles.list)
            ax.plot(trunk_angles_ddt.list, c='y')
            for datum in trunk_angles_p_data:
                y = [trunk_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in trunk_angles_n_data:
                y = [trunk_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            
            fig, ax = plt.subplots()
            ax.plot(trunk_angles.list)
            for datum in trunk_angles_uni_left_data:
                y = [trunk_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in trunk_angles_uni_right_data:
                y = [trunk_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            
            fig, ax = plt.subplots()
            ax.plot(trunk_angles.list)
            for datum in trunk_angles_bi_left_data:
                y = [trunk_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in trunk_angles_bi_right_data:
                y = [trunk_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
        
        trunk_uni_bend_count_per_min = per_min(trunk_angles_uni_count)
        trunk_uni_left_bend_count_per_min = per_min(trunk_angles_uni_left_count)
        trunk_uni_right_bend_count_per_min = per_min(trunk_angles_uni_right_count)
        trunk_bi_bend_count_per_min = per_min(trunk_angles_bi_count)
        trunk_bi_left_bend_count_per_min = per_min(trunk_angles_bi_left_count)
        trunk_bi_right_bend_count_per_min = per_min(trunk_angles_bi_right_count)
        trunk_bend_count = trunk_angles_uni_count + trunk_angles_bi_count
        trunk_bend_count_per_min = per_min(trunk_bend_count)
        trunk_uni_bend_count_preference = cal_preference(trunk_angles_uni_left_count,
                                                         trunk_angles_uni_right_count)
        trunk_bi_bend_count_preference = cal_preference(trunk_angles_bi_left_count,
                                                        trunk_angles_bi_right_count)
        trunk_bend_count_preference = cal_preference(trunk_angles_uni_count,
                                                     trunk_angles_bi_count)
        
        total_trunk_uni_left_bend_angle = sum([datum.change for datum in trunk_angles_uni_left_data]) * 180 / pi
        total_trunk_uni_right_bend_angle = sum([abs(datum.change) for datum in trunk_angles_uni_right_data]) * 180 / pi
        total_trunk_uni_bend_angle = total_trunk_uni_left_bend_angle + total_trunk_uni_right_bend_angle
        total_trunk_bi_left_bend_angle = sum([datum.change for datum in trunk_angles_bi_left_data]) * 180 / pi
        total_trunk_bi_right_bend_angle = sum([abs(datum.change) for datum in trunk_angles_bi_right_data]) * 180 / pi
        total_trunk_bi_bend_angle = total_trunk_bi_left_bend_angle + total_trunk_bi_right_bend_angle
        total_trunk_bend_angle = total_trunk_uni_bend_angle + total_trunk_bi_bend_angle
        
        total_trunk_uni_bend_angle_per_min = per_min(total_trunk_uni_bend_angle)
        total_trunk_uni_left_bend_angle_per_min = per_min(total_trunk_uni_left_bend_angle)
        total_trunk_uni_right_bend_angle_per_min = per_min(total_trunk_uni_right_bend_angle)
        total_trunk_bi_bend_angle_per_min = per_min(total_trunk_bi_bend_angle)
        total_trunk_bi_left_bend_angle_per_min = per_min(total_trunk_bi_left_bend_angle)
        total_trunk_bi_right_bend_angle_per_min = per_min(total_trunk_bi_right_bend_angle)
        total_trunk_bend_angle_per_min = per_min(total_trunk_bend_angle)
        total_trunk_uni_bend_angle_preference = cal_preference(total_trunk_uni_left_bend_angle,
                                                               total_trunk_uni_right_bend_angle)
        total_trunk_bi_bend_angle_preference = cal_preference(total_trunk_bi_left_bend_angle,
                                                              total_trunk_bi_right_bend_angle)
        total_trunk_bend_angle_preference = cal_preference(total_trunk_uni_bend_angle,
                                                           total_trunk_bi_bend_angle)
        
        mean_trunk_uni_bend_angle = total_trunk_uni_bend_angle / Decimal(trunk_angles_uni_count)
        mean_trunk_uni_left_bend_angle = total_trunk_uni_left_bend_angle / Decimal(trunk_angles_uni_left_count)
        mean_trunk_uni_right_bend_angle = total_trunk_uni_right_bend_angle / Decimal(trunk_angles_uni_right_count)
        mean_trunk_bi_bend_angle = total_trunk_bi_bend_angle / Decimal(trunk_angles_bi_count)
        mean_trunk_bi_left_bend_angle = total_trunk_bi_left_bend_angle / Decimal(trunk_angles_bi_left_count)
        mean_trunk_bi_right_bend_angle = total_trunk_bi_right_bend_angle / Decimal(trunk_angles_bi_right_count)
        mean_trunk_bend_angle = total_trunk_bend_angle / Decimal(trunk_bend_count)
        mean_trunk_uni_bend_angle_preference = cal_preference(mean_trunk_uni_left_bend_angle,
                                                              mean_trunk_uni_right_bend_angle)
        mean_trunk_bi_bend_angle_preference = cal_preference(mean_trunk_bi_left_bend_angle,
                                                             mean_trunk_bi_right_bend_angle)
        mean_trunk_bend_angle_preference = cal_preference(mean_trunk_uni_bend_angle,
                                                          mean_trunk_bi_bend_angle)
        
        total_trunk_uni_left_bend_angular_speed = sum([datum.meanslope for datum in trunk_angles_uni_left_data]) * 180 / pi
        total_trunk_uni_right_bend_angular_speed = sum([abs(datum.meanslope) for datum in trunk_angles_uni_right_data]) * 180 / pi
        total_trunk_uni_bend_angular_speed = total_trunk_uni_left_bend_angular_speed + total_trunk_uni_right_bend_angular_speed
        total_trunk_bi_left_bend_angular_speed = sum([datum.meanslope for datum in trunk_angles_bi_left_data]) * 180 / pi
        total_trunk_bi_right_bend_angular_speed = sum([abs(datum.meanslope) for datum in trunk_angles_bi_right_data]) * 180 / pi
        total_trunk_bi_bend_angular_speed = total_trunk_bi_left_bend_angular_speed + total_trunk_bi_right_bend_angular_speed
        total_trunk_bend_angular_speed = total_trunk_uni_bend_angular_speed + total_trunk_bi_bend_angular_speed
        
        total_trunk_rec_left_angular_speed = sum([datum.meanslope for datum in trunk_angles_rec_left_data]) * 180 / pi
        total_trunk_rec_right_angular_speed = sum([abs(datum.meanslope) for datum in trunk_angles_rec_right_data]) * 180 / pi
        total_trunk_rec_angular_speed = total_trunk_rec_left_angular_speed + total_trunk_rec_right_angular_speed
        
        mean_trunk_uni_left_bend_angular_speed = total_trunk_uni_left_bend_angular_speed / Decimal(trunk_angles_uni_left_count)
        mean_trunk_uni_right_bend_angular_speed = total_trunk_uni_right_bend_angular_speed / Decimal(trunk_angles_uni_right_count)
        mean_trunk_uni_bend_angular_speed = total_trunk_uni_bend_angular_speed / Decimal(trunk_angles_uni_count)
        mean_trunk_bi_left_bend_angular_speed = total_trunk_bi_left_bend_angular_speed / Decimal(trunk_angles_bi_left_count)
        mean_trunk_bi_right_bend_angular_speed = total_trunk_bi_right_bend_angular_speed / Decimal(trunk_angles_bi_right_count)
        mean_trunk_bi_bend_angular_speed = total_trunk_bi_bend_angular_speed / Decimal(trunk_angles_bi_count)
        mean_trunk_bend_angular_speed = total_trunk_bend_angular_speed / Decimal(trunk_angles_count)
        mean_trunk_uni_bend_angular_speed_preference = cal_preference(mean_trunk_uni_left_bend_angular_speed,
                                                                      mean_trunk_uni_right_bend_angular_speed)
        mean_trunk_bi_bend_angular_speed_preference = cal_preference(mean_trunk_bi_left_bend_angular_speed,
                                                                     mean_trunk_bi_right_bend_angular_speed)
        mean_trunk_bend_angular_speed_preference = cal_preference(mean_trunk_uni_bend_angular_speed,
                                                                  mean_trunk_bi_bend_angular_speed)
        
        mean_trunk_rec_left_angular_speed = total_trunk_rec_left_angular_speed / Decimal(trunk_angles_rec_left_count)
        mean_trunk_rec_right_angular_speed = total_trunk_rec_right_angular_speed / Decimal(trunk_angles_rec_right_count)
        mean_trunk_rec_angular_speed = total_trunk_rec_angular_speed / Decimal(trunk_angles_rec_count)
        mean_trunk_rec_angular_speed_preference = cal_preference(mean_trunk_rec_left_angular_speed,
                                                                 mean_trunk_rec_right_angular_speed)
        
        analysis.update({
            'trunk_uni_bend_count_per_min': trunk_uni_bend_count_per_min,
            'trunk_uni_left_bend_count_per_min': trunk_uni_left_bend_count_per_min,
            'trunk_uni_right_bend_count_per_min': trunk_uni_right_bend_count_per_min,
            'trunk_bi_bend_count_per_min': trunk_bi_bend_count_per_min,
            'trunk_bi_left_bend_count_per_min': trunk_bi_left_bend_count_per_min,
            'trunk_bi_right_bend_count_per_min': trunk_bi_right_bend_count_per_min,
            'trunk_bend_count_per_min': trunk_bend_count_per_min,
            'trunk_uni_bend_count_preference': trunk_uni_bend_count_preference,
            'trunk_bi_bend_count_preference': trunk_bi_bend_count_preference,
            'trunk_bend_count_preference': trunk_bend_count_preference,
            
            'total_trunk_uni_bend_angle_per_min': total_trunk_uni_bend_angle_per_min,
            'total_trunk_uni_left_bend_angle_per_min': total_trunk_uni_left_bend_angle_per_min,
            'total_trunk_uni_right_bend_angle_per_min': total_trunk_uni_right_bend_angle_per_min,
            'total_trunk_bi_bend_angle_per_min': total_trunk_bi_bend_angle_per_min,
            'total_trunk_bi_left_bend_angle_per_min': total_trunk_bi_left_bend_angle_per_min,
            'total_trunk_bi_right_bend_angle_per_min': total_trunk_bi_right_bend_angle_per_min,
            'total_trunk_bend_angle_per_min': total_trunk_bend_angle_per_min,
            'total_trunk_uni_bend_angle_preference': total_trunk_uni_bend_angle_preference,
            'total_trunk_bi_bend_angle_preference': total_trunk_bi_bend_angle_preference,
            'total_trunk_bend_angle_preference': total_trunk_bend_angle_preference,
            
            'mean_trunk_uni_bend_angle': mean_trunk_uni_bend_angle,
            'mean_trunk_uni_left_bend_angle': mean_trunk_uni_left_bend_angle,
            'mean_trunk_uni_right_bend_angle': mean_trunk_uni_right_bend_angle,
            'mean_trunk_bi_bend_angle': mean_trunk_bi_bend_angle,
            'mean_trunk_bi_left_bend_angle': mean_trunk_bi_left_bend_angle,
            'mean_trunk_bi_right_bend_angle': mean_trunk_bi_right_bend_angle,
            'mean_trunk_bend_angle': mean_trunk_bend_angle,
            'mean_trunk_uni_bend_angle_preference': mean_trunk_uni_bend_angle_preference,
            'mean_trunk_bi_bend_angle_preference': mean_trunk_bi_bend_angle_preference,
            'mean_trunk_bend_angle_preference': mean_trunk_bend_angle_preference,
            
            'mean_trunk_uni_bend_angular_speed': mean_trunk_uni_bend_angular_speed,
            'mean_trunk_uni_left_bend_angular_speed': mean_trunk_uni_left_bend_angular_speed,
            'mean_trunk_uni_right_bend_angular_speed': mean_trunk_uni_right_bend_angular_speed,
            'mean_trunk_bi_bend_angular_speed': mean_trunk_bi_bend_angular_speed,
            'mean_trunk_bi_left_bend_angular_speed': mean_trunk_bi_left_bend_angular_speed,
            'mean_trunk_bi_right_bend_angular_speed': mean_trunk_bi_right_bend_angular_speed,
            'mean_trunk_bend_angular_speed': mean_trunk_bend_angular_speed,
            'mean_trunk_uni_bend_angular_speed_preference': mean_trunk_uni_bend_angular_speed_preference,
            'mean_trunk_bi_bend_angular_speed_preference': mean_trunk_bi_bend_angular_speed_preference,
            'mean_trunk_bend_angular_speed_preference': mean_trunk_bend_angular_speed_preference,
            
            'mean_trunk_rec_left_angular_speed': mean_trunk_rec_left_angular_speed,
            'mean_trunk_rec_right_angular_speed': mean_trunk_rec_right_angular_speed,
            'mean_trunk_rec_angular_speed': mean_trunk_rec_angular_speed,
            'mean_trunk_rec_angular_speed_preference': mean_trunk_rec_angular_speed_preference
        })
        
        # unit is rad/frame. turning left is +, turning right is -
        tail_angles = list_set(tail_angles, window=5, start=0, end=l)
        tail_angles_ddt = [0 for i in range(l)]
        for i in range(1, l):
            tail_angles_ddt[i] = tail_angles.list[i] - tail_angles.list[i - 1]
        tail_angles_ddt = list_set(tail_angles_ddt, window=5, start=1, end=l)
        
        def tail_angles_p_criteria_f(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def tail_angles_p_criteria_b(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def tail_angles_p_criteria_peak(datum):
            if datum.length >= int(min_bend_dur * fps) and datum.maxslope * fps >= min_bend_speed and datum.change >= min_bend_angle:
                return True
            else:
                return False
        tail_angles_p_data = get_peaks(tail_angles_ddt.p_list, tail_angles.list,
                                       bend_cutoff / fps,
                                       tail_angles_p_criteria_f, tail_angles_p_criteria_b,
                                       tail_angles_p_criteria_peak)
        tail_angles_p_count = len(tail_angles_p_data)
        def tail_angles_n_criteria_f(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def tail_angles_n_criteria_b(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def tail_angles_n_criteria_peak(datum):
            if datum.length >= int(min_bend_dur * fps) and abs(datum.maxslope) * fps >= min_bend_speed and abs(datum.change) >= min_bend_angle:
                return True
            else:
                return False
        tail_angles_n_data = get_peaks(tail_angles_ddt.n_list, tail_angles.list,
                                       bend_cutoff / fps,
                                       tail_angles_n_criteria_f, tail_angles_n_criteria_b,
                                       tail_angles_n_criteria_peak)
        tail_angles_n_count = len(tail_angles_n_data)
        
        tail_angles_data = merge(tail_angles_p_data, tail_angles_n_data)
        tail_angles_count = len(tail_angles_data)
        
        # classify tail bends into uni and bi
        tail_angles_uni_data = []
        tail_angles_uni_left_data = []
        tail_angles_uni_right_data = []
        tail_angles_bi_data = []
        tail_angles_bi_left_data = []
        tail_angles_bi_right_data = []
        for datum in tail_angles_data:
            first_third = (datum.start * 2 + datum.end) / 3
            second_third = (datum.start + datum.end * 2) / 3
            if datum.end > datum.start:
                if first_third > 0:
                    tail_angles_uni_data.append(copy(datum))
                    tail_angles_uni_left_data.append(copy(datum))
                elif second_third > 0:
                    tail_angles_bi_data.append(copy(datum))
                    tail_angles_bi_left_data.append(copy(datum))
            elif datum.start > datum.end:
                if first_third < 0:
                    tail_angles_uni_data.append(copy(datum))
                    tail_angles_uni_right_data.append(copy(datum))
                elif second_third < 0:
                    tail_angles_bi_data.append(copy(datum))
                    tail_angles_bi_right_data.append(copy(datum))
        tail_angles_uni_count = len(tail_angles_uni_data)
        tail_angles_uni_left_count = len(tail_angles_uni_left_data)
        tail_angles_uni_right_count = len(tail_angles_uni_right_data)
        tail_angles_bi_count = len(tail_angles_bi_data)
        tail_angles_bi_left_count = len(tail_angles_bi_left_data)
        tail_angles_bi_right_count = len(tail_angles_bi_right_data)
        
        if plot_figure:
            
            fig, ax = plt.subplots()
            ax.plot(tail_angles.list)
            ax.plot(tail_angles_ddt.list, c='y')
            for datum in tail_angles_p_data:
                y = [tail_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in tail_angles_n_data:
                y = [tail_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            
            fig, ax = plt.subplots()
            ax.plot(tail_angles.list)
            for datum in tail_angles_uni_left_data:
                y = [tail_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in tail_angles_uni_right_data:
                y = [tail_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            
            fig, ax = plt.subplots()
            ax.plot(tail_angles.list)
            for datum in tail_angles_bi_left_data:
                y = [tail_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in tail_angles_bi_right_data:
                y = [tail_angles.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
        
        tail_uni_bend_count_per_min = per_min(tail_angles_uni_count)
        tail_uni_left_bend_count_per_min = per_min(tail_angles_uni_left_count)
        tail_uni_right_bend_count_per_min = per_min(tail_angles_uni_right_count)
        tail_bi_bend_count_per_min = per_min(tail_angles_bi_count)
        tail_bi_left_bend_count_per_min = per_min(tail_angles_bi_left_count)
        tail_bi_right_bend_count_per_min = per_min(tail_angles_bi_right_count)
        tail_bend_count = tail_angles_uni_count + tail_angles_bi_count
        tail_bend_count_per_min = per_min(tail_bend_count)
        tail_uni_bend_count_preference = cal_preference(tail_angles_uni_left_count,
                                                        tail_angles_uni_right_count)
        tail_bi_bend_count_preference = cal_preference(tail_angles_bi_left_count,
                                                       tail_angles_bi_right_count)
        tail_bend_count_preference = cal_preference(tail_angles_uni_count,
                                                    tail_angles_bi_count)
        
        total_tail_uni_left_bend_angle = sum([datum.change for datum in tail_angles_uni_left_data]) * 180 / pi
        total_tail_uni_right_bend_angle = sum([abs(datum.change) for datum in tail_angles_uni_right_data]) * 180 / pi
        total_tail_uni_bend_angle = total_tail_uni_left_bend_angle + total_tail_uni_right_bend_angle
        total_tail_bi_left_bend_angle = sum([datum.change for datum in tail_angles_bi_left_data]) * 180 / pi
        total_tail_bi_right_bend_angle = sum([abs(datum.change) for datum in tail_angles_bi_right_data]) * 180 / pi
        total_tail_bi_bend_angle = total_tail_bi_left_bend_angle + total_tail_bi_right_bend_angle
        total_tail_bend_angle = total_tail_uni_bend_angle + total_tail_bi_bend_angle
        
        total_tail_uni_bend_angle_per_min = per_min(total_tail_uni_bend_angle)
        total_tail_uni_left_bend_angle_per_min = per_min(total_tail_uni_left_bend_angle)
        total_tail_uni_right_bend_angle_per_min = per_min(total_tail_uni_right_bend_angle)
        total_tail_bi_bend_angle_per_min = per_min(total_tail_bi_bend_angle)
        total_tail_bi_left_bend_angle_per_min = per_min(total_tail_bi_left_bend_angle)
        total_tail_bi_right_bend_angle_per_min = per_min(total_tail_bi_right_bend_angle)
        total_tail_bend_angle_per_min = per_min(total_tail_bend_angle)
        total_tail_uni_bend_angle_preference = cal_preference(total_tail_uni_left_bend_angle,
                                                              total_tail_uni_right_bend_angle)
        total_tail_bi_bend_angle_preference = cal_preference(total_tail_bi_left_bend_angle,
                                                             total_tail_bi_right_bend_angle)
        total_tail_bend_angle_preference = cal_preference(total_tail_uni_bend_angle,
                                                          total_tail_bi_bend_angle)
        
        mean_tail_uni_bend_angle = total_tail_uni_bend_angle / Decimal(tail_angles_uni_count)
        mean_tail_uni_left_bend_angle = total_tail_uni_left_bend_angle / Decimal(tail_angles_uni_left_count)
        mean_tail_uni_right_bend_angle = total_tail_uni_right_bend_angle / Decimal(tail_angles_uni_right_count)
        mean_tail_bi_bend_angle = total_tail_bi_bend_angle / Decimal(tail_angles_bi_count)
        mean_tail_bi_left_bend_angle = total_tail_bi_left_bend_angle / Decimal(tail_angles_bi_left_count)
        mean_tail_bi_right_bend_angle = total_tail_bi_right_bend_angle / Decimal(tail_angles_bi_right_count)
        mean_tail_bend_angle = total_tail_bend_angle / Decimal(tail_bend_count)
        mean_tail_uni_bend_angle_preference = cal_preference(mean_tail_uni_left_bend_angle,
                                                             mean_tail_uni_right_bend_angle)
        mean_tail_bi_bend_angle_preference = cal_preference(mean_tail_bi_left_bend_angle,
                                                            mean_tail_bi_right_bend_angle)
        mean_tail_bend_angle_preference = cal_preference(mean_tail_uni_bend_angle,
                                                         mean_tail_bi_bend_angle)
        
        total_tail_uni_left_bend_angular_speed = sum([datum.meanslope for datum in tail_angles_uni_left_data]) * 180 / pi
        total_tail_uni_right_bend_angular_speed = sum([abs(datum.meanslope) for datum in tail_angles_uni_right_data]) * 180 / pi
        total_tail_uni_bend_angular_speed = total_tail_uni_left_bend_angular_speed + total_tail_uni_right_bend_angular_speed
        total_tail_bi_left_bend_angular_speed = sum([datum.meanslope for datum in tail_angles_bi_left_data]) * 180 / pi
        total_tail_bi_right_bend_angular_speed = sum([abs(datum.meanslope) for datum in tail_angles_bi_right_data]) * 180 / pi
        total_tail_bi_bend_angular_speed = total_tail_bi_left_bend_angular_speed + total_tail_bi_right_bend_angular_speed
        total_tail_bend_angular_speed = total_tail_uni_bend_angular_speed + total_tail_bi_bend_angular_speed
        
        mean_tail_uni_left_bend_angular_speed = total_tail_uni_left_bend_angular_speed / Decimal(tail_angles_uni_left_count)
        mean_tail_uni_right_bend_angular_speed = total_tail_uni_right_bend_angular_speed / Decimal(tail_angles_uni_right_count)
        mean_tail_uni_bend_angular_speed = total_tail_uni_bend_angular_speed / Decimal(tail_angles_uni_count)
        mean_tail_bi_left_bend_angular_speed = total_tail_bi_left_bend_angular_speed / Decimal(tail_angles_bi_left_count)
        mean_tail_bi_right_bend_angular_speed = total_tail_bi_right_bend_angular_speed / Decimal(tail_angles_bi_right_count)
        mean_tail_bi_bend_angular_speed = total_tail_bi_bend_angular_speed / Decimal(tail_angles_bi_count)
        mean_tail_bend_angular_speed = total_tail_bend_angular_speed / Decimal(tail_angles_count)
        mean_tail_uni_bend_angular_speed_preference = cal_preference(mean_tail_uni_left_bend_angular_speed,
                                                                     mean_tail_uni_right_bend_angular_speed)
        mean_tail_bi_bend_angular_speed_preference = cal_preference(mean_tail_bi_left_bend_angular_speed,
                                                                    mean_tail_bi_right_bend_angular_speed)
        mean_tail_bend_angular_speed_preference = cal_preference(mean_tail_uni_bend_angular_speed,
                                                                 mean_tail_bi_bend_angular_speed)
        
        analysis.update({
            'tail_uni_bend_count_per_min': tail_uni_bend_count_per_min,
            'tail_uni_left_bend_count_per_min': tail_uni_left_bend_count_per_min,
            'tail_uni_right_bend_count_per_min': tail_uni_right_bend_count_per_min,
            'tail_bi_bend_count_per_min': tail_bi_bend_count_per_min,
            'tail_bi_left_bend_count_per_min': tail_bi_left_bend_count_per_min,
            'tail_bi_right_bend_count_per_min': tail_bi_right_bend_count_per_min,
            'tail_bend_count_per_min': tail_bend_count_per_min,
            'tail_uni_bend_count_preference': tail_uni_bend_count_preference,
            'tail_bi_bend_count_preference': tail_bi_bend_count_preference,
            'tail_bend_count_preference': tail_bend_count_preference,
            
            'total_tail_uni_bend_angle_per_min': total_tail_uni_bend_angle_per_min,
            'total_tail_uni_left_bend_angle_per_min': total_tail_uni_left_bend_angle_per_min,
            'total_tail_uni_right_bend_angle_per_min': total_tail_uni_right_bend_angle_per_min,
            'total_tail_bi_bend_angle_per_min': total_tail_bi_bend_angle_per_min,
            'total_tail_bi_left_bend_angle_per_min': total_tail_bi_left_bend_angle_per_min,
            'total_tail_bi_right_bend_angle_per_min': total_tail_bi_right_bend_angle_per_min,
            'total_tail_bend_angle_per_min': total_tail_bend_angle_per_min,
            'total_tail_uni_bend_angle_preference': total_tail_uni_bend_angle_preference,
            'total_tail_bi_bend_angle_preference': total_tail_bi_bend_angle_preference,
            'total_tail_bend_angle_preference': total_tail_bend_angle_preference,
            
            'mean_tail_uni_bend_angle': mean_tail_uni_bend_angle,
            'mean_tail_uni_left_bend_angle': mean_tail_uni_left_bend_angle,
            'mean_tail_uni_right_bend_angle': mean_tail_uni_right_bend_angle,
            'mean_tail_bi_bend_angle': mean_tail_bi_bend_angle,
            'mean_tail_bi_left_bend_angle': mean_tail_bi_left_bend_angle,
            'mean_tail_bi_right_bend_angle': mean_tail_bi_right_bend_angle,
            'mean_tail_bend_angle': mean_tail_bend_angle,
            'mean_tail_uni_bend_angle_preference': mean_tail_uni_bend_angle_preference,
            'mean_tail_bi_bend_angle_preference': mean_tail_bi_bend_angle_preference,
            'mean_tail_bend_angle_preference': mean_tail_bend_angle_preference,
            
            'mean_tail_uni_bend_angular_speed': mean_tail_uni_bend_angular_speed,
            'mean_tail_uni_left_bend_angular_speed': mean_tail_uni_left_bend_angular_speed,
            'mean_tail_uni_right_bend_angular_speed': mean_tail_uni_right_bend_angular_speed,
            'mean_tail_bi_bend_angular_speed': mean_tail_bi_bend_angular_speed,
            'mean_tail_bi_left_bend_angular_speed': mean_tail_bi_left_bend_angular_speed,
            'mean_tail_bi_right_bend_angular_speed': mean_tail_bi_right_bend_angular_speed,
            'mean_tail_bend_angular_speed': mean_tail_bend_angular_speed,
            'mean_tail_uni_bend_angular_speed_preference': mean_tail_uni_bend_angular_speed_preference,
            'mean_tail_bi_bend_angular_speed_preference': mean_tail_bi_bend_angular_speed_preference,
            'mean_tail_bend_angular_speed_preference': mean_tail_bend_angular_speed_preference
        })
        
        # unit is mm/frame. turning left is +, turning right is -
        amplitudes = list_set(amplitudes, window=5, start=0, end=l)
        amplitudes_ddt = [0 for i in range(l)]
        for i in range(1, l):
            amplitudes_ddt[i] = amplitudes.list[i] - amplitudes.list[i - 1]
        amplitudes_ddt = list_set(amplitudes_ddt, window=5, start=1, end=l)
        
        def amplitudes_p_criteria_f(slope, lst):
            if slope < 0:
                return True
            else:
                return False
        def amplitudes_p_criteria_b(slope, lst):
            if slope < 0:
                return True
            else:
                return False
        def amplitudes_p_criteria_peak(datum):
            if datum.length >= int(min_amplitude_dur * fps) and datum.maxslope * fps >= min_amplitude_dt and datum.change >= min_amplitude:
                return True
            else:
                return False
        amplitudes_p_data = get_peaks(amplitudes_ddt.p_list, amplitudes.list,
                                      amplitude_cutoff / fps,
                                      amplitudes_p_criteria_f, amplitudes_p_criteria_b,
                                      amplitudes_p_criteria_peak)
        amplitudes_p_count = len(amplitudes_p_data)
        def amplitudes_n_criteria_f(slope, lst):
            if slope > 0:
                return True
            else:
                return False
        def amplitudes_n_criteria_b(slope, lst):
            if slope > 0:
                return True
            else:
                return False
        def amplitudes_n_criteria_peak(datum):
            if datum.length >= int(min_amplitude_dur * fps) and abs(datum.maxslope) * fps >= min_amplitude_dt and abs(datum.change) >= min_amplitude:
                return True
            else:
                return False
        amplitudes_n_data = get_peaks(amplitudes_ddt.n_list, amplitudes.list,
                                      amplitude_cutoff / fps,
                                      amplitudes_n_criteria_f, amplitudes_n_criteria_b,
                                      amplitudes_n_criteria_peak)
        amplitudes_n_count = len(amplitudes_n_data)
        
        amplitudes_data = merge(amplitudes_p_data, amplitudes_n_data)
        amplitudes_count = len(amplitudes_data)
        
        # classify tail bends into uni and bi
        amplitudes_uni_data = []
        amplitudes_uni_left_data = []
        amplitudes_uni_right_data = []
        amplitudes_bi_data = []
        amplitudes_bi_left_data = []
        amplitudes_bi_right_data = []
        for datum in amplitudes_data:
            first_third = (datum.start * 2 + datum.end) / 3
            second_third = (datum.start + datum.end * 2) / 3
            if datum.end > datum.start:
                if first_third > 0:
                    amplitudes_uni_data.append(copy(datum))
                    amplitudes_uni_left_data.append(copy(datum))
                elif second_third > 0:
                    amplitudes_bi_data.append(copy(datum))
                    amplitudes_bi_left_data.append(copy(datum))
            elif datum.start > datum.end:
                if first_third < 0:
                    amplitudes_uni_data.append(copy(datum))
                    amplitudes_uni_right_data.append(copy(datum))
                elif second_third < 0:
                    amplitudes_bi_data.append(copy(datum))
                    amplitudes_bi_right_data.append(copy(datum))
        amplitudes_uni_count = len(amplitudes_uni_data)
        amplitudes_uni_left_count = len(amplitudes_uni_left_data)
        amplitudes_uni_right_count = len(amplitudes_uni_right_data)
        amplitudes_bi_count = len(amplitudes_bi_data)
        amplitudes_bi_left_count = len(amplitudes_bi_left_data)
        amplitudes_bi_right_count = len(amplitudes_bi_right_data)
        
        if plot_figure:
        
            fig, ax = plt.subplots()
            ax.plot(amplitudes.list)
            ax.plot(amplitudes_ddt.list, c='y')
            for datum in amplitudes_p_data:
                y = [amplitudes.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in amplitudes_n_data:
                y = [amplitudes.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            
            fig, ax = plt.subplots()
            ax.plot(amplitudes.list)
            for datum in amplitudes_uni_left_data:
                y = [amplitudes.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in amplitudes_uni_right_data:
                y = [amplitudes.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            
            fig, ax = plt.subplots()
            ax.plot(amplitudes.list)
            for datum in amplitudes_bi_left_data:
                y = [amplitudes.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='b')
            for datum in amplitudes_bi_right_data:
                y = [amplitudes.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
        
        amplitudes_uni_count_per_min = per_min(amplitudes_uni_count)
        amplitudes_uni_left_count_per_min = per_min(amplitudes_uni_left_count)
        amplitudes_uni_right_count_per_min = per_min(amplitudes_uni_right_count)
        amplitudes_bi_count_per_min = per_min(amplitudes_bi_count)
        amplitudes_bi_left_count_per_min = per_min(amplitudes_bi_left_count)
        amplitudes_bi_right_count_per_min = per_min(amplitudes_bi_right_count)
        amplitudes_count = amplitudes_uni_count + amplitudes_bi_count
        amplitudes_count_per_min = per_min(amplitudes_count)
        amplitudes_uni_count_preference = cal_preference(amplitudes_uni_left_count,
                                                         amplitudes_uni_right_count)
        amplitudes_bi_count_preference = cal_preference(amplitudes_bi_left_count,
                                                        amplitudes_bi_right_count)
        amplitudes_count_preference = cal_preference(amplitudes_uni_count,
                                                     amplitudes_bi_count)
        
        total_amplitudes_uni_left_angle = sum([datum.change for datum in amplitudes_uni_left_data]) * 180 / pi
        total_amplitudes_uni_right_angle = sum([abs(datum.change) for datum in amplitudes_uni_right_data]) * 180 / pi
        total_amplitudes_uni_angle = total_amplitudes_uni_left_angle + total_amplitudes_uni_right_angle
        total_amplitudes_bi_left_angle = sum([datum.change for datum in amplitudes_bi_left_data]) * 180 / pi
        total_amplitudes_bi_right_angle = sum([abs(datum.change) for datum in amplitudes_bi_right_data]) * 180 / pi
        total_amplitudes_bi_angle = total_amplitudes_bi_left_angle + total_amplitudes_bi_right_angle
        total_amplitudes_angle = total_amplitudes_uni_angle + total_amplitudes_bi_angle
        
        total_amplitudes_uni_angle_per_min = per_min(total_amplitudes_uni_angle)
        total_amplitudes_uni_left_angle_per_min = per_min(total_amplitudes_uni_left_angle)
        total_amplitudes_uni_right_angle_per_min = per_min(total_amplitudes_uni_right_angle)
        total_amplitudes_bi_angle_per_min = per_min(total_amplitudes_bi_angle)
        total_amplitudes_bi_left_angle_per_min = per_min(total_amplitudes_bi_left_angle)
        total_amplitudes_bi_right_angle_per_min = per_min(total_amplitudes_bi_right_angle)
        total_amplitudes_angle_per_min = per_min(total_amplitudes_angle)
        total_amplitudes_uni_angle_preference = cal_preference(total_amplitudes_uni_left_angle,
                                                               total_amplitudes_uni_right_angle)
        total_amplitudes_bi_angle_preference = cal_preference(total_amplitudes_bi_left_angle,
                                                              total_amplitudes_bi_right_angle)
        total_amplitudes_angle_preference = cal_preference(total_amplitudes_uni_angle,
                                                           total_amplitudes_bi_angle)
        
        mean_amplitudes_uni_angle = total_amplitudes_uni_angle / Decimal(amplitudes_uni_count)
        mean_amplitudes_uni_left_angle = total_amplitudes_uni_left_angle / Decimal(amplitudes_uni_left_count)
        mean_amplitudes_uni_right_angle = total_amplitudes_uni_right_angle / Decimal(amplitudes_uni_right_count)
        mean_amplitudes_bi_angle = total_amplitudes_bi_angle / Decimal(amplitudes_bi_count)
        mean_amplitudes_bi_left_angle = total_amplitudes_bi_left_angle / Decimal(amplitudes_bi_left_count)
        mean_amplitudes_bi_right_angle = total_amplitudes_bi_right_angle / Decimal(amplitudes_bi_right_count)
        mean_amplitudes_angle = total_amplitudes_angle / Decimal(amplitudes_count)
        mean_amplitudes_uni_angle_preference = cal_preference(mean_amplitudes_uni_left_angle,
                                                              mean_amplitudes_uni_right_angle)
        mean_amplitudes_bi_angle_preference = cal_preference(mean_amplitudes_bi_left_angle,
                                                             mean_amplitudes_bi_right_angle)
        mean_amplitudes_angle_preference = cal_preference(mean_amplitudes_uni_angle,
                                                          mean_amplitudes_bi_angle)
        
        total_amplitudes_uni_left_angular_speed = sum([datum.meanslope for datum in amplitudes_uni_left_data]) * 180 / pi
        total_amplitudes_uni_right_angular_speed = sum([abs(datum.meanslope) for datum in amplitudes_uni_right_data]) * 180 / pi
        total_amplitudes_uni_angular_speed = total_amplitudes_uni_left_angular_speed + total_amplitudes_uni_right_angular_speed
        total_amplitudes_bi_left_angular_speed = sum([datum.meanslope for datum in amplitudes_bi_left_data]) * 180 / pi
        total_amplitudes_bi_right_angular_speed = sum([abs(datum.meanslope) for datum in amplitudes_bi_right_data]) * 180 / pi
        total_amplitudes_bi_angular_speed = total_amplitudes_bi_left_angular_speed + total_amplitudes_bi_right_angular_speed
        total_amplitudes_angular_speed = total_amplitudes_uni_angular_speed + total_amplitudes_bi_angular_speed
        
        mean_amplitudes_uni_left_angular_speed = total_amplitudes_uni_left_angular_speed / Decimal(amplitudes_uni_left_count)
        mean_amplitudes_uni_right_angular_speed = total_amplitudes_uni_right_angular_speed / Decimal(amplitudes_uni_right_count)
        mean_amplitudes_uni_angular_speed = total_amplitudes_uni_angular_speed / Decimal(amplitudes_uni_count)
        mean_amplitudes_bi_left_angular_speed = total_amplitudes_bi_left_angular_speed / Decimal(amplitudes_bi_left_count)
        mean_amplitudes_bi_right_angular_speed = total_amplitudes_bi_right_angular_speed / Decimal(amplitudes_bi_right_count)
        mean_amplitudes_bi_angular_speed = total_amplitudes_bi_angular_speed / Decimal(amplitudes_bi_count)
        mean_amplitudes_angular_speed = total_amplitudes_angular_speed / Decimal(amplitudes_count)
        mean_amplitudes_uni_angular_speed_preference = cal_preference(mean_amplitudes_uni_left_angular_speed,
                                                                      mean_amplitudes_uni_right_angular_speed)
        mean_amplitudes_bi_angular_speed_preference = cal_preference(mean_amplitudes_bi_left_angular_speed,
                                                                     mean_amplitudes_bi_right_angular_speed)
        mean_amplitudes_angular_speed_preference = cal_preference(mean_amplitudes_uni_angular_speed,
                                                                  mean_amplitudes_bi_angular_speed)
        
        analysis.update({
            'amplitudes_uni_count_per_min': amplitudes_uni_count_per_min,
            'amplitudes_uni_left_count_per_min': amplitudes_uni_left_count_per_min,
            'amplitudes_uni_right_count_per_min': amplitudes_uni_right_count_per_min,
            'amplitudes_bi_count_per_min': amplitudes_bi_count_per_min,
            'amplitudes_bi_left_count_per_min': amplitudes_bi_left_count_per_min,
            'amplitudes_bi_right_count_per_min': amplitudes_bi_right_count_per_min,
            'amplitudes_count_per_min': amplitudes_count_per_min,
            'amplitudes_uni_count_preference': amplitudes_uni_count_preference,
            'amplitudes_bi_count_preference': amplitudes_bi_count_preference,
            'amplitudes_count_preference': amplitudes_count_preference,
            
            'total_amplitudes_uni_angle_per_min': total_amplitudes_uni_angle_per_min,
            'total_amplitudes_uni_left_angle_per_min': total_amplitudes_uni_left_angle_per_min,
            'total_amplitudes_uni_right_angle_per_min': total_amplitudes_uni_right_angle_per_min,
            'total_amplitudes_bi_angle_per_min': total_amplitudes_bi_angle_per_min,
            'total_amplitudes_bi_left_angle_per_min': total_amplitudes_bi_left_angle_per_min,
            'total_amplitudes_bi_right_angle_per_min': total_amplitudes_bi_right_angle_per_min,
            'total_amplitudes_angle_per_min': total_amplitudes_angle_per_min,
            'total_amplitudes_uni_angle_preference': total_amplitudes_uni_angle_preference,
            'total_amplitudes_bi_angle_preference': total_amplitudes_bi_angle_preference,
            'total_amplitudes_angle_preference': total_amplitudes_angle_preference,
            
            'mean_amplitudes_uni_angle': mean_amplitudes_uni_angle,
            'mean_amplitudes_uni_left_angle': mean_amplitudes_uni_left_angle,
            'mean_amplitudes_uni_right_angle': mean_amplitudes_uni_right_angle,
            'mean_amplitudes_bi_angle': mean_amplitudes_bi_angle,
            'mean_amplitudes_bi_left_angle': mean_amplitudes_bi_left_angle,
            'mean_amplitudes_bi_right_angle': mean_amplitudes_bi_right_angle,
            'mean_amplitudes_angle': mean_amplitudes_angle,
            'mean_amplitudes_uni_angle_preference': mean_amplitudes_uni_angle_preference,
            'mean_amplitudes_bi_angle_preference': mean_amplitudes_bi_angle_preference,
            'mean_amplitudes_angle_preference': mean_amplitudes_angle_preference,
            
            'mean_amplitudes_uni_angular_speed': mean_amplitudes_uni_angular_speed,
            'mean_amplitudes_uni_left_angular_speed': mean_amplitudes_uni_left_angular_speed,
            'mean_amplitudes_uni_right_angular_speed': mean_amplitudes_uni_right_angular_speed,
            'mean_amplitudes_bi_angular_speed': mean_amplitudes_bi_angular_speed,
            'mean_amplitudes_bi_left_angular_speed': mean_amplitudes_bi_left_angular_speed,
            'mean_amplitudes_bi_right_angular_speed': mean_amplitudes_bi_right_angular_speed,
            'mean_amplitudes_angular_speed': mean_amplitudes_angular_speed,
            'mean_amplitudes_uni_angular_speed_preference': mean_amplitudes_uni_angular_speed_preference,
            'mean_amplitudes_bi_angular_speed_preference': mean_amplitudes_bi_angular_speed_preference,
            'mean_amplitudes_angular_speed_preference': mean_amplitudes_angular_speed_preference
        })
        
        trunk_curvs_filtered = list_set(trunk_curvs_filtered, window=5, start=0, end=l)
        def trunk_curvs_criteria_f(slope, lst):
            if slope < 0:
                return True
            else:
                return False
        def trunk_curvs_criteria_b(slope, lst):
            if slope > 0:
                return True
            else:
                return False
        def trunk_curvs_criteria_peak(datum):
            if datum.length >= int(min_bend_dur * fps) and datum.maxslope * fps >= min_bend_speed and datum.upheight >= min_bend_angle:
                return True
            else:
                return False
        trunk_curvs_data = get_peaks(trunk_curvs_filtered.list, trunk_curvs_filtered.list,
                                     bend_cutoff / fps, trunk_curvs_criteria_f,
                                     trunk_curvs_criteria_b, trunk_curvs_criteria_peak)
        trunk_curvs_count = len(trunk_curvs_data)
        
        if plot_figure:
            fig, ax = plt.subplots()
            ax.plot(trunk_curvs_filtered.list)
            for datum in trunk_curvs_data:
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                y = [trunk_curvs_filtered.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            x = [datum.peakpos for datum in trunk_curvs_data]
            y = [trunk_curvs_filtered.list[datum.peakpos] for datum in trunk_curvs_data]
            ax.scatter(x, y, c='r', marker='o')
        
        trunk_curv_count_per_min = per_min(trunk_curvs_count)
        total_trunk_curv_angle = sum([datum.upheight for datum in trunk_curvs_data]) * 180 / pi
        total_trunk_curv_angle_per_min = per_min(total_trunk_curv_angle)
        mean_trunk_curv_angle = total_trunk_curv_angle / Decimal(trunk_curvs_count)
        
        analysis.update({
            'trunk_curv_count_per_min': trunk_curv_count_per_min,
            'total_trunk_curv_angle_per_min': total_trunk_curv_angle_per_min,
            'mean_trunk_curv_angle': mean_trunk_curv_angle
        })
        
        if plot_figure:
            fig, ax = plt.subplots()
            x = [datum.upheight for datum in trunk_curvs_data]
            y = [cdist1s[datum.peakpos] for datum in trunk_curvs_data]
            ax.scatter(x, y)
        
        trunk_curvs_cdist1s = [(datum.upheight, cdist1s[datum.peakpos]) for datum in trunk_curvs_data]
        def choose_x(item):
            return item[0]
        trunk_curvs_cdist1s_correlation = correlate(trunk_curvs_cdist1s, choose_y, choose_x, 3)
        mean_trunk_curv_angle_low_cdist1 = trunk_curvs_cdist1s_correlation[0] * 180 / pi
        mean_trunk_curv_angle_mid_cdist1 = trunk_curvs_cdist1s_correlation[1] * 180 / pi
        mean_trunk_curv_angle_hi_cdist1 = trunk_curvs_cdist1s_correlation[2] * 180 / pi
        trunk_curvs_cdist1s_correlation = correlate(trunk_curvs_cdist1s, choose_x, choose_y, 3)
        mean_cdist1_low_trunk_curv_angle = trunk_curvs_cdist1s_correlation[0]
        mean_cdist1_mid_trunk_curv_angle = trunk_curvs_cdist1s_correlation[1]
        mean_cdist1_hi_trunk_curv_angle = trunk_curvs_cdist1s_correlation[2]
        
        trunk_curvs_steps, sum_trunk_curv_step = find_steps(trunk_curvs_data, cen_dists)
        mean_trunk_curv_step = sum_trunk_curv_step / Decimal(trunk_curvs_count)
        def choose_upheight(item):
            return item[0].upheight
        trunk_curvs_steps_correlation = correlate(trunk_curvs_steps, choose_upheight, choose_y, 3)
        mean_step_low_trunk_curv_angle = trunk_curvs_steps_correlation[0]
        mean_step_mid_trunk_curv_angle = trunk_curvs_steps_correlation[1]
        mean_step_hi_trunk_curv_angle = trunk_curvs_steps_correlation[2]
        trunk_curvs_steps_correlation = correlate(trunk_curvs_steps, choose_y, choose_upheight, 3)
        mean_trunk_curv_angle_low_step = trunk_curvs_steps_correlation[0]
        mean_trunk_curv_angle_mid_step = trunk_curvs_steps_correlation[1]
        mean_trunk_curv_angle_hi_step = trunk_curvs_steps_correlation[2]
        
        analysis.update({
            'mean_trunk_curv_angle_low_cdist1': mean_trunk_curv_angle_low_cdist1,
            'mean_trunk_curv_angle_mid_cdist1': mean_trunk_curv_angle_mid_cdist1,
            'mean_trunk_curv_angle_hi_cdist1': mean_trunk_curv_angle_hi_cdist1,
            'mean_cdist1_low_trunk_curv_angle': mean_cdist1_low_trunk_curv_angle,
            'mean_cdist1_mid_trunk_curv_angle': mean_cdist1_mid_trunk_curv_angle,
            'mean_cdist1_hi_trunk_curv_angle': mean_cdist1_hi_trunk_curv_angle,
            
            'mean_trunk_curv_step': mean_trunk_curv_step,
            'mean_step_low_trunk_curv_angle': mean_step_low_trunk_curv_angle,
            'mean_step_mid_trunk_curv_angle': mean_step_mid_trunk_curv_angle,
            'mean_step_hi_trunk_curv_angle': mean_step_hi_trunk_curv_angle,
            'mean_trunk_curv_angle_low_step': mean_trunk_curv_angle_low_step,
            'mean_trunk_curv_angle_mid_step': mean_trunk_curv_angle_mid_step,
            'mean_trunk_curv_angle_hi_step': mean_trunk_curv_angle_hi_step
        })
        
        total_curvs_filtered = list_set(total_curvs_filtered, window=5, start=0, end=l)
        def total_curvs_criteria_f(slope, lst):
            if slope < 0:
                return True
            else:
                return False
        def total_curvs_criteria_b(slope, lst):
            if slope > 0:
                return True
            else:
                return False
        def total_curvs_criteria_peak(datum):
            if datum.length >= int(min_bend_dur * fps) and datum.maxslope * fps >= min_bend_speed and datum.upheight >= min_bend_angle:
                return True
            else:
                return False
        total_curvs_data = get_peaks(total_curvs_filtered.list, total_curvs_filtered.list,
                                     bend_cutoff / fps, total_curvs_criteria_f,
                                     total_curvs_criteria_b, total_curvs_criteria_peak)
        total_curvs_count = len(total_curvs_data)
        
        if plot_figure:
            fig, ax = plt.subplots()
            ax.plot(total_curvs_filtered.list)
            for datum in total_curvs_data:
                x = [i for i in range(datum.startpos, datum.endpos + 1)]
                y = [total_curvs_filtered.list[i] for i in range(datum.startpos, datum.endpos + 1)]
                ax.plot(x, y, c='r')
            x = [datum.peakpos for datum in total_curvs_data]
            y = [total_curvs_filtered.list[datum.peakpos] for datum in total_curvs_data]
            ax.scatter(x, y, c='r', marker='o')
        
        total_curv_count_per_min = per_min(total_curvs_count)
        total_total_curv_angle = sum([datum.upheight for datum in total_curvs_data]) * 180 / pi
        total_total_curv_angle_per_min = per_min(total_total_curv_angle)
        mean_total_curv_angle = total_total_curv_angle / Decimal(total_curvs_count)
        
        analysis.update({
            'total_curv_count_per_min': total_curv_count_per_min,
            'total_total_curv_angle_per_min': total_total_curv_angle_per_min,
            'mean_total_curv_angle': mean_total_curv_angle
        })
        
        if plot_figure:
            fig, ax = plt.subplots()
            x = [datum.upheight for datum in total_curvs_data]
            y = [cdist1s[datum.peakpos] for datum in total_curvs_data]
            ax.scatter(x, y)
        
        total_curvs_cdist1s = [(datum.upheight, cdist1s[datum.peakpos]) for datum in total_curvs_data]
        total_curvs_cdist1s_correlation = correlate(total_curvs_cdist1s, choose_y, choose_x, 3)
        mean_total_curv_angle_low_cdist1 = total_curvs_cdist1s_correlation[0] * 180 / pi
        mean_total_curv_angle_mid_cdist1 = total_curvs_cdist1s_correlation[1] * 180 / pi
        mean_total_curv_angle_hi_cdist1 = total_curvs_cdist1s_correlation[2] * 180 / pi
        total_curvs_cdist1s_correlation = correlate(total_curvs_cdist1s, choose_x, choose_y, 3)
        mean_cdist1_low_total_curv_angle = total_curvs_cdist1s_correlation[0]
        mean_cdist1_mid_total_curv_angle = total_curvs_cdist1s_correlation[1]
        mean_cdist1_hi_total_curv_angle = total_curvs_cdist1s_correlation[2]
        
        total_curvs_steps, sum_total_curv_step = find_steps(total_curvs_data, cen_dists)
        mean_total_curv_step = sum_total_curv_step / Decimal(total_curvs_count)
        def choose_upheight(item):
            return item[0].upheight
        total_curvs_steps_correlation = correlate(total_curvs_steps, choose_upheight, choose_y, 3)
        mean_step_low_total_curv_angle = total_curvs_steps_correlation[0]
        mean_step_mid_total_curv_angle = total_curvs_steps_correlation[1]
        mean_step_hi_total_curv_angle = total_curvs_steps_correlation[2]
        total_curvs_steps_correlation = correlate(total_curvs_steps, choose_y, choose_upheight, 3)
        mean_total_curv_angle_low_step = total_curvs_steps_correlation[0]
        mean_total_curv_angle_mid_step = total_curvs_steps_correlation[1]
        mean_total_curv_angle_hi_step = total_curvs_steps_correlation[2]
        
        analysis.update({
            'mean_total_curv_angle_low_cdist1': mean_total_curv_angle_low_cdist1,
            'mean_total_curv_angle_mid_cdist1': mean_total_curv_angle_mid_cdist1,
            'mean_total_curv_angle_hi_cdist1': mean_total_curv_angle_hi_cdist1,
            'mean_cdist1_low_total_curv_angle': mean_cdist1_low_total_curv_angle,
            'mean_cdist1_mid_total_curv_angle': mean_cdist1_mid_total_curv_angle,
            'mean_cdist1_hi_total_curv_angle': mean_cdist1_hi_total_curv_angle,
            
            'mean_total_curv_step': mean_total_curv_step,
            'mean_step_low_total_curv_angle': mean_step_low_total_curv_angle,
            'mean_step_mid_total_curv_angle': mean_step_mid_total_curv_angle,
            'mean_step_hi_total_curv_angle': mean_step_hi_total_curv_angle,
            'mean_total_curv_angle_low_step': mean_total_curv_angle_low_step,
            'mean_total_curv_angle_mid_step': mean_total_curv_angle_mid_step,
            'mean_total_curv_angle_hi_step': mean_total_curv_angle_hi_step
        })
        '''
        total_curvs_accels = [(datum.upheight, (peak_data(), 0)) for datum in total_curvs_data]
        i = 0
        j = 0
        while i < total_curvs_count - 1:
            while accels_data[j].startpos < total_curvs_data[i].startpos and j < accels_count:
                j += 1
            if accels_data[j].startpos >= total_curvs_data[i].startpos and accels_data[j].startpos < total_curvs_data[i + 1].startpos:
                total_curvs_accels[i] = (total_curvs_data[i].upheight, accel_dists[j])
            i += 1
        i = total_curvs_count - 1
        while accels_data[j].startpos < total_curvs_data[i].startpos and j < accels_count:
            j += 1
        if accels_data[j].startpos >= total_curvs_data[i].startpos:
            total_curvs_accels[i] = (total_curvs_data[i].upheight, accel_dists[j])
        
        def choose_meanslope(item):
            return item[1][0].meanslope
        total_curvs_accels_correlation = correlate(total_curvs_accels, choose_meanslope,
                                                   choose_x, 3)
        mean_total_curv_angle_low_accel = total_curvs_accels_correlation[0] * 180 / pi
        mean_total_curv_angle_mid_accel = total_curvs_accels_correlation[1] * 180 / pi
        mean_total_curv_angle_hi_accel = total_curvs_accels_correlation[2] * 180 / pi
        def choose_change(item):
            return item[1][0].change
        total_curvs_accels_correlation = correlate(total_curvs_accels, choose_change,
                                                   choose_x, 3)
        mean_total_curv_angle_low_speed_change = total_curvs_accels_correlation[0] * 180 / pi
        mean_total_curv_angle_mid_speed_change = total_curvs_accels_correlation[1] * 180 / pi
        mean_total_curv_angle_hi_speed_change = total_curvs_accels_correlation[2] * 180 / pi
        def choose_dist(item):
            return item[1][1]
        total_curvs_accels_correlation = correlate(total_curvs_accels, choose_dist,
                                                   choose_x, 3)
        mean_total_curv_angle_low_dist = total_curvs_accels_correlation[0] * 180 / pi
        mean_total_curv_angle_mid_dist = total_curvs_accels_correlation[1] * 180 / pi
        mean_total_curv_angle_hi_dist = total_curvs_accels_correlation[2] * 180 / pi
        total_curvs_accels_correlation = correlate(total_curvs_accels, choose_x,
                                                   choose_meanslope, 3)
        mean_accel_low_total_curv_angle = total_curvs_accels_correlation[0] * fps
        mean_accel_mid_total_curv_angle = total_curvs_accels_correlation[1] * fps
        mean_accel_hi_total_curv_angle = total_curvs_accels_correlation[2] * fps
        total_curvs_accels_correlation = correlate(total_curvs_accels, choose_x,
                                                   choose_change, 3)
        mean_speed_change_low_total_curv_angle = total_curvs_accels_correlation[0]
        mean_speed_change_mid_total_curv_angle = total_curvs_accels_correlation[1]
        mean_speed_change_hi_total_curv_angle = total_curvs_accels_correlation[2]
        total_curvs_accels_correlation = correlate(total_curvs_accels, choose_x,
                                                   choose_dist, 3)
        mean_dist_low_total_curv_angle = total_curvs_accels_correlation[0]
        mean_dist_mid_total_curv_angle = total_curvs_accels_correlation[1]
        mean_dist_hi_total_curv_angle = total_curvs_accels_correlation[2]
        
        analysis.update({
            'mean_total_curv_angle_low_accel': mean_total_curv_angle_low_accel,
            'mean_total_curv_angle_mid_accel': mean_total_curv_angle_mid_accel,
            'mean_total_curv_angle_hi_accel': mean_total_curv_angle_hi_accel,
            'mean_total_curv_angle_low_speed_change': mean_total_curv_angle_low_speed_change,
            'mean_total_curv_angle_mid_speed_change': mean_total_curv_angle_mid_speed_change,
            'mean_total_curv_angle_hi_speed_change': mean_total_curv_angle_hi_speed_change,
            'mean_total_curv_angle_low_dist': mean_total_curv_angle_low_dist,
            'mean_total_curv_angle_mid_dist': mean_total_curv_angle_mid_dist,
            'mean_total_curv_angle_hi_dist': mean_total_curv_angle_hi_dist,
            'mean_accel_low_total_curv_angle': mean_accel_low_total_curv_angle,
            'mean_accel_mid_total_curv_angle': mean_accel_mid_total_curv_angle,
            'mean_accel_hi_total_curv_angle': mean_accel_hi_total_curv_angle,
            'mean_speed_change_low_total_curv_angle': mean_speed_change_low_total_curv_angle,
            'mean_speed_change_mid_total_curv_angle': mean_speed_change_mid_total_curv_angle,
            'mean_speed_change_hi_total_curv_angle': mean_speed_change_hi_total_curv_angle,
            'mean_dist_low_total_curv_angle': mean_dist_low_total_curv_angle,
            'mean_dist_mid_total_curv_angle': mean_dist_mid_total_curv_angle,
            'mean_dist_hi_total_curv_angle': mean_dist_hi_total_curv_angle
        })
        '''
        if export:
            export_data(turns_data, path + '/' + videoname + '_turns.csv')
            export_data(trunk_angles_uni_data, path + '/' + videoname + '_trunk_angles_uni.csv')
            export_data(trunk_angles_bi_data, path + '/' + videoname + '_trunk_angles_bi.csv')
            export_data(tail_angles_uni_data, path + '/' + videoname + '_tail_angles_uni.csv')
            export_data(tail_angles_bi_data, path + '/' + videoname + '_tail_angles_bi.csv')
            export_data(amplitudes_uni_data, path + '/' + videoname + '_amplitudes_uni.csv')
            export_data(amplitudes_bi_data, path + '/' + videoname + '_amplitudes_bi.csv')
            export_data(trunk_curvs_data, path + '/' + videoname + '_trunk_curvs.csv')
            export_data(total_curvs_data, path + '/' + videoname + '_total_curvs.csv')
        
    with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
        for key in analysis:
            f.write(key + ', ' + str(analysis[key]) + '\n')
    print('Analysis of ' + videoname + ' complete.')
    analyses.append(analysis)
    
    with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
        for key in settings:
            f.write(key + ', ' + str(settings[key]) + '\n')

with open('analyses.csv', 'w') as f:
    for key in analysis:
        f.write(key + ', ')
        for video in analyses:
            f.write(str(video[key]) + ', ')
        f.write('\n')
print('All analyses complete.')
