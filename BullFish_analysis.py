import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.signal import find_peaks
from copy import copy, deepcopy
from BullFish_pkg.math import pyth, cal_direction, cal_direction_change
from BullFish_pkg.general import csvtodict, load_settings

default_settings = {
    "tank_x": 210,
    "tank_y": 144,
    "plot_figure": 0,
    "sampling": 2,
    "speed_limit": 2000,
    "accel_cutoff": 100,
    "min_accel_dur": 0.02,
    "min_max_accel": 0,
    "min_speed_change": 10,
    "spine_analysis": 1,
    "turn_cutoff": 2,
    "min_turn_dur": 0.02,
    "min_max_turn_velocity": 0,
    "min_turn_angle": 0.087,
    "bend_cutoff": 2,
    "min_bend_dur": 0.02,
    "min_bend_speed": 0,
    "min_bend_angle": 0.035,
    "amp_cutoff": 50,
    "min_amp_dur": 0.02,
    "min_amp_dt": 0,
    "min_amp": 2,
    'front_window': 0.05,
    'back_window': 0.05,
    "correlation_portion": 3,
    'use_s1': 0,
    'analysis_extend': 0}

settings = load_settings('analysis', default_settings)

def per_min(value):
    return value * 60 / total_time

def cal_preference(p, n):
    try:
        return (p - n) / (p + n)
    except:
        return 0

def errors_correct(lst, errors):
    error_groups = []
    errors_count = len(errors)
    i = 0
    while i < errors_count:
        error_group = [errors[i]]
        j = i + 1
        while j < errors_count:
            if errors[j] - errors[j - 1] == 1:
                error_group.append(errors[j])
                j += 1
            else:
                i = j - 1
                break
        error_groups.append((error_group[0], len(error_group)))
        i += 1
    for error_group in error_groups:
        start = error_group[0]
        length = error_group[1]
        end = start + length
        for i in range(length):
            lst[start + i] = (lst[start - 1] * (length - i) + lst[end] * (i + 1)) / (length + 1)
    return lst

class list_set:
    def __init__(self, inputlist, start, end, window=0):
        self.original_list = copy(inputlist)
        self.list = copy(inputlist)
        if window > 2:
            for i in range(start, start + window // 2):
                self.list[i] = sum(self.original_list[start:(i + window // 2 + 1)]) / (i + window // 2 - start + 1)
            for i in range(start + window // 2 + 1, end - window // 2):
                self.list[i] = sum(self.original_list[(i - window // 2):(i + window // 2 + 1)]) / window
            for i in range(end - window // 2, end):
                self.list[i] = sum(self.original_list[(i - window // 2):end]) / (end - i + window // 2)
        self.p_list = [(item if item > 0 else 0) for item in self.list]
        self.n_list = [(-item if item < 0 else 0) for item in self.list]

# identify local maxima in list1 to characterize peaks in list2
def get_peaks(list1, list2, prominence, criteria_f, criteria_b, criteria_peak):
    peaks, _ = find_peaks(list1, prominence=prominence)
    peaks = [int(peak) for peak in peaks]
    data = []
    l = len(list2)
    for peak in peaks:
        datum = {
            'start': 0,
            'startpos': 0,
            'length': 0,
            'end': 0,
            'endpos': 0,
            'peakpos': 0,
            'uplength': 0,
            'upheight': 0,
            'height': 0,
            'change': 0,
            'auc': 0,
            'meanslope': 0,
            'maxslope': 0,
            'maxslopepos': 0}
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
            meanslope = upheight / length if length > 0 else 0
            maxslope = 0
            maxslopepos = 0
            for j in range(startpos + 1, peakpos + 1):
                slope = list2[j] - list2[j - 1]
                if maxslope < slope:
                    maxslope = slope
                    maxslopepos = j
            datum['peakpos'] = peakpos
            datum['uplength'] = uplength
            datum['upheight'] = upheight
            datum['auc'] = auc
        else:
            height = list2[endpos]
            change = height - list2[startpos]
            meanslope = change / length if length > 0 else 0
            maxslope = 0
            maxslopepos = 0
            for j in range(startpos + 1, endpos + 1):
                slope = abs(list2[j] - list2[j - 1])
                if maxslope < slope:
                    maxslope = slope
                    maxslopepos = j
        datum['start'] = start
        datum['startpos'] = startpos
        datum['length'] = length
        datum['end'] = end
        datum['endpos'] = endpos
        datum['height'] = height
        datum['change'] = change
        datum['meanslope'] = meanslope
        datum['maxslope'] = maxslope
        datum['maxslopepos'] = maxslopepos
        if criteria_peak(datum):
            data.append(datum)
    return data
                
def remove_duplicates(lst):
    dct = {}
    for item in lst:
        dct.update({str(item['startpos']): item})
    nlst = []
    for key in dct.keys():
        nlst.append(dct[key])
    return nlst

class step_datum:
    def __init__(self, step):
        self.accel = step
        self.turns = []
        self.turns_peaks = []
        self.bends = []
        self.bends_peaks = []
        self.extrinsic = {
            'current speed': 0,
            'step length': 0,
            'step speed change': 0,
            'step velocity change': 0,
            'step accel': 0,
            'step dur': 0,
            'turn angle overall': 0,
            'turn laterality': 'neutral',
            'turn angle max': 0,
            'turn angle mean': 0,
            'turn angle sum': 0,
            'turn dur max': 0,
            'turn dur sum': 0,
            'turn angular velocity max': 0,
            'turn angular velocity mean': 0}
        self.intrinsic = {
            'bend angle max': 0,
            'bend angle laterality': 'neutral',
            #'bend angle max left': 0,
            #'bend angle max right': 0,
            'bend angle mean': 0,
            'bend angle traveled total': 0,
            #'bend angular velocity max': 0,
            'bend angular velocity mean': 0,
            'period mean': 0,
            'bend dur': 0,
            'bend amp max': 0,
            'bend amp mean': 0,
            'bend pos': 0}

def df_dict(df, name, name2=''):
    dic = df.to_dict()
    new_dic = {}
    for key1 in dic.keys():
        if type(key1) == str:
            for key2 in dic[key1].keys():
                new_dic.update({name + '_' + key1 + '_' + key2: dic[key1][key2]})
        elif type(key1) == tuple:
            name1 = ''
            for i in key1:
                name1 += ('_' + i)
            for key2 in dic[key1].keys():
                new_dic.update({name + '_' + name2 + '_' + key2 + name1: dic[key1][key2]})
    return new_dic

def compare(dic, item1, item2):
    new_dic = {}
    for key in dic.keys():
        if item1 in key:
            new_key = key.replace(item1, item1 + ' vs ' + item2)
            key2 = key.replace(item1, item2)
            value = cal_preference(dic[key], dic[key2])
            new_dic.update({new_key: value})
    return new_dic

def plot_data(ax, lst, startpos, endpos, c='c'):
    x = [i for i in range(startpos, endpos)]
    y = [lst[i] for i in range(startpos, endpos)]
    ax.plot(x, y, c)

def plot_scatter(ax, steps, key, lst, c='c', marker='o'):
    scatter_x = []
    scatter_y = []
    for step in steps:
        x = step.choose(key)
        y = lst[x]
        scatter_x.append(x)
        scatter_y.append(y)
    ax.scatter(scatter_x, scatter_y, c=c, marker=marker)

analyses = []
analyses_extend = []

# search for videos in the current directory
for file in os.listdir('.'):
    
    # check if the file is a video in the supported formats
    filename = os.fsdecode(file)
    filename_split = os.path.splitext(filename)
    supported_formats = {'.avi', '.mp4'}
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
    metadata = csvtodict(path + '/' + videoname + '_metadata.csv')
    l = metadata['video_end'] - metadata['video_start']
    if metadata['swimarea_x'] > metadata['swimarea_y']:
        ratio = settings['tank_x'] / metadata['swimarea_x']
    else:
        ratio = settings['tank_y'] / metadata['swimarea_y']
    
    # load essential tracking data
    with open(path + '/' + videoname + '_cen.csv', 'r') as f:
        cen = [[cell for cell in row] for row in csv.reader(f)]
        cen.pop(0)
        for i in range(l):
            cen[i] = (float(cen[i][0]) * ratio, float(cen[i][1]) * ratio)
    
    # obtain a list of speed at each frame
    cen_dists = [0 for i in range(l)]
    speeds = [0 for i in range(l)]
    for i in range(settings['sampling'], l, settings['sampling']):
        cen_dists[i] = pyth(cen[i], cen[i - settings['sampling']]) / settings['sampling']
        speeds[i] = cen_dists[i] * metadata['fps']
    for i in range(0, settings['sampling']):
        cen_dists[i] = cen_dists[settings['sampling']]
    for i in range(settings['sampling'] * 2, l, settings['sampling']):
        for j in range(i - settings['sampling'] + 1, i):
            cen_dists[j] = (cen_dists[i - settings['sampling']] * (i - j) + cen_dists[i] * (j - (i - settings['sampling']))) / settings['sampling']
            speeds[j] = cen_dists[j] * metadata['fps']
    
    error1_frames = []
    for i in range(l):
        if speeds[i] > settings['speed_limit']:
            error1_frames.append(i)
    cen_dists = errors_correct(cen_dists, error1_frames)
    speeds = errors_correct(speeds, error1_frames)
    
    # obtain total distance travelled and average speed
    total_distance = sum(cen_dists)
    total_time = l / metadata['fps']
    speed_avg = total_distance / total_time
    
    # determine whether the fish is freezing for each frame
    freeze = [0 for i in range(l)]
    for i in range(int(metadata['fps'] * 3), l):
        cdist1 = pyth(cen[int(i - metadata['fps'] * 2)], cen[int(i - metadata['fps'] * 3)])
        cdist2 = pyth(cen[int(i - metadata['fps'])], cen[int(i - metadata['fps'] * 2)])
        cdist3 = pyth(cen[i], cen[int(i - metadata['fps'])])
        if cdist1 < 1 and cdist2 < 1 and cdist3 < 1:
            for j in range(int(i - metadata['fps'] * 3 + 1), i + 1):
                freeze[j] = 1
        elif cdist1 > 1 and cdist2 > 1 and cdist3 > 1:
            for j in range(int(i - metadata['fps'] * 2 + 1), i + 1):
                freeze[j] = 0
    
    # obtain parameters related to freezing
    total_freeze_time = sum(freeze) / metadata['fps']
    freeze_percent = total_freeze_time / total_time * 100
    active_time = total_time - total_freeze_time
    active_speed = total_distance / active_time
    freeze_count = 0
    for i in range(2, l):
        if freeze[i] - freeze[i - 1] == 1:
            freeze_count += 1
    freeze_freq = per_min(freeze_count)
    
    # obtain a list of speeds with running average and obtain max speed
    speeds = list_set(speeds, start=1, end=l, window=5)
    max_speed = max(speeds.list)
    
    # obtain a list of speed measured over 1s
    cdist1s = [0 for i in range(l)]
    for i in range(1, l):
        start = max(1, round(i + 1 - metadata['fps'] / 2))
        end = min(l, round(i + 1 + metadata['fps'] / 2))
        cdist1s[i] = sum([cen_dists[j] for j in range(start, end)]) * metadata['fps'] / (end - start)
    max_distance_1s = max(cdist1s)
    
    # output basic extrinsic to analysis dictionary
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
        'freeze_freq': freeze_freq}
    
    # obtain a list of acceleration over each frame, with running average
    # the unit of acceleration here is mm/s/frame
    accels = [0 for i in range(l)]
    for i in range(1, l):
        accels[i] = speeds.list[i] - speeds.list[i - 1]
    accels[0] = accels[1]
    accels = list_set(accels, start=1, end=l, window=5)
    
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
        if datum['length'] >= int(settings['min_accel_dur'] * metadata['fps']):
            if datum['maxslope'] * metadata['fps'] >= settings['min_max_accel']:
                if datum['change'] >= settings['min_speed_change']:
                    return True
        return False
    accels_data = get_peaks(accels.p_list, speeds.list, settings['accel_cutoff'] / metadata['fps'],
                            accels_criteria_f, accels_criteria_b, accels_criteria_peak)
    accels_data = remove_duplicates(accels_data)
    accels_count = len(accels_data)
    
    if accels_count == 0:
        print('No detectable movement')
        with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
            for key in analysis:
                f.write(key + ',' + str(analysis[key]) + '\n')
        print('Analysis of ' + videoname + ' complete.')
        analyses.append(analysis)
        continue
    '''
    ke = [0 for i in range(accels_count)]
    for i in range(accels_count):
        ke[i] = speeds_data[i]['height'] ** 2 - speeds.list[speeds_data[i]['startpos']] ** 2
    '''
    if settings['plot_figure']:
        fig, ax = plt.subplots()
        ax.plot([i * metadata['fps'] for i in accels.p_list])
        ax.plot(speeds.list, c='b')
        for datum in accels_data:
            x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
            y = [speeds.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
            ax.plot(x, y, c='r')
        x = [datum['maxslopepos'] for datum in accels_data]
        y = [speeds.list[datum['maxslopepos']] for datum in accels_data]
        ax.scatter(x, y, c='y', marker='o')
    
    # load midline points data
    spine_lens = [0 for i in range(l)]
    with open(path + '/' + videoname + '_spine.csv', 'r') as f:
        spines = [[cell for cell in row] for row in csv.reader(f)]
        spines.pop(0)
        for i in range(l):
            spine_lens[i] = int(spines[i][0])
            spine_temp = []
            for j in range(1, spine_lens[i] + 1):
                spine_temp.append([float(spines[i][j * 2 - 1]), float(spines[i][j * 2])])
            spines[i] = spine_temp
    if not settings['use_s1']:
        with open(path + '/' + videoname + '_s0s.csv', 'r') as f:
            s0_temp = [[cell for cell in row] for row in csv.reader(f)]
            s0_temp.pop(0)
            for i in range(l):
                spines[i].insert(0, [float(s0_temp[i][0]), float(s0_temp[i][1])])
                spine_lens[i] += 1
    directions = [0 for i in range(l)]
    turns = [0 for i in range(l)]
    with open(path + '/' + videoname + '_directions.csv', 'r') as f:
        direction_temp = [[cell for cell in row] for row in csv.reader(f)]
        direction_temp.pop(0)
        for i in range(l):
            directions[i] = float(direction_temp[i][0])
            turns[i] = float(direction_temp[i][1])
    heads = [0 for i in range(l)]
    with open(path + '/' + videoname + '_sn+1s.csv', 'r') as f:
        temp = [[cell for cell in row] for row in csv.reader(f)]
        temp.pop(0)
        for i in range(l):
            heads[i] = [float(temp[i][0]), float(temp[i][1])]
    
    # calculate bend angles, amps
    spine_angles = [[] for i in range(l)]
    angles = [0 for i in range(l)]
    for i in range(l):
        if spine_lens[i] < 3:
            continue
        # calculate direction from one midline point to another, caudal to cranial
        spine_dirs = []
        for j in range(1, spine_lens[i]):
            spine_dirs.append(cal_direction(spines[i][j - 1], spines[i][j]))
        # calculate bend angles. left is +, right is -
        for j in range(2, spine_lens[i]):
            spine_angles[i].append(cal_direction_change(spine_dirs[j - 1], spine_dirs[j - 2]))
            angles[i] += spine_angles[i][j - 2]
    
    amps = [0 for i in range(l)]
    trunk_amps = [[0 for j in range(spine_lens[i] - 2)] for i in range(l)]
    for i in range(l):
        if spines[i][spine_lens[i] - 1][0] == spines[i][spine_lens[i] - 2][0]:
            for j in range(spine_lens[i] - 2):
                trunk_amps[i][j] = abs(spines[i][j][0] - spines[i][spine_lens[i] - 1][0]) * ratio
        else:
            m = (spines[i][spine_lens[i] - 1][1] - spines[i][spine_lens[i] - 2][1]) / (spines[i][spine_lens[i] - 1][0] - spines[i][spine_lens[i] - 2][0])
            c = spines[i][spine_lens[i] - 1][1] - m * spines[i][spine_lens[i] - 1][0]
            for j in range(spine_lens[i] - 2):
                trunk_amps[i][j] = abs(m * spines[i][j][0] - spines[i][j][1] + c) / math.sqrt(m ** 2 + 1) * ratio
        if spine_lens[i] > 2:
            amps[i] = trunk_amps[i][0]
    
    fish_segs = [[0 for j in range(spine_lens[i])] for i in range(l)]
    for i in range(l):
        for j in range(spine_lens[i] - 1):
            fish_segs[i][j] = pyth(spines[i][j], spines[i][j + 1])
        fish_segs[i][spine_lens[i] - 1] = pyth(spines[i][spine_lens[i] - 1], heads[i])
    fish_lengths = [sum(fish_segs[i]) for i in range(l)]
    bend_poss = [0 for i in range(l)]
    for i in range(l):
        for j in range(spine_lens[i] - 2):
            if trunk_amps[i][j] < settings['min_amp']:
                if j >= 1:
                    bend_poss[i] = sum(fish_segs[i][0:j]) / fish_lengths[i]
                break
        
    error_frames = []
    with open(path + '/' + videoname + '_errors.csv', 'r') as f:
        for row in csv.reader(f):
            for cell in row:
                if cell.isnumeric():
                    error_frames.append(int(cell))
    angles = errors_correct(angles, error_frames)
    amps = errors_correct(amps, error_frames)
    
    # directions_free is a list of special running average of direction of locomotion
    # turns is derived from directions_free. unit is rad/frame
    # turning left is -, turning right is +
    directions_free = [0 for i in range(l)]
    directions_free[0] = directions[0]
    for i in range(1, l):
        directions_free[i] = directions_free[i - 1] + turns[i] / metadata['fps']
    directions_free = list_set(directions_free, start=0, end=l, window=3)
    turns_original = list(turns)
    turns = [0 for i in range(l)]
    for i in range(1, l):
        turns[i] = directions_free.list[i] - directions_free.list[i - 1]
    turns = list_set(turns, start=1, end=l, window=3)
    
    total_left_turn = sum([(abs(turn) if turn < 0 else 0) for turn in turns.original_list])
    total_right_turn = sum([(turn if turn > 0 else 0) for turn in turns.original_list])
    total_turn = total_left_turn + total_right_turn
    turn_preference = cal_preference(total_left_turn, total_right_turn)
    meandering = total_turn / total_distance
    analysis.update({
        'total_turn': total_turn,
        'total_left_turn': total_left_turn,
        'total_right_turn': total_right_turn,
        'turn_preference': turn_preference,
        'meandering': meandering})
    
    def turns_p_criteria_f(slope, lst):
        if slope < 0.02:
            return True
        else:
            return False
    def turns_p_criteria_b(slope, lst):
        if slope < 0.02:
            return True
        else:
            return False
    def turns_p_criteria_peak(datum):
        if datum['length'] >= int(settings['min_turn_dur'] * metadata['fps']):
            if datum['maxslope'] * metadata['fps'] >= settings['min_max_turn_velocity']:
                if datum['change'] >= settings['min_turn_angle']:
                    return True
        return False
    turns_p_data = get_peaks(turns.p_list, directions_free.list,
                             settings['turn_cutoff'] / metadata['fps'],
                             turns_p_criteria_f, turns_p_criteria_b,
                             turns_p_criteria_peak)
    def turns_n_criteria_f(slope, lst):
        if slope > -0.02:
            return True
        else:
            return False
    def turns_n_criteria_b(slope, lst):
        if slope > -0.02:
            return True
        else:
            return False
    def turns_n_criteria_peak(datum):
        if datum['length'] >= int(settings['min_turn_dur'] * metadata['fps']):
            if abs(datum['maxslope']) * metadata['fps'] >= settings['min_max_turn_velocity']:
                if abs(datum['change']) >= settings['min_turn_angle']:
                    return True
        return False
    turns_n_data = get_peaks(turns.n_list, directions_free.list,
                             settings['turn_cutoff'] / metadata['fps'],
                             turns_n_criteria_f, turns_n_criteria_b,
                             turns_n_criteria_peak)
    
    if settings['plot_figure']:
        fig, ax = plt.subplots()
        ax.plot(directions_free.list)
        for datum in turns_p_data:
            plot_data(ax, directions_free.list, datum['startpos'], datum['endpos'] + 1, 'b')
        for datum in turns_n_data:
            plot_data(ax, directions_free.list, datum['startpos'], datum['endpos'] + 1, 'r')
    
    turns_data = deepcopy(turns_p_data)
    turns_data.extend(deepcopy(turns_n_data))
    turns_data.sort(key=lambda a: a['startpos'])
    turns_steps = []
    for peak in turns_data:
        turn = {
            'angle': abs(peak['change']),
            'dur': peak['length'] / metadata['fps'],
            'angular velocity': abs(peak['meanslope']) / peak['length'] * metadata['fps']}
        if peak['change'] > 0:
            turn.update({'laterality': 'right'})
        else:
            turn.update({'laterality': 'left'})
        turns_steps.append(turn)
    turns_df = pd.DataFrame(turns_steps)
    
    # unit is rad/frame. turning left is +, turning right is -
    angles = list_set(angles, start=0, end=l, window=5)
    angles_ddt = [0 for i in range(l)]
    for i in range(1, l):
        angles_ddt[i] = angles.list[i] - angles.list[i - 1]
    angles_ddt = list_set(angles_ddt, start=1, end=l, window=5)
    
    def angles_p_criteria_f(slope, lst):
        if slope < 0.02:
            return True
        else:
            return False
    def angles_p_criteria_b(slope, lst):
        if slope < 0.02:
            return True
        else:
            return False
    def angles_p_criteria_peak(datum):
        if datum['length'] >= int(settings['min_bend_dur'] * metadata['fps']):
            if datum['maxslope'] * metadata['fps'] >= settings['min_bend_speed']:
                if datum['change'] >= settings['min_bend_angle']:
                    return True
        return False
    angles_p_data = get_peaks(angles_ddt.p_list, angles.list,
                              settings['bend_cutoff'] / metadata['fps'],
                              angles_p_criteria_f, angles_p_criteria_b,
                              angles_p_criteria_peak)
    angles_p_data = remove_duplicates(angles_p_data)
    def angles_n_criteria_f(slope, lst):
        if slope > -0.02:
            return True
        else:
            return False
    def angles_n_criteria_b(slope, lst):
        if slope > -0.02:
            return True
        else:
            return False
    def angles_n_criteria_peak(datum):
        if datum['length'] >= int(settings['min_bend_dur'] * metadata['fps']):
            if abs(datum['maxslope']) * metadata['fps'] >= settings['min_bend_speed']:
                if abs(datum['change']) >= settings['min_bend_angle']:
                    return True
        return False
    angles_n_data = get_peaks(angles_ddt.n_list, angles.list,
                              settings['bend_cutoff'] / metadata['fps'],
                              angles_n_criteria_f, angles_n_criteria_b,
                              angles_n_criteria_peak)
    angles_n_data = remove_duplicates(angles_n_data)
    
    angles_neutral = copy(angles.list)
    angles_neutral_count = l
    for peak in angles_p_data:
        for i in range(peak['startpos'], peak['endpos'] + 1):
            angles_neutral[i] = 0
            angles_neutral_count -= 1
    for peak in angles_n_data:
        for i in range(peak['startpos'], peak['endpos'] + 1):
            angles_neutral[i] = 0
            angles_neutral_count -= 1
    angle_neutral = sum(angles_neutral) / angles_neutral_count
    analysis.update({'angle_neutral': angle_neutral})
    
    angles_data = deepcopy(angles_p_data)
    angles_data.extend(deepcopy(angles_n_data))
    angles_data.sort(key=lambda a: a['startpos'])
    angles_bends = []
    for peak in angles_data:
        start = angles.original_list[peak['startpos']]
        end = angles.original_list[peak['endpos']]
        angle_change = end - start
        bend = {
            'angle change': abs(angle_change),
            'angle end': abs(end),
            'dur': peak['length'] / metadata['fps'],
            'angular velocity': abs(angle_change) / peak['length'] * metadata['fps'],
            'amp change': abs(amps[peak['endpos']] - amps[peak['startpos']]),
            'amp end': abs(amps[peak['endpos']]),
            'bend pos': bend_poss[peak['endpos']]}
        if angle_change > 0:
            bend.update({'laterality': 'left'})
            if start * 0.8 + end * 0.2 > angle_neutral:
                bend.update({'recoil': False})
            else:
                bend.update({'recoil': True})
        else:
            bend.update({'laterality': 'right'})
            if start * 0.8 + end * 0.2 < angle_neutral:
                bend.update({'recoil': False})
            else:
                bend.update({'recoil': True})
        angles_bends.append(bend)
    angles_df = pd.DataFrame(angles_bends)
    
    front = round(settings['front_window'] * metadata['fps'])
    back = round(settings['back_window'] * metadata['fps'])
    steps = [step_datum(datum) for datum in accels_data]
    steps_count = len(steps)
    
    for i in range(steps_count):
        if i >= 1:
            startpos = (steps[i - 1].accel['endpos'] + steps[i].accel['startpos'] + 1) // 2
        else:
            startpos = max(0, steps[i].accel['startpos'] - metadata['fps'] // 5)
        if i <= steps_count - 2:
            endpos = (steps[i].accel['endpos'] + steps[i + 1].accel['startpos'] - 1) // 2
        else:
            endpos = min(l - 1, steps[i].accel['endpos'] + metadata['fps'] // 5)
        for j in range(len(turns_data)):
            if turns_data[j]['startpos'] >= startpos and turns_data[j]['startpos'] <= endpos:
                turns_data[j].update({'belong': j})
                steps[i].turns.append(turns_steps[j])
                steps[i].turns_peaks.append(turns_data[j])
        for j in range(len(angles_data)):
            if angles_data[j]['startpos'] >= startpos and angles_data[j]['startpos'] <= endpos:
                angles_data[j].update({'belong': j})
                steps[i].bends.append(angles_bends[j])
                steps[i].bends_peaks.append(angles_data[j])
        if len(steps[i].bends) >= 1 and steps[i].bends[0]['recoil']:
            steps[i].bends.pop(0)
            steps[i].bends_peaks[0]['belong'] = None
            steps[i].bends_peaks.pop(0)
        steps[i].turns_count = len(steps[i].turns)
        steps[i].bends_count = len(steps[i].bends)
    
    for i in range(steps_count):
        startpos = steps[i].accel['startpos']
        if i <= steps_count - 2:
            endpos = steps[i + 1].accel['startpos']
        else:
            endpos = l - 1
        steps[i].extrinsic.update({
            'current speed': cdist1s[startpos],
            'step length': sum(cen_dists[startpos:endpos]),
            'step speed change': steps[i].accel['change'],
            'step accel': steps[i].accel['meanslope'] * metadata['fps'],
            'step dur': (endpos - startpos) / metadata['fps']})
        
    for step in steps:
        
        if step.turns_count >= 1:
            turns_durs = [turn['dur'] for turn in step.turns]
            turns_angles = [turn['angle'] for turn in step.turns]
            turns_angular_velocitys = [turn['angular velocity'] for turn in step.turns]
            step.extrinsic.update({
                'turn angle max': max(turns_angles),
                'turn angle sum': sum(turns_angles),
                'turn angle mean': sum(turns_angles) / step.turns_count,
                'turn dur max': max(turns_durs),
                'turn dur sum': sum(turns_durs),
                'turn angular velocity max': max(turns_angular_velocitys),
                'turn angular velocity mean': sum(turns_angular_velocitys) / step.turns_count})
            turn_angle_overall = directions_free.list[step.turns_peaks[step.turns_count - 1]['endpos']] - directions_free.list[step.turns_peaks[0]['startpos']]
            if turn_angle_overall > settings['min_turn_angle']:
                step.extrinsic.update({'turn laterality': 'right'})
            elif turn_angle_overall < -settings['min_turn_angle']:
                step.extrinsic.update({'turn laterality': 'left'})
            else:
                step.extrinsic.update({'turn laterality': 'neutral'})
            step.extrinsic.update({'turn angle overall': abs(turn_angle_overall)})
            a = speeds.list[step.accel['startpos']]
            b = speeds.list[step.accel['endpos']]
            step.extrinsic.update({'step velocity change': math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(abs(turn_angle_overall)))})
        
        if step.bends_count == 0:
            step.mode = 'UK'
        elif step.bends_count == 1:
            step.mode = 'HT'
            step.intrinsic.update({
                'bend angle max': step.bends[0]['angle change'],
                'bend angle mean': step.bends[0]['angle change'],
                'bend angle laterality': step.bends[0]['laterality'],
                'bend angle traveled total': step.bends[0]['angle change'],
                'bend dur': step.bends[0]['dur'],
                'period mean': step.bends[0]['dur'] * 4,
                #'bend angular velocity max': step.bends[0]['angular velocity'],
                'bend angular velocity mean': step.bends[0]['angular velocity'],
                'bend amp max': step.bends[0]['amp change'],
                'bend amp mean': step.bends[0]['amp change'],
                'bend pos': step.bends[0]['bend pos']})
            '''if step.intrinsic['bend angle laterality'] == 'left':
                step.intrinsic['bend angle max left'] = step.intrinsic['bend angle max']
            else:
                step.intrinsic['bend angle max right'] = step.intrinsic['bend angle max']'''
        elif step.bends_count == 2:
            if step.bends[0]['laterality'] != step.bends[1]['laterality']:
                step.mode = 'HT'
                step.intrinsic.update({
                    'bend angle max': step.bends[0]['angle change'],
                    'bend angle mean': step.bends[0]['angle change'],
                    'bend angle laterality': step.bends[0]['laterality'],
                    'bend angle traveled total': step.bends[0]['angle change'] + step.bends[1]['angle change'],
                    'bend dur': step.bends[0]['dur'] + step.bends[1]['dur'],
                    'period mean': step.bends[0]['dur'] * 4,
                    #'bend angular velocity max': step.bends[0]['angular velocity'],
                    'bend angular velocity mean': (step.bends[0]['angular velocity'] + step.bends[1]['angular velocity']) / 2,
                    'bend amp max': step.bends[0]['amp change'],
                    'bend amp mean': step.bends[0]['amp change'],
                    'bend pos': step.bends[0]['bend pos']})
                '''if step.intrinsic['bend angle laterality'] == 'left':
                    step.intrinsic['bend angle max left'] = step.intrinsic['bend angle max']
                else:
                    step.intrinsic['bend angle max right'] = step.intrinsic['bend angle max']'''
            else:
                step.mode = 'UK'
        elif step.bends_count >= 3:
            step.mode = 'MT'
            angles_traveled = [step.bends[i]['angle change'] for i in range(step.bends_count)]
            angular_velocitys = [bend['angular velocity'] for bend in step.bends]
            durs = [bend['dur'] for bend in step.bends]
            angles_reached = [step.bends[i]['angle end'] for i in range(step.bends_count - 1)]
            angles_reached_sum = 0
            angle_max = 0
            angle_max_left = 0
            angle_max_right = 0
            amps_traveled = [step.bends[i]['amp change'] for i in range(step.bends_count)]
            bend_pos = [step.bends[i]['bend pos'] for i in range(step.bends_count - 1)]
            for angle in angles_reached:
                angles_reached_sum += abs(angle)
                if angle > 0 and angle > angle_max_left:
                    angle_max_left = angle
                elif angle < 0 and abs(angle) > angle_max_right:
                    angle_max_right = abs(angle)
            if angle_max_left > angle_max_right + settings['min_bend_angle']:
                angle_max = angle_max_left
                angle_laterality = 'left'
            elif angle_max_right > angle_max_left + settings['min_bend_angle']:
                angle_max = angle_max_right
                angle_laterality = 'right'
            else:
                angle_max = max(angle_max_left, angle_max_right)
                angle_laterality = 'neutral'
            step.intrinsic.update({
                'bend angle traveled total': sum(angles_traveled),
                'bend angle mean': angles_reached_sum / (step.bends_count - 1),
                'bend angle max': angle_max,
                'bend angle laterality': angle_laterality,
                #'bend angle max left': angle_max_left,
                #'bend angle max right': angle_max_right,
                'bend dur': sum(durs),
                'period mean': sum(durs[1:(step.bends_count - 1)]) / (step.bends_count - 2) * 2,
                #'bend angular velocity max': max(angular_velocitys),
                'bend angular velocity mean': sum(angular_velocitys) / step.bends_count,
                'bend amp max': max(amps_traveled),
                'bend amp mean': sum(amps_traveled) / step.bends_count,
                'bend pos': max(bend_pos)})
    
    if settings['plot_figure']:
        fig, ax = plt.subplots()
        speeds_scaled = [speed / 100 for speed in speeds.list]
        ax.plot(speeds_scaled, 'c')
        ax.plot(turns.list, 'y')
        ax.plot(angles.list, 'm')
        for step in steps:
            if step.mode == 'HT':
                c = 'g'
            elif step.mode == 'MT':
                c = 'b'
            else:
                c = 'r'
            plot_data(ax, speeds_scaled, step.accel['startpos'], step.accel['endpos'] + 1, c)
            for peak in step.turns_peaks:
                plot_data(ax, turns.list, peak['startpos'], peak['endpos'] + 1, c)
            for peak in step.bends_peaks:
                plot_data(ax, angles.list, peak['startpos'], peak['endpos'] + 1, c)
    
    for step in steps:
        step.proprtys = copy(step.extrinsic)
        step.proprtys.update(copy(step.intrinsic))
        step.proprtys.update({'mode': step.mode})
    steps_df = pd.DataFrame([step.proprtys for step in steps])
    
    analysis.update({'turn_count': len(turns_df)})
    methods = ['sum', 'mean', 'std', 'max']
    methods2 = ['mean', 'std', 'max']
    turns_methods = {
        'angle': methods,
        'dur': methods,
        'angular velocity': methods2}
    turns_describe = df_dict(turns_df.agg(turns_methods), 'turn')
    analysis.update(turns_describe)
    analysis.update({'turn_left_count': len(turns_df[turns_df['laterality'] == 'left']),
                     'turn_right_count': len(turns_df[turns_df['laterality'] == 'right'])})
    analysis.update({'turn_left vs right_count': cal_preference(analysis['turn_left_count'], analysis['turn_right_count'])})
    turns_lr_describe = df_dict(turns_df.groupby('laterality').agg(turns_methods), 'turn')
    analysis.update(turns_lr_describe)
    turns_lr_compare = compare(turns_lr_describe, 'left', 'right')
    analysis.update(turns_lr_compare)
    angles_methods = {
        'angle change': methods,
        'dur': methods,
        'angular velocity': methods2,
        'amp change': methods2}
    analysis.update({'bend_count': len(angles_df)})
    angles_describe = df_dict(angles_df.agg(angles_methods), 'bend')
    analysis.update(angles_describe)
    analysis.update({'bend_left_count': len(angles_df[angles_df['laterality'] == 'left']),
                     'bend_right_count': len(angles_df[angles_df['laterality'] == 'right'])})
    analysis.update({'bend_left vs right_count': cal_preference(analysis['bend_left_count'], analysis['bend_right_count'])})
    angles_lr_describe = df_dict(angles_df.groupby('laterality').agg(angles_methods), 'bend')
    analysis.update(angles_lr_describe)
    angles_lr_compare = compare(angles_lr_describe, 'left', 'right')
    analysis.update(angles_lr_compare)
    steps_methods = {
        'step length': methods2,
        'step speed change': methods2,
        'step velocity change': methods2,
        'step accel': methods2,
        'step dur': methods2,
        'turn angle overall': methods2,
        'turn angle max': methods2,
        'turn dur sum': methods2,
        'turn angular velocity max': methods2,
        'bend angle max': methods2,
        'bend angle mean': methods2,
        'bend angle traveled total': methods2,
        'bend angular velocity mean': methods2,
        'bend amp max': methods2,
        'bend amp mean': methods2,
        'period mean': methods2,
        'bend dur': methods2,
        'bend pos': methods2
        }
    analysis.update({'step_count': len(steps_df)})
    steps_df = steps_df[steps_df['mode'] != 'UK']
    steps_describe = df_dict(steps_df.agg(steps_methods), 'step')
    analysis.update(steps_describe)
    analysis.update({'step_mode_HT_count': len(steps_df[steps_df['mode'] == 'HT']),
                     'step_mode_MT_count': len(steps_df[steps_df['mode'] == 'MT'])})
    analysis.update({'step_mode_HT vs MT_count': cal_preference(analysis['step_mode_HT_count'], analysis['step_mode_MT_count'])})
    steps_htmt_describe = df_dict(steps_df.groupby('mode').agg(steps_methods), 'step', 'mode')
    analysis.update(steps_htmt_describe)
    steps_htmt_compare = compare(steps_htmt_describe, 'HT', 'MT')
    analysis.update(steps_htmt_compare)
    analysis.update({'step_turn_left_count': len(steps_df[steps_df['turn laterality'] == 'left']),
                     'step_turn_right_count': len(steps_df[steps_df['turn laterality'] == 'right'])})
    analysis.update({'step_turn_left vs right_count': cal_preference(analysis['step_turn_left_count'], analysis['step_turn_right_count'])})
    steps_turn_lr_describe = df_dict(steps_df.groupby('turn laterality').agg(steps_methods), 'step', 'turn')
    analysis.update(steps_turn_lr_describe)
    steps_turn_lr_compare = compare(steps_turn_lr_describe, 'left', 'right')
    analysis.update(steps_turn_lr_compare)
    analysis.update({'step_bend_left_count': len(steps_df[steps_df['bend angle laterality'] == 'left']),
                     'step_bend_right_count': len(steps_df[steps_df['bend angle laterality'] == 'right'])})
    analysis.update({'step_bend_left vs right_count': cal_preference(analysis['step_bend_left_count'], analysis['step_bend_right_count'])})
    steps_bend_lr_describe = df_dict(steps_df.groupby('bend angle laterality').agg(steps_methods), 'step', 'bend')
    analysis.update(steps_bend_lr_describe)
    steps_bend_lr_compare = compare(steps_bend_lr_describe, 'left', 'right')
    analysis.update(steps_bend_lr_compare)
    
    intervals_df = steps_df.describe()
    intrinsics = step_datum(None).intrinsic
    extrinsics = step_datum(None).extrinsic
    correlate_ex_df = pd.DataFrame()
    for i in extrinsics.keys():
        if type(extrinsics[i]) == str:
            continue
        df1 = steps_df.loc[steps_df[i] <= intervals_df.loc['25%'][i], intrinsics.keys()]
        df2 = steps_df.loc[(steps_df[i] > intervals_df.loc['25%'][i]) & (steps_df[i] < intervals_df.loc['75%'][i]), intrinsics.keys()]
        df3 = steps_df.loc[steps_df[i] >= intervals_df.loc['75%'][i], intrinsics.keys()]
        df1_mean = pd.DataFrame(df1.mean(numeric_only=True)).T
        df1_mean['stratify by'] = i + '_1'
        df2_mean = pd.DataFrame(df2.mean(numeric_only=True)).T
        df2_mean['stratify by'] = i + '_2'
        df3_mean = pd.DataFrame(df3.mean(numeric_only=True)).T
        df3_mean['stratify by'] = i + '_3'
        mean_df = pd.concat([df1_mean, df2_mean, df3_mean])
        correlate_ex_df = pd.concat([correlate_ex_df, mean_df])
    correlate_ex_df = correlate_ex_df.set_index('stratify by')
    correlate_in_df = pd.DataFrame()
    for i in intrinsics.keys():
        if type(intrinsics[i]) == str:
            continue
        df1 = steps_df.loc[steps_df[i] <= intervals_df.loc['25%'][i], extrinsics.keys()]
        df2 = steps_df.loc[(steps_df[i] > intervals_df.loc['25%'][i]) & (steps_df[i] < intervals_df.loc['75%'][i]), extrinsics.keys()]
        df3 = steps_df.loc[steps_df[i] >= intervals_df.loc['75%'][i], extrinsics.keys()]
        df1_mean = pd.DataFrame(df1.mean(numeric_only=True)).T
        df1_mean['stratify by'] = i + '_1'
        df2_mean = pd.DataFrame(df2.mean(numeric_only=True)).T
        df2_mean['stratify by'] = i + '_2'
        df3_mean = pd.DataFrame(df3.mean(numeric_only=True)).T
        df3_mean['stratify by'] = i + '_3'
        mean_df = pd.concat([df1_mean, df2_mean, df3_mean])
        correlate_in_df = pd.concat([correlate_in_df, mean_df])
    correlate_in_df = correlate_in_df.set_index('stratify by')
    analysis.update(df_dict(correlate_ex_df, 'step'))
    analysis.update(df_dict(correlate_in_df, 'step'))
    
    turns_df.to_csv(path + '/' + videoname + '_turns_df.csv')
    angles_df.to_csv(path + '/' + videoname + '_angles_df.csv')
    steps_df.to_csv(path + '/' + videoname + '_steps_df.csv')
    
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
