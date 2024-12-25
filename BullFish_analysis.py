import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats
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
    "min_speed_change": 0,
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
    'min_step_turn': 0.175,
    'front_window': 0.05,
    'back_window': 0.05,
    'use_s1': 0}

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
        end = min(start + length, l - 1)
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
            'maxslopepos': 0,
            'belong': None}
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
    def __init__(self):
        self.startpos = 0
        self.coastpos = 0
        self.endpos = 0
        self.accel = None
        self.turns = []
        self.turns_peaks = []
        self.bends = []
        self.bends_peaks = []
        self.properties = {
            'current speed': 0,
            'step length': 0,
            'speed change': 0,
            'velocity change': 0,
            'accel': 0,
            'step dur': 0,
            'turn angle': None,
            'turn laterality': 'neutral',
            'turn dur': None,
            'turn angular velocity': None,
            'bend angle reached': 0,
            'bend laterality': 'neutral',
            'bend angle traveled': 0,
            'bend angular velocity': 0,
            'bend dur': 0,
            'bend pos': 0,
            'mode': 'UK'}

def agg(df, analysis_dict, methods):
    analysis_list = []
    df_agg = df.agg(methods)
    for i in df_agg.columns:
        for j in df_agg.index:
            if df_agg.at[j, i] == math.nan:
                continue
            value_dict = copy(analysis_dict)
            value_dict.update({'Stratify': None,
                               'Parameter': i,
                               'Method': j,
                               'Value': df_agg.at[j, i]})
            analysis_list.append(value_dict)
    value_dict = copy(analysis_dict)
    value_dict.update({'Stratify': None,
                       'Parameter': None,
                       'Method': 'count',
                       'Value': len(df)})
    analysis_list.append(value_dict)
    return analysis_list
'''
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
'''
def compare(s1, s2):
    result = scipy.stats.ttest_ind(s1, s2, equal_var=False)
    ci = result.confidence_interval()
    dic = {
        's1 mean': s1.mean(),
        's2 mean': s2.mean(),
        't': result.statistic,
        'low t': ci[0],
        'hi t': ci[1],
        'p': result.pvalue}
    return dic
'''
def stratify(df):
    
    intervals_df = df.describe()
    axis1 = step_datum().properties
    stratify_df = pd.DataFrame()
    for i in axis1.keys():
        if type(axis1[i]) == str:
            continue
        axis2 = copy(axis1)
        axis2.pop(i)
        axis2 = axis2.keys()
        df1 = df.loc[df[i] <= intervals_df.loc['25%'][i], axis2]
        df2 = df.loc[(df[i] > intervals_df.loc['25%'][i]) & (df[i] < intervals_df.loc['75%'][i]), axis2]
        df3 = df.loc[df[i] >= intervals_df.loc['75%'][i], axis2]
        df1_mean = pd.DataFrame(df1.mean(numeric_only=True)).T
        df1_mean['stratify by'] = i + '_1'
        df2_mean = pd.DataFrame(df2.mean(numeric_only=True)).T
        df2_mean['stratify by'] = i + '_2'
        df3_mean = pd.DataFrame(df3.mean(numeric_only=True)).T
        df3_mean['stratify by'] = i + '_3'
        mean_df = pd.concat([df1_mean, df2_mean, df3_mean])
        stratify_df = pd.concat([stratify_df, mean_df])
    stratify_df = stratify_df.set_index('stratify by')
    return stratify_df
'''
def stratify(df, analysis_dict):
    analysis_list = []
    intervals_df = df.describe()
    axis1 = step_datum().properties
    for i in axis1.keys():
        if type(axis1[i]) == str:
            continue
        axis2 = copy(axis1)
        axis2.pop(i)
        axis2 = axis2.keys()
        df1 = df.loc[df[i] <= intervals_df.loc['25%'][i], axis2]
        df2 = df.loc[(df[i] > intervals_df.loc['25%'][i]) & (df[i] < intervals_df.loc['75%'][i]), axis2]
        df3 = df.loc[df[i] >= intervals_df.loc['75%'][i], axis2]
        df1_mean = df1.mean(numeric_only=True)
        df2_mean = df2.mean(numeric_only=True)
        df3_mean = df3.mean(numeric_only=True)
        for j in df1_mean.index:
            value_dict = copy(analysis_dict)
            value_dict.update({'Stratify': i + '_low',
                               'Parameter': j,
                               'Method': 'mean',
                               'Value': df1_mean.at[j]})
            analysis_list.append(value_dict)
            value_dict = copy(analysis_dict)
            value_dict.update({'Stratify': i + '_mid',
                               'Parameter': j,
                               'Method': 'mean',
                               'Value': df2_mean.at[j]})
            analysis_list.append(value_dict)
            value_dict = copy(analysis_dict)
            value_dict.update({'Stratify': i + '_high',
                               'Parameter': j,
                               'Method': 'mean',
                               'Value': df3_mean.at[j]})
            analysis_list.append(value_dict)
    return analysis_list

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

videonames = []
analyses = []
analyses_df = pd.DataFrame()
steps_all = pd.DataFrame()
steps_comparisons_all = pd.DataFrame()
rps_df = pd.DataFrame()
first_video = True

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
    videonames.append(videoname)
    
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
        analyses.append(analysis)
        print('Analysis of ' + videoname + ' complete.')
        with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
            for key in settings:
                f.write(key + ', ' + str(settings[key]) + '\n')
        continue
    
    if not settings['spine_analysis']:
        with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
            for key in analysis:
                f.write(key + ', ' + str(analysis[key]) + '\n')
        analyses.append(analysis)
        print('Analysis of ' + videoname + ' complete.')
        with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
            for key in settings:
                f.write(key + ', ' + str(settings[key]) + '\n')
        continue
    
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
    
    fish_length_med = float(pd.DataFrame(fish_lengths).median().iloc[0])
    analysis.update({'fish_length': fish_length_med})
    
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
    '''
    if settings['plot_figure']:
        fig, ax = plt.subplots()
        ax.plot(directions_free.list)
        for datum in turns_p_data:
            plot_data(ax, directions_free.list, datum['startpos'], datum['endpos'] + 1, 'b')
        for datum in turns_n_data:
            plot_data(ax, directions_free.list, datum['startpos'], datum['endpos'] + 1, 'r')
    '''
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
            'bend pos': bend_poss[peak['endpos']]}
        if angle_change > 0:
            bend.update({'laterality': 'left'})
            if start * 0.2 + end * 0.8 > angle_neutral:
                bend.update({'recoil': False})
            else:
                bend.update({'recoil': True})
        else:
            bend.update({'laterality': 'right'})
            if start * 0.2 + end * 0.8 < angle_neutral:
                bend.update({'recoil': False})
            else:
                bend.update({'recoil': True})
        angles_bends.append(bend)
    
    steps = []
    for datum in accels_data:
        step = step_datum()
        step.accel = datum
        steps.append(step)
    steps_count = len(steps)
    for i in range(steps_count):
        for j in range(len(turns_data)):
            if turns_data[j]['belong'] == None:
                if turns_data[j]['endpos'] >= steps[i].accel['startpos'] and turns_data[j]['startpos'] <= steps[i].accel['endpos']:
                    turns_data[j].update({'belong': steps[i]})
                    steps[i].turns.append(turns_steps[j])
                    steps[i].turns_peaks.append(turns_data[j])
        for j in range(len(angles_data)):
            if angles_data[j]['belong'] == None:
                if angles_data[j]['endpos'] >= steps[i].accel['startpos'] and angles_data[j]['startpos'] <= steps[i].accel['endpos']:
                    angles_data[j].update({'belong': steps[i]})
                    steps[i].bends.append(angles_bends[j])
                    steps[i].bends_peaks.append(angles_data[j])
    for i in range(len(turns_data)):
        if turns_data[i]['belong'] == None and turns_steps[i]['angle'] > settings['min_step_turn']:
            step = step_datum()
            step.turns_peaks = [turns_data[i]]
            step.turns = [turns_steps[i]]
            steps.append(step)
            for j in range(len(angles_data)):
                if angles_data[j]['belong'] == None:
                    if angles_data[j]['endpos'] >= turns_data[i]['startpos'] and angles_data[j]['startpos'] <= turns_data[i]['endpos']:
                        angles_data[j].update({'belong': steps[len(steps) - 1]})
                        steps[len(steps) - 1].bends.append(angles_bends[j])
                        steps[len(steps) - 1].bends_peaks.append(angles_data[j])
    steps_count = len(steps)
    steps.sort(key=lambda a: a.startpos)
    
    for i in range(steps_count):
        if steps[i].accel != None:
            steps[i].startpos = steps[i].accel['startpos']
            steps[i].coastpos = steps[i].accel['endpos']
        else:
            steps[i].startpos = steps[i].turns_peaks[0]['startpos']
            steps[i].coastpos = steps[i].turns_peaks[0]['endpos']
    for i in range(steps_count - 1):
        steps[i].endpos = steps[i + 1].startpos - 1
    steps[steps_count - 1].endpos = l - 1
    
    front = settings['front_window'] * metadata['fps']
    back = settings['back_window'] * metadata['fps']
    for i in range(len(angles_data)):
        if angles_data[i]['belong'] == None and angles_bends[i]['recoil']:
            for j in range(steps_count):
                if angles_data[i]['startpos'] > steps[j].coastpos:
                    if angles_data[i]['startpos'] <= steps[j].coastpos + back:
                        angles_data[i].update({'belong': steps[j]})
                        steps[j].bends_peaks.append(angles_data[i])
                        steps[j].bends.append(angles_bends[i])
                        break
    for i in range(len(angles_data)):
        if angles_data[i]['belong'] == None and not angles_bends[i]['recoil']:
            for j in range(steps_count):
                if angles_data[i]['startpos'] >= steps[j].startpos - front:
                    if angles_data[i]['startpos'] < steps[j].startpos:
                        if len(steps[j].bends) >= 1:
                            if angles_bends[i]['laterality'] == steps[j].bends[0]['laterality']:
                                continue
                        angles_data[i].update({'belong': steps[j]})
                        steps[j].bends_peaks.append(angles_data[i])
                        steps[j].bends.append(angles_bends[i])
                        break
    
    for step in steps:
        
        step.properties.update({
            'current speed': cdist1s[step.startpos],
            'step length': sum(cen_dists[step.startpos:(step.endpos + 1)]),
            'step dur': (step.endpos + 1 - step.startpos) / metadata['fps'],
            'speed change': max(0, speeds.list[step.coastpos] - speeds.list[step.startpos])})
        if step.accel != None:
            step.properties.update({
                'speed change': step.accel['change'],
                'accel': step.accel['meanslope'] * metadata['fps']})
        else:
            step.properties.update({'accel': step.properties['speed change'] * metadata['fps']})
        
        step.turns_count = len(step.turns)
        step.bends_count = len(step.bends)
        
    for step in steps:
        
        if step.turns_count >= 1:
            
            turns_durs = [turn['dur'] for turn in step.turns]
            turns_angles = [turn['angle'] for turn in step.turns]
            turns_angular_velocitys = [turn['angular velocity'] for turn in step.turns]
            turn_angle_overall = directions_free.list[step.turns_peaks[step.turns_count - 1]['endpos']] - directions_free.list[step.turns_peaks[0]['startpos']]
            step.properties.update({'turn angle': abs(turn_angle_overall)})
            
            if abs(turn_angle_overall) >= settings['min_turn_angle']:
                if turn_angle_overall > settings['min_turn_angle']:
                    step.properties.update({'turn laterality': 'right'})
                elif turn_angle_overall < -settings['min_turn_angle']:
                    step.properties.update({'turn laterality': 'left'})
                step.properties.update({
                    'turn dur': turns_durs[turns_angles.index(max(turns_angles))],
                    'turn angular velocity': turns_angular_velocitys[turns_angles.index(max(turns_angles))]})
        
        startpos = step.startpos
        coastpos = step.coastpos
        if step.accel != None and step.turns_count >= 1:
            if abs(turn_angle_overall) >= settings['min_turn_angle']:
                startpos = min(step.accel['startpos'], step.turns_peaks[0]['startpos'])
                coastpos = max(step.accel['endpos'], step.turns_peaks[step.turns_count - 1]['endpos'])
        a = speeds.list[startpos]
        b = speeds.list[coastpos]
        if step.turns_count >= 1:
            angle = abs(turn_angle_overall)
        else:
            angle = 0
        step.properties.update({'velocity change': math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(angle))})
        
        if step.bends_count == 0:
            
            step.properties['mode'] = 'UK'
        
        elif step.bends_count == 1:
            
            step.properties.update({
                'mode': 'HT',
                'bend angle reached': step.bends[0]['angle end'],
                'bend laterality': step.bends[0]['laterality'],
                'bend angle traveled': step.bends[0]['angle change'],
                'bend dur': step.bends[0]['dur'],
                'bend angular velocity': step.bends[0]['angular velocity'],
                'bend pos': step.bends[0]['bend pos']})
        
        elif step.bends_count == 2:
            
            step.properties.update({
                'mode': 'HT',
                'bend angle reached': step.bends[0]['angle end'],
                'bend laterality': step.bends[0]['laterality'],
                'bend angle traveled': step.bends[0]['angle change'] + step.bends[1]['angle change'],
                'bend dur': step.bends[0]['dur'] + step.bends[1]['dur'],
                'bend angular velocity': step.bends[0]['angular velocity'],
                'bend pos': step.bends[0]['bend pos']})
        
        elif step.bends_count >= 3:
            
            angles_reached = [step.bends[i]['angle end'] for i in range(step.bends_count - 1)]
            angles_traveled = [step.bends[i]['angle change'] for i in range(step.bends_count)]
            angular_velocitys = [bend['angular velocity'] for bend in step.bends]
            durs = [bend['dur'] for bend in step.bends]
            bend_pos = [step.bends[i]['bend pos'] for i in range(step.bends_count - 1)]
            
            angles_reached_sum = 0
            angle_max = 0
            angle_max_left = 0
            angle_max_right = 0
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
            
            step.properties.update({
                'mode': 'MT',
                'bend angle reached': angle_max,
                'bend laterality': angle_laterality,
                'bend angle traveled': sum(angles_traveled),
                'bend dur': sum(durs),
                'bend angular velocity': max(angular_velocitys),
                'bend pos': max(bend_pos)})
    
    if settings['plot_figure']:
        fig, ax = plt.subplots()
        speeds_scaled = [speed / 100 for speed in speeds.list]
        ax.plot(speeds_scaled, 'c')
        ax.plot(turns.list, 'y')
        ax.plot(angles.list, 'm')
        i = 0
        colors = ['r', 'g', 'b']
        for step in steps:
            c = colors[i]
            i += 1
            if i == 3:
                i = 0
            if step.accel != None:
                plot_data(ax, speeds_scaled, step.startpos, step.endpos + 1, c)
            for peak in step.turns_peaks:
                plot_data(ax, turns.list, peak['startpos'], peak['endpos'] + 1, c)
            for peak in step.bends_peaks:
                plot_data(ax, angles.list, peak['startpos'], peak['endpos'] + 1, c)
        fig, ax = plt.subplots()
        speeds_scaled = [speed / 100 for speed in speeds.list]
        ax.plot(speeds_scaled, 'c')
        ax.plot(turns.list, 'y')
        ax.plot(angles.list, 'm')
        for step in steps:
            if step.properties['mode'] == 'HT':
                c = 'g'
            elif step.properties['mode'] == 'MT':
                c = 'b'
            else:
                c = 'r'
            if step.accel != None:
                plot_data(ax, speeds_scaled, step.startpos, step.endpos + 1, c)
            for peak in step.turns_peaks:
                plot_data(ax, turns.list, peak['startpos'], peak['endpos'] + 1, c)
            for peak in step.bends_peaks:
                plot_data(ax, angles.list, peak['startpos'], peak['endpos'] + 1, c)
    
    analysis_df = pd.DataFrame()
    methods = ['sum', 'mean', 'std', 'max']
    methods2 = ['mean', 'std', 'max']
    
    turns_df = pd.DataFrame(turns_steps)
    
    angles_df = pd.DataFrame(angles_bends)
    angles_methods = {'angle change': methods,
                      'dur': methods,
                      'angular velocity': methods2}
    angles_dict = {'Type': 'bend',
                   'Classify': None}
    angles_agg = agg(angles_df, angles_dict, angles_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(angles_agg)])
    angles_dict['Classify'] = 'bend left'
    angles_left_agg = agg(angles_df[angles_df['laterality'] == 'left'], angles_dict, angles_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(angles_left_agg)])
    angles_dict['Classify'] = 'bend right'
    angles_right_agg = agg(angles_df[angles_df['laterality'] == 'right'], angles_dict, angles_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(angles_right_agg)])
    
    recoils_df = angles_df[angles_df['recoil'] == True]
    recoils_dict = {'Type': 'recoil',
                    'Classify': None}
    recoils_agg = agg(recoils_df, recoils_dict, angles_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(recoils_agg)])
    recoils_dict['Classify'] = 'bend left'
    recoils_left_agg = agg(recoils_df[recoils_df['laterality'] == 'left'], recoils_dict, angles_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(recoils_left_agg)])
    recoils_dict['Classify'] = 'bend right'
    recoils_right_agg = agg(recoils_df[recoils_df['laterality'] == 'right'], recoils_dict, angles_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(recoils_right_agg)])
    
    steps_df = pd.DataFrame([step.properties for step in steps])
    steps_df = steps_df[steps_df['mode'] != 'UK']
    steps_methods = {
        'step length': methods,
        'speed change': methods2,
        'velocity change': methods2,
        'accel': methods2,
        'step dur': methods,
        'turn angle': methods,
        'turn dur': methods,
        'turn angular velocity': methods2,
        'bend angle reached': methods2,
        'bend angle traveled': methods2,
        'bend angular velocity': methods2,
        'bend dur': methods2,
        'bend pos': methods2}
    steps_dict = {'Type': 'step',
                  'Classify': None}
    steps_agg = agg(steps_df, steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_agg)])
    steps_dict['Classify'] = 'HT'
    steps_ht_agg = agg(steps_df[steps_df['mode'] == 'HT'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_ht_agg)])
    steps_dict['Classify'] = 'MT'
    steps_mt_agg = agg(steps_df[steps_df['mode'] == 'MT'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_mt_agg)])
    steps_dict['Classify'] = 'turn left'
    steps_turn_left_agg = agg(steps_df[steps_df['turn laterality'] == 'left'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_turn_left_agg)])
    steps_dict['Classify'] = 'turn right'
    steps_turn_right_agg = agg(steps_df[steps_df['turn laterality'] == 'right'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_turn_right_agg)])
    steps_dict['Classify'] = 'with turn'
    steps_turn_agg = agg(steps_df[steps_df['turn laterality'] != 'neutral'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_turn_agg)])
    steps_dict['Classify'] = 'without turn'
    steps_turn_no_agg = agg(steps_df[steps_df['turn laterality'] == 'neutral'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_turn_no_agg)])
    steps_dict['Classify'] = 'bend left'
    steps_bend_left_agg = agg(steps_df[steps_df['bend laterality'] == 'left'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_bend_left_agg)])
    steps_dict['Classify'] = 'bend right'
    steps_bend_right_agg = agg(steps_df[steps_df['bend laterality'] == 'right'], steps_dict, steps_methods)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(steps_bend_right_agg)])
    
    steps_dict['Classify'] = 'None'
    stratification = stratify(steps_df, steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification)])
    steps_dict['Classify'] = 'turn left'
    stratification_turn_left = stratify(steps_df[steps_df['turn laterality'] == 'left'], steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification_turn_left)])
    steps_dict['Classify'] = 'turn right'
    stratification_turn_right = stratify(steps_df[steps_df['turn laterality'] == 'right'], steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification_turn_right)])
    steps_dict['Classify'] = 'with turn'
    stratification_turn = stratify(steps_df[steps_df['turn laterality'] != 'neutral'], steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification_turn)])
    steps_dict['Classify'] = 'without turn'
    stratification_turn_no = stratify(steps_df[steps_df['turn laterality'] == 'neutral'], steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification_turn_no)])
    steps_dict['Classify'] = 'bend left'
    stratification_bend_left = stratify(steps_df[steps_df['bend laterality'] == 'left'], steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification_bend_left)])
    steps_dict['Classify'] = 'bend right'
    stratification_bend_right = stratify(steps_df[steps_df['bend laterality'] == 'right'], steps_dict)
    analysis_df = pd.concat([analysis_df, pd.DataFrame(stratification_bend_right)])
    
    turns_df.to_csv(path + '/' + videoname + '_turns_df.csv', index=False)
    angles_df.to_csv(path + '/' + videoname + '_angles_df.csv', index=False)
    steps_df.to_csv(path + '/' + videoname + '_steps_df.csv', index=False)
    
    steps_df['video'] = videoname
    steps_all = pd.concat([steps_all, steps_df])
    
    rp = []
    for i in steps_df.columns:
        for j in steps_df.columns:
            if i == j:
                continue
            if steps_df[i].dtypes != 'O' and steps_df[j].dtypes != 'O':
                result = scipy.stats.linregress(steps_df[i], steps_df[j])
                ci = scipy.stats.pearsonr(steps_df[i], steps_df[j]).confidence_interval()
                rp.append({
                    'x': i,
                    'y': j,
                    'slope': result.slope,
                    'intercept': result.intercept,
                    'rvalue': result.rvalue,
                    'low r': ci[0],
                    'hi r': ci[1],
                    'pvalue': result.pvalue,
                    'stderr': result.stderr})
    rp_df = pd.DataFrame(rp)
    rp_df['video'] = videoname
    rps_df = pd.concat([rps_df, rp_df])
    
    with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
        for key in analysis:
            f.write(key + ', ' + str(analysis[key]) + '\n')
    print('Analysis of ' + videoname + ' complete.')
    analyses.append(analysis)
    
    analysis_df.to_csv(path + '/' + videoname + '_analysis_df.csv')
    if first_video:
        analyses_df = analysis_df[['Type', 'Classify', 'Stratify', 'Parameter', 'Method']]
        first_video = False
    analyses_df[videoname] = analysis_df['Value']
    
    with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
        for key in settings:
            f.write(key + ', ' + str(settings[key]) + '\n')

steps_all.to_csv('steps_all.csv', index=False)

analyses = pd.DataFrame(analyses).T
analyses = analyses.drop('videoname')
analyses.columns = videonames
analyses.to_csv('analyses.csv')

analyses_df.to_csv('analyses_df.csv', index=False)

analyses_df_adjusted = pd.DataFrame(analyses_df)
adjusts = ['current speed', 'step length', 'speed change', 'velocity change', 'accel']
for videoname in videonames:
    fish_length = analyses.at['fish_length', videoname]
    analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), videoname] = analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), videoname].transform(lambda a: a / fish_length)
analyses_df_adjusted.to_csv('analyses_df_adjusted.csv', index=False)
print('All analyses complete.')

rps_df = rps_df.sort_values(by=['x', 'y'])
rps_df.to_csv('rps.csv', index=False)

with pd.ExcelWriter('properties.xlsx') as writer:
    for i in steps_all.columns:
        if i == 'video':
            continue
        p = pd.DataFrame()
        for videoname in videonames:
            p[videoname] = steps_all[steps_all['video'] == videoname][i]
        p.to_excel(writer, sheet_name=i, index=False)
