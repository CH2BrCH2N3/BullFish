import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats
import numpy as np
from scipy.signal import find_peaks
from copy import copy, deepcopy
from BullFish_pkg.math import pyth, cal_direction, cal_direction_change
from BullFish_pkg.general import create_path, csvtodict, load_settings

default_settings = {
    "tank_x": 210,
    "tank_y": 144,
    "plot_figure": 0,
    "sampling": 2,
    "speed_limit": 2000,
    "accel_cutoff": 100,
    "min_accel_dur": 0.02,
    "min_max_accel": 100,
    "min_speed_change": 0,
    "spine_analysis": 1,
    'alternate_turn': 0,
    "turn_cutoff": 2,
    "min_turn_dur": 0.02,
    "min_max_turn_velocity": 2,
    "min_turn_angle": 0.1,
    "large_turn": 0.349,
    "bend_cutoff": 2,
    "min_bend_dur": 0.02,
    "min_bend_speed": 2,
    "min_bend_angle": 0.05,
    "min_amp": 2,
    'use_s1': 1}

settings = load_settings('analysis', default_settings)
sampling = settings['sampling']

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

class dfln: # dfln means deflection
    def __init__(self):
        self.startpos = 0
        self.startval = 0
        self.endpos = 0
        self.endval = 0
        self.centralpos = 0
        self.length = 0
        self.dur = 0
        self.change = 0
        self.meanslope = 0
        self.maxslope = 0
        self.maxslopepos = 0
        self.belong = -1
        self.dict = {}
        self.env = {}

def remove_duplicates(lst):
    dct = {}
    for item in lst:
        dct.update({str(item.startpos): item})
    nlst = []
    for key in dct.keys():
        nlst.append(dct[key])
    return nlst

def plot_data(ax, lst, startpos, endpos, c='c'):
    x = [i for i in range(startpos, endpos)]
    y = [lst[i] for i in range(startpos, endpos)]
    ax.plot(x, y, c)

class curve:
    
    def __init__(self, inputlist, start, end, window=0):
        
        self.original = copy(inputlist)
        self.l = len(inputlist)
        self.start = start
        self.window = window
        
        self.list = copy(inputlist)
        if window > 2:
            for i in range(start, start + window // 2):
                self.list[i] = sum(self.original[start:(i + window // 2 + 1)]) / (i + window // 2 - start + 1)
            for i in range(start + window // 2 + 1, end - window // 2):
                self.list[i] = sum(self.original[(i - window // 2):(i + window // 2 + 1)]) / window
            for i in range(end - window // 2, end):
                self.list[i] = sum(self.original[(i - window // 2):end]) / (end - i + window // 2)
        self.p = [(item if item > 0 else 0) for item in self.list]
        self.n = [(-item if item < 0 else 0) for item in self.list]
        
        self.p_dflns = []
        self.n_dflns = []
        
    def dt(self, fps):
        
        self.fps = fps
        self.dt_original = [0 for i in range(self.l)]
        for i in range(self.start + 1, self.l):
            self.dt_original[i] = (self.list[i] - self.list[i - 1]) * fps
        self.dt_curve = curve(self.dt_original, self.start + 1, self.l, self.window)
        self.pdt = self.dt_curve.p
        self.ndt = self.dt_curve.n
        
    def get_p_dflns(self, prominence, front, back, dur=0, maxslope=0, change=0):
        
        self.p_peaks, _ = find_peaks(self.pdt, prominence=prominence)
        self.p_peaks = [int(peak) for peak in self.p_peaks]
        
        for peak in self.p_peaks:
            d = dfln()
            j = peak
            while j > 0:
                slope = (self.list[j] - self.list[j - 1]) * self.fps
                if slope < front:
                    break
                j -= 1
            d.startpos = j
            d.startval = self.list[j]
            j = peak
            while j < self.l - 1:
                slope = (self.list[j + 1] - self.list[j]) * self.fps
                if slope < back:
                    break
                j += 1
            d.endpos = j
            d.endval = self.list[j]
            d.centralpos = (d.startpos + d.endpos) / 2
            d.length = d.endpos - d.startpos
            d.dur = d.length / self.fps
            d.change = d.endval - d.startval
            d.meanslope = d.change / d.dur if d.length > 0 else 0
            d.maxslope = 0
            d.maxslopepos = 0
            for j in range(d.startpos + 1, d.endpos + 1):
                slope = abs(self.list[j] - self.list[j - 1]) * self.fps
                if d.maxslope < slope:
                    d.maxslope = slope
                    d.maxslopepos = j
            if d.startpos > 0 and d.dur > dur and d.maxslope > maxslope and d.change > change:
                self.p_dflns.append(d)
        
        self.p_dflns = remove_duplicates(self.p_dflns)
        self.p_dflns_count = len(self.p_dflns)
    
    def get_n_dflns(self, prominence, front, back, dur=0, maxslope=0, change=0):
        
        self.n_peaks, _ = find_peaks(self.ndt, prominence=prominence)
        self.n_peaks = [int(peak) for peak in self.n_peaks]
        
        for peak in self.n_peaks:
            d = dfln()
            j = peak
            while j > 0:
                slope = (self.list[j] - self.list[j - 1]) * self.fps
                if slope > front:
                    break
                j -= 1
            d.startpos = j
            d.startval = self.list[j]
            j = peak
            while j < self.l - 1:
                slope = (self.list[j + 1] - self.list[j]) * self.fps
                if slope > back:
                    break
                j += 1
            d.endpos = j
            d.endval = self.list[j]
            d.centralpos = (d.startpos + d.endpos) / 2
            d.length = d.endpos - d.startpos
            d.dur = d.length / self.fps
            d.change = d.endval - d.startval
            d.meanslope = d.change / d.dur if d.length > 0 else 0
            d.maxslope = 0
            d.maxslopepos = 0
            for j in range(d.startpos + 1, d.endpos + 1):
                slope = abs(self.list[j] - self.list[j - 1]) * self.fps
                if d.maxslope < slope:
                    d.maxslope = slope
                    d.maxslopepos = j
            if d.startpos > 0 and d.dur > dur and abs(d.maxslope) > maxslope and abs(d.change) > change:
                self.n_dflns.append(d)
        
        self.n_dflns = remove_duplicates(self.n_dflns)
        self.n_dflns_count = len(self.n_dflns)
    
    def merge_dflns(self):
        
        self.dflns = deepcopy(self.p_dflns)
        self.dflns.extend(deepcopy(self.n_dflns))
        self.dflns.sort(key=lambda a: a.startpos)
        self.dflns_count = len(self.dflns)
    
    def graph_dflns(self):
        
        fig, ax = plt.subplots()
        ax.plot(self.list, 'y')
        for d in self.p_dflns:
            plot_data(ax, self.list, d.startpos, d.endpos + 1, 'b')
        for d in self.n_dflns:
            plot_data(ax, self.list, d.startpos, d.endpos + 1, 'r')
    
    def get_freq_envs(self, other_dflns, name):
         # the env 'speed' is not included in this function
        for i in range(self.dflns_count):
            start = max(0, round(self.dflns[i].centralpos - self.fps / 2))
            end = min(round(self.dflns[i].centralpos + self.fps / 2), l - 1)
            r = end - start
            count = 0
            for d in other_dflns.dflns:
                if d.centralpos >= start and d.centralpos <= end:
                    count += 1
            self.dflns[i].env.update({
                name + ' frequency': count / r})
   
class step:
    def __init__(self):
        self.startpos = 0
        self.coastpos = 0
        self.centralpos = 0
        self.endpos = 0
        self.accel = None
        self.turns = []
        self.bends = []
        self.properties = {
            'current_speed': 0,
            'step_length': 0,
            'speed_change': 0,
            'velocity_change': 0,
            'accel': 0,
            'step_dur': 0,
            'coast_dur': 0,
            'coast_percent': 100,
            'current_step_s': 0,
            'turn_angle': 0,
            'turn_laterality': 'neutral',
            'turn_dur': 0,
            'turn_angular_velocity': 0,
            'current_bend_s': 0,
            'bend_angle_reached': 0,
            'bend_laterality': 'neutral',
            'bend_pos': 0,
            'bend_angle_traveled': 0,
            'bend_angular_velocity': 0,
            'bend_dur_total': 0,
            'bend_wave_freq': 0,
            'bend_count': 0,
            'max angle pos': 0}

aggs = {
    'sum': np.sum,
    'mean': np.mean,
    'std': np.std,
    'median': np.median,
    'p5': lambda a: np.percentile(a, 5),
    'p95': lambda a: np.percentile(a, 95),
    'ipr': lambda a: np.percentile(a, 95) - np.percentile(a, 5)}
agg1 = ['sum', 'mean', 'std', 'median', 'p5', 'p95', 'ipr']
agg2 = ['mean', 'std', 'median', 'p5', 'p95', 'ipr']
agg3 = ['median', 'p5', 'p95', 'ipr']

def result_dict(Type, Classify, Stratify, Parameter, Method, Value):
    return {'Type': Type,
            'Classify': Classify,
            'Stratify': Stratify,
            'Parameter': Parameter,
            'Method': Method,
            'Value': Value}

class DF:
    
    def __init__(self, df, Type, params):
        self.df = df
        self.Type = Type
        self.params = params # the dictionary of parameters that are dependent variables
        self.l = len(df)
        self.dfs = {}
        self.intervals = self.df.describe()
    
    def agg(self, methods, dfname=None):
        analysis_list = []
        if dfname == None:
            df = self.df
        else:
            df = self.dfs[dfname]
        for i in methods.keys():
            for method in methods[i]:
                analysis_list.append(result_dict(self.Type, dfname, None, i, method, aggs[method](df[i])))
        analysis_list.append(result_dict(self.Type, dfname, None, None, 'count', len(df)))
        return pd.DataFrame(analysis_list)
    
    def stratify0(self, param, cutoff): # param is the independent variable
        analysis_list = []
        df = self.df
        df0 = df[df[param] < cutoff]
        df1 = df[df[param] >= cutoff]
        for i in self.params:
            if i == param:
                continue
            analysis_list.append(result_dict(self.Type, None, param + '_low', i, 'mean', df0[i].mean()))
            analysis_list.append(result_dict(self.Type, None, param + '_high', i, 'mean', df1[i].mean()))
        return pd.DataFrame(analysis_list)
    
    def stratify1(self, param): # param is the independent variable
        analysis_list = []
        df = self.df.copy()
        df.sort_values(by=[param])
        l = len(df)
        df1 = df.iloc[:round(l / 4)]
        df2 = df.iloc[round(l / 4):round(l * 3 / 4)]
        df3 = df.iloc[round(l * 3 / 4):l]
        for i in self.params:
            if i == param:
                continue
            analysis_list.append(result_dict(self.Type, None, param + '_low', i, 'mean', df1[i].mean()))
            analysis_list.append(result_dict(self.Type, None, param + '_mid', i, 'mean', df2[i].mean()))
            analysis_list.append(result_dict(self.Type, None, param + '_high', i, 'mean', df3[i].mean()))
        return pd.DataFrame(analysis_list)
    
    def stratify2(self, dfnamea, dfnameb, param):
        analysis_list = []
        dfa = self.dfs[dfnamea].copy()
        dfa.sort_values(by=[param])
        la = len(dfa)
        dfa1 = dfa.iloc[:round(la / 4)]
        dfa2 = dfa.iloc[round(la / 4):round(la * 3 / 4)]
        dfa3 = dfa.iloc[round(la * 3 / 4):la]
        dfb = self.dfs[dfnameb].copy()
        dfb.sort_values(by=[param])
        lb = len(dfb)
        dfb1 = dfb.iloc[:round(lb / 4)]
        dfb2 = dfb.iloc[round(lb / 4):round(lb * 3 / 4)]
        dfb3 = dfb.iloc[round(lb * 3 / 4):lb]
        for i in self.params:
            if i == param:
                continue
            analysis_list.append(result_dict(self.Type, dfnamea, param + '_low', i, 'mean', dfa1[i].mean()))
            analysis_list.append(result_dict(self.Type, dfnameb, param + '_low', i, 'mean', dfb1[i].mean()))
            analysis_list.append(result_dict(self.Type, dfnamea, param + '_mid', i, 'mean', dfa2[i].mean()))
            analysis_list.append(result_dict(self.Type, dfnameb, param + '_mid', i, 'mean', dfb2[i].mean()))
            analysis_list.append(result_dict(self.Type, dfnamea, param + '_high', i, 'mean', dfa3[i].mean()))
            analysis_list.append(result_dict(self.Type, dfnameb, param + '_high', i, 'mean', dfb3[i].mean()))
        return pd.DataFrame(analysis_list)

class results_df:
    def __init__(self, df):
        self.df = df
    def add(self, results):
        self.df = pd.concat([self.df, results])

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
    fps = metadata['fps']
    
    # load essential tracking data
    with open(path + '/' + videoname + '_cen.csv', 'r') as f:
        cen = [[cell for cell in row] for row in csv.reader(f)]
        cen.pop(0)
        for i in range(l):
            cen[i] = (float(cen[i][0]) * ratio, float(cen[i][1]) * ratio)
    
    # obtain a list of speed at each frame
    cen_dists = [0 for i in range(l)]
    speeds = [0 for i in range(l)]
    for i in range(sampling, l, sampling):
        cen_dists[i] = pyth(cen[i], cen[i - sampling]) / sampling
        speeds[i] = cen_dists[i] * fps
    for i in range(0, sampling):
        cen_dists[i] = cen_dists[sampling]
    for i in range(sampling * 2, l, sampling):
        for j in range(i - sampling + 1, i):
            cen_dists[j] = (cen_dists[i - sampling] * (i - j) + cen_dists[i] * (j - (i - sampling))) / sampling
            speeds[j] = cen_dists[j] * fps
    
    error1_frames = []
    for i in range(l):
        if speeds[i] > settings['speed_limit']:
            error1_frames.append(i)
    cen_dists = errors_correct(cen_dists, error1_frames)
    speeds = errors_correct(speeds, error1_frames)
    
    # obtain total distance travelled and average speed
    total_distance = sum(cen_dists)
    total_time = l / fps
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
    total_freeze_time = sum(freeze) / fps
    freeze_percent = total_freeze_time / total_time * 100
    active_time = total_time - total_freeze_time
    active_speed = total_distance / active_time
    freeze_count = 0
    for i in range(2, l):
        if freeze[i] - freeze[i - 1] == 1:
            freeze_count += 1
    freeze_freq = per_min(freeze_count)
    
    # obtain a list of speeds with running average and obtain max speed
    speeds = curve(speeds, start=1, end=l, window=5)
    max_speed = max(speeds.list)
    
    # obtain a list of speed measured over 1s
    cdist1s = [0 for i in range(l)]
    for i in range(1, l):
        start = max(1, round(i + 1 - fps / 2))
        end = min(l, round(i + 1 + fps / 2))
        cdist1s[i] = sum([cen_dists[j] for j in range(start, end)]) * fps / (end - start)
    cdist1s[0] = cdist1s[1]
    max_distance_1s = max(cdist1s)
    
    ds_from_wall = [0 for i in range(l)]
    thigmotaxis_time = 0
    xl = metadata['swimarea_tlx'] * ratio
    length = metadata['swimarea_x'] * ratio
    xr = xl + length
    yt = metadata['swimarea_tly'] * ratio
    width = metadata['swimarea_y'] * ratio
    yb = yt + width
    d_from_wall = min(length, width) / 4
    for i in range(l):
        x = min(cen[i][0] - xl, xr - cen[i][0])
        y = min(cen[i][1] - yt, yb - cen[i][1])
        ds_from_wall[i] = min(x, y)
        if ds_from_wall[i] < d_from_wall:
            thigmotaxis_time += 1
    thigmotaxis_time /= fps
    
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
        'freeze_freq': freeze_freq,
        'thigmotaxis_time': thigmotaxis_time}
    
    if settings['plot_figure']:
        create_path('Tracks')
        fig, ax = plt.subplots()
        ax.scatter(x=[cen[i][0] for i in range(l)], y=[cen[i][1] for i in range(l)])
        fig.savefig('Tracks/' + videoname + '_track.png')
    
    speeds.dt(fps)
    speeds.get_p_dflns(settings['accel_cutoff'], -10, -20,
                       settings['min_accel_dur'], settings['min_max_accel'], settings['min_speed_change'])
    
    if speeds.p_dflns_count == 0:
        print('No detectable movement')
        if settings['save_individually']:
            with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
                for key in analysis:
                    f.write(key + ',' + str(analysis[key]) + '\n')
        analyses.append(analysis)
        print('Analysis of ' + videoname + ' complete.')
        with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
            for key in settings:
                f.write(key + ', ' + str(settings[key]) + '\n')
        continue
    
    for i in range(speeds.p_dflns_count):
        speeds.p_dflns[i].dict.update({
            'acceleration': speeds.p_dflns[i].maxslope,
            'speed_change': speeds.p_dflns[i].change})
        speeds.p_dflns[i].env.update({
            'speed': cdist1s[round(speeds.p_dflns[i].centralpos)]})
    
    if settings['plot_figure']:
        speeds.graph_dflns()
    
    if not settings['spine_analysis']:
        analyses.append(analysis)
        print('Analysis of ' + videoname + ' complete.')
        with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
            for key in settings:
                f.write(key + ', ' + str(settings[key]) + '\n')
        continue # step analysis when spine_analysis is disabled is not yet available
    
    # load midline points data
    spine_lens = [0 for i in range(l)]
    with open(path + '/' + videoname + '_spine.csv', 'r') as f:
        spines = [[cell for cell in row] for row in csv.reader(f)]
        spines.pop(0)
        for i in range(l):
            spine_lens[i] = int(spines[i][0])
            spine_temp = []
            for j in range(1, spine_lens[i] + 1):
                spine_temp.append((float(spines[i][j * 2 - 1]) * ratio, float(spines[i][j * 2]) * ratio))
            spines[i] = spine_temp
    if not settings['use_s1']:
        with open(path + '/' + videoname + '_s0s.csv', 'r') as f:
            s0_temp = [[cell for cell in row] for row in csv.reader(f)]
            s0_temp.pop(0)
            for i in range(l):
                spines[i].insert(0, (float(s0_temp[i][0]) * ratio, float(s0_temp[i][1]) * ratio))
                spine_lens[i] += 1
    directions = [0 for i in range(l)]
    for i in range(l):
        if settings['alternate_turn']:
            directions[i] = cal_direction(spines[i][round(spine_lens[i] * 2 / 3)], spines[i][spine_lens[i] - 1])
        else:
            directions[i] = cal_direction(spines[i][spine_lens[i] - 2], spines[i][spine_lens[i] - 1])
    turns = [0 for i in range(l)]
    for i in range(1, l):
        turns[i] = cal_direction_change(directions[i - 1], directions[i])
    heads = [0 for i in range(l)]
    with open(path + '/' + videoname + '_sn+1s.csv', 'r') as f:
        temp = [[cell for cell in row] for row in csv.reader(f)]
        temp.pop(0)
        for i in range(l):
            heads[i] = (float(temp[i][0]) * ratio, float(temp[i][1]) * ratio)
    
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
                trunk_amps[i][j] = abs(spines[i][j][0] - spines[i][spine_lens[i] - 1][0])
        else:
            m = (spines[i][spine_lens[i] - 1][1] - spines[i][spine_lens[i] - 2][1]) / (spines[i][spine_lens[i] - 1][0] - spines[i][spine_lens[i] - 2][0])
            c = spines[i][spine_lens[i] - 1][1] - m * spines[i][spine_lens[i] - 1][0]
            for j in range(spine_lens[i] - 2):
                trunk_amps[i][j] = abs(m * spines[i][j][0] - spines[i][j][1] + c) / math.sqrt(m ** 2 + 1)
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
    bend_poss = curve(bend_poss, 0, l, 3)
    
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
    
    fdirs = [0 for i in range(l)] # fdirs is a list of special running average of direction of locomotion
    fdirs[0] = directions[0]
    for i in range(1, l):
        fdirs[i] = fdirs[i - 1] + turns[i]
    
    fdirs = curve(fdirs, start=0, end=l, window=3)
    fdirs.dt(fps) # turning left is -, turning right is +
    fdirs.get_p_dflns(settings['turn_cutoff'], 2, 2,
                      settings['min_turn_dur'], settings['min_max_turn_velocity'], settings['min_turn_angle'])
    fdirs.get_n_dflns(settings['turn_cutoff'], -2, -2,
                      settings['min_turn_dur'], settings['min_max_turn_velocity'], settings['min_turn_angle'])
    fdirs.merge_dflns()
    if settings['plot_figure']:
        fdirs.graph_dflns()
    
    for i in range(fdirs.dflns_count):
        fdirs.dflns[i].dict.update({
            'turn_angle': abs(fdirs.dflns[i].change),
            'turn_angular_velocity': abs(fdirs.dflns[i].maxslope),
            'turn_duration': fdirs.dflns[i].dur,
            'turn_laterality': 'left' if fdirs.dflns[i].change < 0 else 'right'})
        fdirs.dflns[i].env.update({'speed': cdist1s[round(fdirs.dflns[i].centralpos)]})
    
    angles = curve(angles, start=0, end=l, window=5)
    angles.dt(fps) # turning left is +, turning right is -
    angles.get_p_dflns(settings['bend_cutoff'], 2, 2,
                       settings['min_bend_dur'], settings['min_bend_speed'], settings['min_bend_angle'])
    angles.get_n_dflns(settings['bend_cutoff'], -2, -2,
                       settings['min_bend_dur'], settings['min_bend_speed'], settings['min_bend_angle'])
    angles.merge_dflns()
    if settings['plot_figure']:
        angles.graph_dflns()
    
    angles_neutral = copy(angles.list)
    angles_neutral_count = l
    for d in angles.p_dflns:
        for i in range(d.startpos, d.endpos + 1):
            angles_neutral[i] = 0
            angles_neutral_count -= 1
    for d in angles.n_dflns:
        for i in range(d.startpos, d.endpos + 1):
            angles_neutral[i] = 0
            angles_neutral_count -= 1
    angle_neutral = sum(angles_neutral) / angles_neutral_count
    analysis.update({'angle_neutral': angle_neutral})
    
    for i in range(angles.dflns_count):
        angles.dflns[i].dict.update({
            'angle_change': abs(angles.dflns[i].change),
            'angle start': angles.dflns[i].startval,
            'angle end': angles.dflns[i].endval,
            'bend_dur': angles.dflns[i].dur,
            'bend_angular_velocity': angles.dflns[i].maxslope,
            'bend_pos': max(bend_poss.list[angles.dflns[i].startpos:(angles.dflns[i].endpos + 1)])})
        if angles.dflns[i].change > 0:
            angles.dflns[i].dict.update({'bend_laterality': 'left'})
            if angles.dflns[i].startval * 0.2 + angles.dflns[i].endval * 0.8 > angle_neutral:
                angles.dflns[i].dict.update({'recoil': False})
            else:
                angles.dflns[i].dict.update({'recoil': True})
        else:
            angles.dflns[i].dict.update({'bend_laterality': 'right'})
            if angles.dflns[i].startval * 0.2 + angles.dflns[i].endval * 0.8 < angle_neutral:
                angles.dflns[i].dict.update({'recoil': False})
            else:
                angles.dflns[i].dict.update({'recoil': True})
        angles.dflns[i].env.update({'speed': cdist1s[round(angles.dflns[i].centralpos)]})
    
    analysis_df = results_df(pd.DataFrame())
    
    speeds_df = [d.dict for d in speeds.p_dflns]
    for i in range(speeds.p_dflns_count):
        speeds_df[i].update(speeds.p_dflns[i].env)
    speeds_df = pd.DataFrame(speeds_df)
    speeds_methods = {'acceleration': agg2,
                      'speed_change': agg1}
    speeds_DF = DF(speeds_df, 'acceleration', speeds_methods.keys())
    analysis_df.add(speeds_DF.agg(speeds_methods))
    analysis_df.add(speeds_DF.stratify1('speed'))
    
    fdirs_df = [d.dict for d in fdirs.dflns]
    for i in range(fdirs.dflns_count):
        fdirs_df[i].update(fdirs.dflns[i].env)
    fdirs_df = pd.DataFrame(fdirs_df)
    fdirs_methods = {'turn_angle': agg1,
                     'turn_angular_velocity': agg2,
                     'turn_duration': agg1}
    fdirs_DF = DF(fdirs_df, 'turn', fdirs_methods.keys())
    fdirs_DF.dfs.update({
        'turn left': fdirs_df[fdirs_df['turn_laterality'] == 'left'],
        'turn right': fdirs_df[fdirs_df['turn_laterality'] == 'right']})
    analysis_df.add(fdirs_DF.agg(fdirs_methods))
    analysis_df.add(fdirs_DF.agg(fdirs_methods, 'turn left'))
    analysis_df.add(fdirs_DF.agg(fdirs_methods, 'turn right'))
    analysis_df.add(fdirs_DF.stratify1('speed'))
    
    total_turn_angle = analysis_df.df.loc[(analysis_df.df['Type'] == 'turn') &
                                          analysis_df.df['Classify'].isna() &
                                          (analysis_df.df['Parameter'] == 'turn_angle') &
                                          (analysis_df.df['Method'] == 'sum'), 'Value']
    analysis.update({'meandering': total_turn_angle.iloc[0] / total_distance})
    
    angles_df = [d.dict for d in angles.dflns]
    for i in range(angles.dflns_count):
        angles_df[i].update(angles.dflns[i].env)
    angles_df = pd.DataFrame(angles_df)
    angles_methods = {'angle_change': agg1,
                      'bend_dur': agg1,
                      'bend_pos': agg2,
                      'bend_angular_velocity': agg2}
    angles_DF = DF(angles_df, 'bend', angles_methods.keys())
    angles_DF.dfs.update({
        'bend left': angles_df[angles_df['bend_laterality'] == 'left'],
        'bend right': angles_df[angles_df['bend_laterality'] == 'right']})
    analysis_df.add(angles_DF.agg(angles_methods))
    analysis_df.add(angles_DF.agg(angles_methods, 'bend left'))
    analysis_df.add(angles_DF.agg(angles_methods, 'bend right'))
    analysis_df.add(angles_DF.stratify1('speed'))
    
    recoils_df = angles_df[angles_df['recoil'] == True]
    recoils_DF = DF(recoils_df, 'recoil', angles_methods.keys())
    recoils_DF.dfs.update({
        'bend left': recoils_df[recoils_df['bend_laterality'] == 'left'],
        'bend right': recoils_df[recoils_df['bend_laterality'] == 'right']})
    analysis_df.add(recoils_DF.agg(angles_methods))
    analysis_df.add(recoils_DF.agg(angles_methods, 'bend left'))
    analysis_df.add(recoils_DF.agg(angles_methods, 'bend right'))
    analysis_df.add(recoils_DF.stratify1('speed'))
    
    steps = []
    
    for i in range(speeds.p_dflns_count):
        
        candidates = []
        scores = []
        for j in range(angles.dflns_count):
            if angles.dflns[j].belong == -1 and not angles.dflns[j].dict['recoil']:
                if angles.dflns[j].endpos >= speeds.p_dflns[i].startpos and angles.dflns[j].startpos <= speeds.p_dflns[i].endpos:
                    candidates.append(j)
                    scores.append(abs(angles.dflns[j].centralpos - speeds.p_dflns[i].centralpos))
        
        if len(candidates) == 0:
            continue
        
        choice = candidates[scores.index(min(scores))]
        s = step()
        s.accel = speeds.p_dflns[i]
        s.bends.append(angles.dflns[choice])
        angles.dflns[choice].belong = s.accel.centralpos
        
        for j in range(fdirs.dflns_count):
            if fdirs.dflns[j].belong == -1:
                if fdirs.dflns[j].endpos >= s.accel.startpos and fdirs.dflns[j].startpos <= s.accel.endpos:
                    fdirs.dflns[j].belong = s.accel.centralpos
                    s.turns.append(fdirs.dflns[j])
        
        steps.append(s)
    
    for i in range(fdirs.dflns_count):
        
        if fdirs.dflns[i].belong != -1:
            continue
        
        candidates = []
        scores = []
        for j in range(angles.dflns_count):
            if angles.dflns[j].belong == -1:
                if angles.dflns[j].endpos >= fdirs.dflns[i].startpos and angles.dflns[j].startpos <= fdirs.dflns[i].endpos:
                    candidates.append(j)
                    scores.append(abs(angles.dflns[j].centralpos - fdirs.dflns[i].centralpos))
        
        if len(candidates) == 0:
            continue
        
        choice = candidates[scores.index(min(scores))]
        s = step()
        s.turns.append(fdirs.dflns[i])
        fdirs.dflns[i].belong = fdirs.dflns[i].centralpos
        s.bends.append(angles.dflns[choice])
        angles.dflns[choice].belong = s.turns[0].centralpos
        
        steps.append(s)
    
    steps_count = len(steps)
    for i in range(steps_count):
        if steps[i].accel != None:
            steps[i].startpos = steps[i].accel.startpos
            steps[i].coastpos = steps[i].accel.endpos
        else:
            steps[i].startpos = steps[i].turns[0].startpos
            steps[i].coastpos = steps[i].turns[0].endpos
        steps[i].centralpos = (steps[i].startpos + steps[i].coastpos) / 2
    steps.sort(key=lambda a: a.startpos)
    for i in range(steps_count - 1):
        steps[i].endpos = steps[i + 1].startpos - 1
    steps[steps_count - 1].endpos = l - 1
    
    for i in range(angles.dflns_count):
        if angles.dflns[i].belong == -1 and not angles.dflns[i].dict['recoil']:
            candidates = []
            scores = []
            for j in range(steps_count):
                if angles.dflns[i].endpos >= steps[j].startpos and angles.dflns[i].startpos <= steps[j].coastpos:
                    candidates.append(j)
                    scores.append(abs(angles.dflns[i].centralpos - steps[j].centralpos))
            if len(candidates) == 0:
                continue
            choice = candidates[scores.index(min(scores))]
            steps[choice].bends.append(angles.dflns[i])
            angles.dflns[i].belong = steps[choice].centralpos
    
    for i in range(1, angles.dflns_count):
        if angles.dflns[i].belong == -1 and angles.dflns[i].dict['recoil']:
            if angles.dflns[i - 1].belong != -1:
                if angles.dflns[i - 1].endpos >= angles.dflns[i].startpos - 2:
                    choice = -1
                    for j in range(steps_count):
                        if steps[j].centralpos == angles.dflns[i - 1].belong:
                            choice = j
                            break
                    if choice == -1:
                        continue
                    steps[choice].bends.append(angles.dflns[i])
                    angles.dflns[i].belong = steps[choice].centralpos
    
    if settings['plot_figure']:
        fig, ax = plt.subplots() #step pairing graph
        speeds_scaled = [speed / fps for speed in speeds.list]
        turns_scaled = [turn / fps for turn in fdirs.dt_curve.list]
        ax.plot(speeds_scaled, 'c')
        ax.plot(turns_scaled, 'y')
        ax.plot(angles.list, 'm')
        i = 0
        colors = ['r', 'g', 'b']
        for s in steps:
            c = colors[i]
            i += 1
            if i == 3:
                i = 0
            if s.accel != None:
                plot_data(ax, speeds_scaled, s.startpos, s.coastpos + 1, c)
            for turn in s.turns:
                plot_data(ax, turns_scaled, turn.startpos, turn.endpos + 1, c)
            for bend in s.bends:
                plot_data(ax, angles.list, bend.startpos, bend.endpos + 1, c)
    
    for s in steps:
        
        s.properties.update({
            'current_speed': cdist1s[round(s.centralpos)],
            'step_length': sum(cen_dists[(s.startpos + 1):(s.endpos + 1)]),
            'step_dur': (s.endpos + 1 - s.startpos) / fps})
        if s.accel != None:
            s.properties.update({
                'speed_change': s.accel.dict['speed_change'],
                'accel': s.accel.dict['acceleration']})
        else:
            s.properties.update({
                'speed_change': 0,
                'accel': 0})
        
        s.properties['coast_dur'] = max(0, s.endpos + 1 - s.coastpos) / fps
        s.properties['coast_percent'] = s.properties['coast_dur'] / s.properties['step_dur'] * 100
        
        s.turns_count = len(s.turns)
        s.bends_count = len(s.bends)
        s.properties['bend_count'] = s.bends_count
        
    for s in steps:
        
        startpos = round(max(0, s.startpos - fps / 2))
        endpos = round(min(l - 1, s.startpos + fps / 2))
        current_bend_count = 0
        for i in range(angles.dflns_count):
            bend_startpos = angles.dflns[i].startpos
            if bend_startpos >= startpos and bend_startpos <= endpos:
                current_bend_count += 1
        s.properties['current_bend_s'] = current_bend_count / (endpos - startpos) * fps
        current_step_count = 0
        for i in range(steps_count):
            if steps[i].startpos >= startpos and steps[i].startpos <= endpos:
                current_step_count += 1
        s.properties['current_step_s'] = current_step_count / (endpos - startpos) * fps
        
        if s.turns_count >= 1:
            
            turns_durs = [turn.dict['turn_duration'] for turn in s.turns]
            turns_angles = [turn.dict['turn_angle'] for turn in s.turns]
            turns_angular_velocitys = [turn.dict['turn_angular_velocity'] for turn in s.turns]
            turn_angle_overall = fdirs.list[s.turns[s.turns_count - 1].endpos] - fdirs.list[s.turns[0].startpos]
            s.properties.update({'turn_angle': abs(turn_angle_overall)})
            
            if abs(turn_angle_overall) >= settings['large_turn']:
                if turn_angle_overall > settings['large_turn']:
                    s.properties.update({'turn_laterality': 'right'})
                elif turn_angle_overall < -settings['large_turn']:
                    s.properties.update({'turn_laterality': 'left'})
                s.properties.update({
                    'turn_dur': turns_durs[turns_angles.index(max(turns_angles))],
                    'turn_angular_velocity': turns_angular_velocitys[turns_angles.index(max(turns_angles))]})
        
        startpos = s.startpos
        coastpos = s.coastpos
        if s.accel != None and s.turns_count >= 1:
            if abs(turn_angle_overall) >= settings['min_turn_angle']:
                startpos = min(s.accel.startpos, s.turns[0].startpos)
                coastpos = max(s.accel.endpos, s.turns[s.turns_count - 1].endpos)
        a = speeds.list[startpos]
        b = speeds.list[coastpos]
        if s.turns_count >= 1:
            angle = abs(turn_angle_overall)
        else:
            angle = 0
        s.properties.update({'velocity_change': math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(angle))})
        
        if s.bends_count == 0:
            
            continue
        
        elif s.bends_count == 1:
            
            s.properties.update({
                'bend_angle_reached': max(abs(s.bends[0].dict['angle start']), abs(s.bends[0].dict['angle end'])),
                'bend_laterality': s.bends[0].dict['bend_laterality'],
                'bend_angle_traveled': s.bends[0].dict['angle_change'],
                'bend_dur_total': s.bends[0].dict['bend_dur'],
                'bend_angular_velocity': s.bends[0].dict['bend_angular_velocity'],
                'bend_pos': s.bends[0].dict['bend_pos']})
        
        elif s.bends_count == 2:
            
            s.properties.update({
                'bend_angle_reached': abs(s.bends[0].dict['angle end']),
                'bend_laterality': s.bends[0].dict['bend_laterality'],
                'bend_angle_traveled': s.bends[0].dict['angle_change'] + s.bends[1].dict['angle_change'],
                'bend_dur_total': s.bends[0].dict['bend_dur'] + s.bends[1].dict['bend_dur'],
                'bend_angular_velocity': s.bends[0].dict['bend_angular_velocity'],
                'bend_pos': s.bends[0].dict['bend_pos']})
            if s.bends[0].dict['angle_change'] < s.bends[1].dict['angle_change']:
                s.properties['max angle pos'] = 1
        
        elif s.bends_count >= 3:
            
            angles_reached = [s.bends[i].dict['angle end'] for i in range(s.bends_count - 1)]
            angles_traveled = [s.bends[i].dict['angle_change'] for i in range(s.bends_count)]
            angular_velocitys = [bend.dict['bend_angular_velocity'] for bend in s.bends]
            durs = [bend.dict['bend_dur'] for bend in s.bends]
            bend_pos = [s.bends[i].dict['bend_pos'] for i in range(s.bends_count)]
            
            angle_max = 0
            angle_max_left = 0
            angle_max_right = 0
            for angle in angles_reached:
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
            max_angle_pos = [abs(angle) for angle in angles_reached].index(angle_max)
            
            s.properties.update({
                'bend_angle_reached': angle_max,
                'bend_laterality': angle_laterality,
                'bend_angle_traveled': sum(angles_traveled),
                'bend_dur_total': sum(durs),
                'bend_angular_velocity': max(angular_velocitys),
                'bend_pos': max(bend_pos),
                'max angle pos': max_angle_pos})
            
        s.properties.update({'bend_wave_freq': s.bends_count / s.properties['bend_dur_total']})
    
    steps_df = pd.DataFrame([s.properties for s in steps])
    steps_df = steps_df[steps_df['bend_count'] >= 1]
    steps_methods = {
        'step_length': agg2,
        'speed_change': agg1,
        'velocity_change': agg1,
        'accel': agg2,
        'current_speed': [],
        'step_dur': agg2,
        'coast_dur': agg1,
        'coast_percent': agg2,
        'current_step_s': agg3,
        'current_bend_s': agg3,
        'turn_angle': agg2,
        'turn_dur': agg2,
        'turn_angular_velocity': agg2,
        'bend_count': agg2,
        'bend_angle_reached': agg2,
        'bend_pos': agg2,
        'bend_angle_traveled': agg2,
        'bend_angular_velocity': agg2,
        'bend_dur_total': agg2,
        'bend_wave_freq': agg2}
    steps_DF = DF(steps_df, 'step', steps_methods.keys())
    steps_DF.dfs.update({
        'turn left': steps_df[steps_df['turn_laterality'] == 'left'],
        'turn right': steps_df[steps_df['turn_laterality'] == 'right'],
        'with turn': steps_df[steps_df['turn_laterality'] != 'neutral'],
        'without turn': steps_df[steps_df['turn_laterality'] == 'neutral'],
        'bend left': steps_df[steps_df['bend_laterality'] == 'left'],
        'bend right': steps_df[steps_df['bend_laterality'] == 'right']})
    
    analysis_df.add(steps_DF.agg(steps_methods))
    analysis_df.add(steps_DF.agg(steps_methods, 'turn left'))
    analysis_df.add(steps_DF.agg(steps_methods, 'turn right'))
    analysis_df.add(steps_DF.agg(steps_methods, 'with turn'))
    analysis_df.add(steps_DF.agg(steps_methods, 'without turn'))
    analysis_df.add(steps_DF.agg(steps_methods, 'bend left'))
    analysis_df.add(steps_DF.agg(steps_methods, 'bend right'))
    
    for i in steps_methods:
        analysis_df.add(steps_DF.stratify1(i))
    for i in steps_methods:
        analysis_df.add(steps_DF.stratify2('turn left', 'turn right', i))
        analysis_df.add(steps_DF.stratify2('with turn', 'without turn', i))
        analysis_df.add(steps_DF.stratify2('bend left', 'bend right', i))
    
    speeds_df.to_csv(path + '/' + videoname + '_accelerations_df.csv', index=False)
    fdirs_df.to_csv(path + '/' + videoname + '_turns_df.csv', index=False)
    angles_df.to_csv(path + '/' + videoname + '_angles_df.csv', index=False)
    
    steps_df['video'] = videoname
    steps_all = pd.concat([steps_all, steps_df])
    
    rp = []
    for i in steps_methods.keys():
        for j in steps_methods.keys():
            if i == j:
                continue
            si = steps_df[i]
            if steps_df[i].dtypes != 'O' and steps_df[j].dtypes != 'O':
                sj = steps_df[j]
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
    
    print('Analysis of ' + videoname + ' complete.')
    analyses.append(analysis)
    
    if first_video:
        analyses_df = analysis_df.df[['Type', 'Classify', 'Stratify', 'Parameter', 'Method']]
        first_video = False
    analyses_df[videoname] = analysis_df.df['Value']
    
    with open(path + '/' + videoname + '_analysis_notes.csv', 'w') as f:
        for key in settings:
            f.write(key + ', ' + str(settings[key]) + '\n')

steps_all.to_csv('steps_all.csv', index=False)

analyses = pd.DataFrame(analyses).T
analyses = analyses.drop('videoname')
analyses.columns = videonames
analyses.to_csv('analyses.csv')

analyses_df.to_csv('analyses_df.csv', index=False)
if not settings['spine_analysis']:
    from sys import exit
    exit()

steps_all_adjusted = pd.DataFrame(steps_all)
analyses_df_adjusted = pd.DataFrame(analyses_df)
adjusts = ['current_speed', 'step_length', 'speed_change', 'velocity_change', 'accel']
for videoname in videonames:
    fish_length = analyses.at['fish_length', videoname]
    steps_all_adjusted.loc[steps_all_adjusted['video'] == videoname, adjusts] = steps_all_adjusted.loc[steps_all_adjusted['video'] == videoname, adjusts].transform(lambda a: a / fish_length)
    analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), videoname] = analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), videoname].transform(lambda a: a / fish_length)
steps_all_adjusted.to_csv('steps_all_adjusted.csv', index=False)
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
