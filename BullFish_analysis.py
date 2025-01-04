import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats
import numpy as np
from scipy.signal import find_peaks
from copy import copy, deepcopy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from BullFish_pkg.math import pyth, cal_direction, cal_direction_change
from BullFish_pkg.general import create_path, csvtodict, load_settings

default_settings = {
    'save_individually': 0,
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
    'alternate_turn': 0,
    "turn_cutoff": 2,
    "min_turn_dur": 0.02,
    "min_max_turn_velocity": 0,
    "min_turn_angle": 0.052,
    "bend_cutoff": 2,
    "min_bend_dur": 0.02,
    "min_bend_speed": 0,
    "min_bend_angle": 0.035,
    "amp_cutoff": 50,
    "min_amp_dur": 0.02,
    "min_amp_dt": 0,
    "min_amp": 2,
    'min_step_turn': 0.052,
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
            'belong': False}
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
            'coast dur': 0,
            'coast percent': 100,
            'current step per second': 0,
            'turn angle': 0,
            'turn laterality': 'neutral',
            'turn dur': 0,
            'turn angular velocity': 0,
            'current bend per second': 0,
            'bend angle reached': 0,
            'bend laterality': 'neutral',
            'bend pos': 0,
            'bend angle traveled': 0,
            'bend angular velocity': 0,
            'bend dur total': 0,
            'bend wave frequency': 0,
            'bend count': 0,
            'max angle pos': 0}

def p5(s):
    return np.percentile(s, 5)
def p95(s):
    return np.percentile(s, 95)
def ipr(s):
    return p95(s) - p5(s)
aggs = {
    'sum': np.sum,
    'mean': np.mean,
    'std': np.std,
    'median': np.median,
    'p5': p5,
    'p95': p95,
    'ipr': ipr}
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
        self.params = params
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
    
    def stratify(self):
        analysis_list = []
        df = self.df
        for i in self.params:
            keys = copy(list(self.params))
            keys.remove(i)
            df1 = df.loc[df[i] <= self.intervals.loc['25%'][i], keys]
            df2 = df.loc[(df[i] > self.intervals.loc['25%'][i]) & (df[i] < self.intervals.loc['75%'][i]), keys]
            df3 = df.loc[df[i] >= self.intervals.loc['75%'][i], keys]
            df1_mean = df1.mean(numeric_only=True)
            df2_mean = df2.mean(numeric_only=True)
            df3_mean = df3.mean(numeric_only=True)
            for j in keys:
                analysis_list.append(result_dict(self.Type, None, i + '_low', j, 'mean', df1_mean.at[j]))
                analysis_list.append(result_dict(self.Type, None, i + '_mid', j, 'mean', df2_mean.at[j]))
                analysis_list.append(result_dict(self.Type, None, i + '_high', j, 'mean', df3_mean.at[j]))
        return pd.DataFrame(analysis_list)
    
    def stratify2(self, dfnamea, dfnameb):
        analysis_list = []
        dfa = self.dfs[dfnamea]
        dfb = self.dfs[dfnameb]
        for i in self.params:
            keys = copy(list(self.params))
            keys.remove(i)
            dfa1 = dfa.loc[dfa[i] <= self.intervals.loc['25%'][i], keys]
            dfa2 = dfa.loc[(dfa[i] > self.intervals.loc['25%'][i]) & (dfa[i] < self.intervals.loc['75%'][i]), keys]
            dfa3 = dfa.loc[dfa[i] >= self.intervals.loc['75%'][i], keys]
            dfa1_mean = dfa1.mean(numeric_only=True)
            dfa2_mean = dfa2.mean(numeric_only=True)
            dfa3_mean = dfa3.mean(numeric_only=True)
            dfb1 = dfb.loc[dfb[i] <= self.intervals.loc['25%'][i], keys]
            dfb2 = dfb.loc[(dfb[i] > self.intervals.loc['25%'][i]) & (dfb[i] < self.intervals.loc['75%'][i]), keys]
            dfb3 = dfb.loc[dfb[i] >= self.intervals.loc['75%'][i], keys]
            dfb1_mean = dfb1.mean(numeric_only=True)
            dfb2_mean = dfb2.mean(numeric_only=True)
            dfb3_mean = dfb3.mean(numeric_only=True)
            for j in keys:
                analysis_list.append(result_dict(self.Type, dfnamea, i + '_low', j, 'mean', dfa1_mean.at[j]))
                analysis_list.append(result_dict(self.Type, dfnamea, i + '_mid', j, 'mean', dfa2_mean.at[j]))
                analysis_list.append(result_dict(self.Type, dfnamea, i + '_high', j, 'mean', dfa3_mean.at[j]))
                analysis_list.append(result_dict(self.Type, dfnameb, i + '_low', j, 'mean', dfb1_mean.at[j]))
                analysis_list.append(result_dict(self.Type, dfnameb, i + '_mid', j, 'mean', dfb2_mean.at[j]))
                analysis_list.append(result_dict(self.Type, dfnameb, i + '_high', j, 'mean', dfb3_mean.at[j]))
            analysis_list.append(result_dict(self.Type, dfnamea, i + '_low', None, 'count', len(dfa1)))
            analysis_list.append(result_dict(self.Type, dfnamea, i + '_mid', None, 'count', len(dfa2)))
            analysis_list.append(result_dict(self.Type, dfnamea, i + '_high', None, 'count', len(dfa3)))
            analysis_list.append(result_dict(self.Type, dfnameb, i + '_low', None, 'count', len(dfb1)))
            analysis_list.append(result_dict(self.Type, dfnameb, i + '_mid', None, 'count', len(dfb2)))
            analysis_list.append(result_dict(self.Type, dfnameb, i + '_high', None, 'count', len(dfb3)))
        return pd.DataFrame(analysis_list)
    
    def compare(self, dfnamea, dfnameb):
        results = []
        dfa = self.dfs[dfnamea]
        dfb = self.dfs[dfnameb]
        for i in self.params:
            sa = dfa[i]
            sb = dfb[i]
            test = scipy.stats.ttest_ind(sa, sb, equal_var=False)
            ci = test.confidence_interval()
            result = {
                'Type': self.Type,
                'a': dfnamea,
                'b': dfnameb,
                'Parameter': i,
                'Stratify': None,
                'a mean': sa.mean(),
                'b mean': sb.mean(),
                't': test.statistic,
                'low t': ci[0],
                'hi t': ci[1],
                'p': test.pvalue}
            results.append(result)
        return pd.DataFrame(results)
    
    def twoway(self, dfnamea, dfnameb):
        results = pd.DataFrame()
        dfa = self.dfs[dfnamea]
        dfb = self.dfs[dfnameb]
        for i in self.params:
            keys = copy(list(self.params))
            keys.remove(i)
            dfa1 = dfa.loc[dfa[i] <= self.intervals.loc['25%'][i], keys]
            dfa2 = dfa.loc[(dfa[i] > self.intervals.loc['25%'][i]) & (dfa[i] < self.intervals.loc['75%'][i]), keys]
            dfa3 = dfa.loc[dfa[i] >= self.intervals.loc['75%'][i], keys]
            dfb1 = dfb.loc[dfb[i] <= self.intervals.loc['25%'][i], keys]
            dfb2 = dfb.loc[(dfb[i] > self.intervals.loc['25%'][i]) & (dfb[i] < self.intervals.loc['75%'][i]), keys]
            dfb3 = dfb.loc[dfb[i] >= self.intervals.loc['75%'][i], keys]
            for j in keys:
                data = []
                for ii in range(len(dfa1)):
                    data.append({
                        'Value': dfa1[j].iloc[ii],
                        'Group': dfnamea,
                        'Stratify': i + '_low'})
                for ii in range(len(dfa2)):
                    data.append({
                        'Value': dfa2[j].iloc[ii],
                        'Group': dfnamea,
                        'Stratify': i + '_mid'})
                for ii in range(len(dfa3)):
                    data.append({
                        'Value': dfa3[j].iloc[ii],
                        'Group': dfnamea,
                        'Stratify': i + '_high'})
                for ii in range(len(dfb1)):
                    data.append({
                        'Value': dfb1[j].iloc[ii],
                        'Group': dfnameb,
                        'Stratify': i + '_low'})
                for ii in range(len(dfb2)):
                    data.append({
                        'Value': dfb2[j].iloc[ii],
                        'Group': dfnameb,
                        'Stratify': i + '_mid'})
                for ii in range(len(dfb3)):
                    data.append({
                        'Value': dfb3[j].iloc[ii],
                        'Group': dfnameb,
                        'Stratify': i + '_high'})
                data = pd.DataFrame(data)
                if data.isnull().values.any():
                    continue
                model = ols('Value ~ C(Group) * C(Stratify)', data=data).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                result = anova_table['PR(>F)'].T
                result.pop('Residual')
                result['a'] = dfnamea
                result['b'] = dfnameb
                result['Parameter'] = j
                result['Stratify'] = i
                results = pd.concat([results, pd.DataFrame(result).T])
        return results

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
comparisons_all = pd.DataFrame()
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
    thigmotaxis_time /= metadata['fps']
    
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
    
    create_path('Tracks')
    fig, ax = plt.subplots()
    ax.scatter(x=[cen[i][0] for i in range(l)], y=[cen[i][1] for i in range(l)])
    fig.savefig('Tracks/' + videoname + '_track.png')
    
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
    
    if not settings['spine_analysis']:
        if settings['save_individually']:
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
        #ax.plot([i * metadata['fps'] for i in accels.p_list])
        ax.plot(speeds.list, c='b')
        for datum in accels_data:
            x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
            y = [speeds.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
            ax.plot(x, y, c='r')
    
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
    bend_poss = list_set(bend_poss, 0, l, 3)
    
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
        directions_free[i] = directions_free[i - 1] + turns[i]
    directions_free = list_set(directions_free, start=0, end=l, window=3)
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
    
    if settings['plot_figure']:
        fig, ax = plt.subplots()
        ax.plot(angles.list)
        for datum in angles_p_data:
            plot_data(ax, angles.list, datum['startpos'], datum['endpos'] + 1, 'b')
        for datum in angles_n_data:
            plot_data(ax, angles.list, datum['startpos'], datum['endpos'] + 1, 'r')
    
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
            'angle start': abs(start),
            'angle end': abs(end),
            'dur': peak['length'] / metadata['fps'],
            'angular velocity': abs(angle_change) / peak['length'] * metadata['fps'],
            'bend pos': max(bend_poss.list[peak['startpos']:(peak['endpos'] + 1)])}
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
            if not turns_data[j]['belong']:
                if turns_data[j]['endpos'] >= steps[i].accel['startpos'] and turns_data[j]['startpos'] <= steps[i].accel['endpos']:
                    turns_data[j].update({'belong': True})
                    steps[i].turns.append(turns_steps[j])
                    steps[i].turns_peaks.append(turns_data[j])
        for j in range(len(angles_data)):
            if not angles_data[j]['belong']:
                if angles_data[j]['endpos'] >= steps[i].accel['startpos'] and angles_data[j]['startpos'] <= steps[i].accel['endpos']:
                    angles_data[j].update({'belong': True})
                    steps[i].bends.append(angles_bends[j])
                    steps[i].bends_peaks.append(angles_data[j])
    for i in range(len(turns_data)):
        if turns_data[i]['belong'] == False and turns_steps[i]['angle'] > settings['min_step_turn']:
            step = step_datum()
            step.turns_peaks = [turns_data[i]]
            step.turns = [turns_steps[i]]
            for j in range(len(angles_data)):
                if not angles_data[j]['belong']:
                    if angles_data[j]['endpos'] >= turns_data[i]['startpos'] and angles_data[j]['startpos'] <= turns_data[i]['endpos']:
                        angles_data[j].update({'belong': True})
                        step.bends.append(angles_bends[j])
                        step.bends_peaks.append(angles_data[j])
            if len(step.bends) >= 1:
                steps.append(step)
    steps_count = len(steps)
    
    for i in range(steps_count):
        if steps[i].accel != None:
            steps[i].startpos = steps[i].accel['startpos']
            steps[i].coastpos = steps[i].accel['endpos']
        else:
            steps[i].startpos = steps[i].turns_peaks[0]['startpos']
            steps[i].coastpos = steps[i].turns_peaks[0]['endpos']
    steps.sort(key=lambda a: a.startpos)
    for i in range(steps_count - 1):
        steps[i].endpos = steps[i + 1].startpos - 1
    steps[steps_count - 1].endpos = l - 1
    
    front = settings['front_window'] * metadata['fps']
    back = settings['back_window'] * metadata['fps']
    for i in range(len(angles_data)):
        if angles_data[i]['belong'] == False and angles_bends[i]['recoil']:
            for j in range(steps_count):
                if angles_data[i]['startpos'] > steps[j].coastpos:
                    if angles_data[i]['startpos'] <= steps[j].coastpos + back:
                        angles_data[i].update({'belong': True})
                        steps[j].bends_peaks.append(angles_data[i])
                        steps[j].bends.append(angles_bends[i])
                        break
    for i in range(len(angles_data)):
        if angles_data[i]['belong'] == False and not angles_bends[i]['recoil']:
            for j in range(steps_count):
                if angles_data[i]['startpos'] >= steps[j].startpos - front:
                    if angles_data[i]['startpos'] < steps[j].startpos:
                        if len(steps[j].bends) >= 1:
                            if angles_bends[i]['laterality'] == steps[j].bends[0]['laterality']:
                                continue
                        angles_data[i].update({'belong': True})
                        steps[j].bends_peaks.append(angles_data[i])
                        steps[j].bends.append(angles_bends[i])
                        break
    
    for step in steps:
        
        step.properties.update({
            'current speed': cdist1s[step.startpos],
            'step length': sum(cen_dists[(step.startpos + 1):(step.endpos + 1)]),
            'step dur': (step.endpos + 1 - step.startpos) / metadata['fps'],
            'speed change': max(0, speeds.list[step.coastpos] - speeds.list[step.startpos])})
        if step.accel != None:
            step.properties.update({
                'speed change': step.accel['change'],
                'accel': step.accel['meanslope'] * metadata['fps']})
        else:
            step.properties.update({'accel': step.properties['speed change'] * metadata['fps']})
        
        step.properties['coast dur'] = max(0, step.endpos + 1 - step.coastpos) / metadata['fps']
        step.properties['coast percent'] = step.properties['coast dur'] / step.properties['step dur'] * 100
        
        step.turns_count = len(step.turns)
        step.bends_count = len(step.bends)
        step.properties['bend count'] = step.bends_count
        
    for step in steps:
        
        startpos = round(max(0, step.startpos - metadata['fps'] / 2))
        endpos = round(min(l - 1, step.startpos + metadata['fps'] / 2))
        current_bend_count = 0
        for i in range(len(angles_data)):
            bend_startpos = angles_data[i]['startpos']
            if bend_startpos >= startpos and bend_startpos <= endpos:
                current_bend_count += 1
        step.properties['current bend per second'] = current_bend_count / (endpos - startpos) * metadata['fps']
        current_step_count = 0
        for i in range(steps_count):
            if steps[i].startpos >= startpos and steps[i].startpos <= endpos:
                current_step_count += 1
        step.properties['current step per second'] = current_step_count / (endpos - startpos) * metadata['fps']
        
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
            
            continue
        
        elif step.bends_count == 1:
            
            step.properties.update({
                'bend angle reached': max(step.bends[0]['angle start'], step.bends[0]['angle end']),
                'bend laterality': step.bends[0]['laterality'],
                'bend angle traveled': step.bends[0]['angle change'],
                'bend dur total': step.bends[0]['dur'],
                'bend angular velocity': step.bends[0]['angular velocity'],
                'bend pos': step.bends[0]['bend pos']})
        
        elif step.bends_count == 2:
            
            step.properties.update({
                'bend angle reached': step.bends[0]['angle end'],
                'bend laterality': step.bends[0]['laterality'],
                'bend angle traveled': step.bends[0]['angle change'] + step.bends[1]['angle change'],
                'bend dur total': step.bends[0]['dur'] + step.bends[1]['dur'],
                'bend angular velocity': step.bends[0]['angular velocity'],
                'bend pos': step.bends[0]['bend pos']})
            if step.bends[0]['angle change'] < step.bends[1]['angle change']:
                step.properties['max angle pos'] = 1
        
        elif step.bends_count >= 3:
            
            angles_reached = [step.bends[i]['angle end'] for i in range(step.bends_count - 1)]
            angles_traveled = [step.bends[i]['angle change'] for i in range(step.bends_count)]
            angular_velocitys = [bend['angular velocity'] for bend in step.bends]
            durs = [bend['dur'] for bend in step.bends]
            bend_pos = [step.bends[i]['bend pos'] for i in range(step.bends_count)]
            
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
            
            step.properties.update({
                'bend angle reached': angle_max,
                'bend laterality': angle_laterality,
                'bend angle traveled': sum(angles_traveled),
                'bend dur total': sum(durs),
                'bend angular velocity': max(angular_velocitys),
                'bend pos': max(bend_pos),
                'max angle pos': max_angle_pos})
            
        step.properties.update({'bend wave frequency': step.bends_count / step.properties['bend dur total']})
    
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
                plot_data(ax, speeds_scaled, step.startpos, step.coastpos + 1, c)
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
            if step.properties['bend count'] <= 2:
                c = 'g'
            elif step.properties['bend count'] >= 3:
                c = 'b'
            else:
                c = 'r'
            if step.accel != None:
                plot_data(ax, speeds_scaled, step.startpos, step.coastpos + 1, c)
            for peak in step.turns_peaks:
                plot_data(ax, turns.list, peak['startpos'], peak['endpos'] + 1, c)
            for peak in step.bends_peaks:
                plot_data(ax, angles.list, peak['startpos'], peak['endpos'] + 1, c)
    
    analysis_df = pd.DataFrame()
    
    turns_df = pd.DataFrame(turns_steps)
    
    angles_df = pd.DataFrame(angles_bends)
    angles_methods = {'angle change': agg1,
                      'dur': agg1,
                      'bend pos': agg2,
                      'angular velocity': agg2}
    angles_DF = DF(angles_df, 'bend', angles_methods.keys())
    angles_DF.dfs.update({
        'bend left': angles_df[angles_df['laterality'] == 'left'],
        'bend right': angles_df[angles_df['laterality'] == 'right']})
    analysis_df = pd.concat([analysis_df, angles_DF.agg(angles_methods)])
    analysis_df = pd.concat([analysis_df, angles_DF.agg(angles_methods, 'bend left')])
    analysis_df = pd.concat([analysis_df, angles_DF.agg(angles_methods, 'bend right')])
    
    recoils_df = angles_df[angles_df['recoil'] == True]
    recoils_DF = DF(recoils_df, 'recoil', angles_methods.keys())
    recoils_DF.dfs.update({
        'bend left': recoils_df[recoils_df['laterality'] == 'left'],
        'bend right': recoils_df[recoils_df['laterality'] == 'right']})
    analysis_df = pd.concat([analysis_df, recoils_DF.agg(angles_methods)])
    analysis_df = pd.concat([analysis_df, recoils_DF.agg(angles_methods, 'bend left')])
    analysis_df = pd.concat([analysis_df, recoils_DF.agg(angles_methods, 'bend right')])
    
    steps_df = pd.DataFrame([step.properties for step in steps])
    steps_df = steps_df[steps_df['bend count'] >= 1]
    steps_methods = {
        'step length': agg2,
        'speed change': agg1,
        'velocity change': agg1,
        'accel': agg2,
        'current speed': [],
        'step dur': agg2,
        'coast dur': agg1,
        'coast percent': agg2,
        'current step per second': agg3,
        'current bend per second': agg3,
        'turn angle': agg1,
        'turn dur': agg1,
        'turn angular velocity': agg2,
        'bend angle reached': agg2,
        'bend pos': agg2,
        'bend angle traveled': agg2,
        'bend angular velocity': agg2,
        'bend dur total': agg2,
        'bend wave frequency': agg2}
    steps_DF = DF(steps_df, 'step', steps_methods.keys())
    steps_DF.dfs.update({
        'HT': steps_df[steps_df['bend count'] <= 2],
        'MT': steps_df[steps_df['bend count'] >= 3],
        'turn left': steps_df[steps_df['turn laterality'] == 'left'],
        'turn right': steps_df[steps_df['turn laterality'] == 'right'],
        'with turn': steps_df[steps_df['turn laterality'] != 'neutral'],
        'without turn': steps_df[steps_df['turn laterality'] == 'neutral'],
        'bend left': steps_df[steps_df['bend laterality'] == 'left'],
        'bend right': steps_df[steps_df['bend laterality'] == 'right']})
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods)])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'HT')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'MT')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'turn left')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'turn right')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'with turn')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'without turn')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'bend left')])
    analysis_df = pd.concat([analysis_df, steps_DF.agg(steps_methods, 'bend right')])
    
    analysis_df = pd.concat([analysis_df, steps_DF.stratify()])
    analysis_df = pd.concat([analysis_df, steps_DF.stratify2('HT', 'MT')])
    analysis_df = pd.concat([analysis_df, steps_DF.stratify2('turn left', 'turn right')])
    analysis_df = pd.concat([analysis_df, steps_DF.stratify2('with turn', 'without turn')])
    analysis_df = pd.concat([analysis_df, steps_DF.stratify2('bend left', 'bend right')])
    
    comparisons = pd.DataFrame()
    comparisons = pd.concat([comparisons, angles_DF.compare('bend left', 'bend right')])
    comparisons = pd.concat([comparisons, steps_DF.compare('HT', 'MT')])
    comparisons = pd.concat([comparisons, steps_DF.compare('turn left', 'turn right')])
    comparisons = pd.concat([comparisons, steps_DF.compare('with turn', 'without turn')])
    comparisons = pd.concat([comparisons, steps_DF.compare('bend left', 'bend right')])
    comparisons = pd.concat([comparisons, steps_DF.twoway('HT', 'MT')])
    comparisons = pd.concat([comparisons, steps_DF.twoway('turn left', 'turn right')])
    comparisons = pd.concat([comparisons, steps_DF.twoway('with turn', 'without turn')])
    comparisons = pd.concat([comparisons, steps_DF.twoway('bend left', 'bend right')])
    
    if settings['save_individually']:
        turns_df.to_csv(path + '/' + videoname + '_turns_df.csv', index=False)
        angles_df.to_csv(path + '/' + videoname + '_angles_df.csv', index=False)
        steps_df.to_csv(path + '/' + videoname + '_steps_df.csv', index=False)
        comparisons.to_csv(path + '/' + videoname + '_comparisons.csv', index=False)
    
    steps_df['video'] = videoname
    steps_all = pd.concat([steps_all, steps_df])
    comparisons['video'] = videoname
    comparisons_all = pd.concat([comparisons_all, comparisons])
    
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
    
    if settings['save_individually']:
        with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
            for key in analysis:
                f.write(key + ', ' + str(analysis[key]) + '\n')
        print('Analysis of ' + videoname + ' complete.')
    
    analyses.append(analysis)
    
    #analysis_df.to_csv(path + '/' + videoname + '_analysis_df.csv')
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

steps_all_adjusted = pd.DataFrame(steps_all)
analyses_df_adjusted = pd.DataFrame(analyses_df)
adjusts = ['current speed', 'step length', 'speed change', 'velocity change', 'accel']
for videoname in videonames:
    fish_length = analyses.at['fish_length', videoname]
    steps_all_adjusted.loc[steps_all_adjusted['video'] == videoname, adjusts] = steps_all_adjusted.loc[steps_all_adjusted['video'] == videoname, adjusts].transform(lambda a: a / fish_length)
    analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), videoname] = analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), videoname].transform(lambda a: a / fish_length)
steps_all_adjusted.to_csv('steps_all_adjusted.csv', index=False)
analyses_df_adjusted.to_csv('analyses_df_adjusted.csv', index=False)
print('All analyses complete.')

comparisons_all.to_csv('comparisons_all.csv', index=False)
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
