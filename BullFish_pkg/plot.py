import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from copy import copy, deepcopy
import matplotlib.pyplot as plt

def errors_correct(lst, errors, l):
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
        df = df.sort_values(by=[param])
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
        dfa = dfa.sort_values(by=[param])
        la = len(dfa)
        dfa1 = dfa.iloc[:round(la / 4)]
        dfa2 = dfa.iloc[round(la / 4):round(la * 3 / 4)]
        dfa3 = dfa.iloc[round(la * 3 / 4):la]
        dfb = self.dfs[dfnameb].copy()
        dfb = dfb.sort_values(by=[param])
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
