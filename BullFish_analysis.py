import os
import csv
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks
from copy import copy, deepcopy
from statistics import median, stdev
from pandas import json_normalize
from BullFish_pkg.math import pyth, cal_direction, cal_direction_change
from BullFish_pkg.general import csvtodict, load_settings

default_settings = {
    "tank_x": 210,
    "tank_y": 144,
    "plot_figure": 0,
    "export": 0,
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
    "amplitude_cutoff": 50,
    "min_amplitude_dur": 0.02,
    "min_amplitude_dt": 0,
    "min_amplitude": 2,
    "correlation_portion": 3,
    'find_s0': 1
}

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
            'maxslopepos': 0
            }
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
    
    def __init__(self, peak_datum):
        self.peak_datum = peak_datum
        self.typ = {
            'measure': '',
            'method': '',
            'laterality': '',
            'mode': ''}
        self.intrinsic = {
            'dur': 0,
            'angle': 0,
            'amplitude': 0,
            'angular_velocity': 0,
            'max_angular_velocity': 0}
        self.extrinsic = {
            'dist': 0,
            'curspeed': 0,
            'speed': {},
            'turn': {}}
        self.pos = {
            'step_startpos': 0,
            'step_endpos': 0,
            'bend_pos': 0}
    
    def match(self, typ_criterion):
        if self.typ['measure'] == typ_criterion['measure']:
            if self.typ['method'] == typ_criterion['method']:
                if self.typ['laterality'] == typ_criterion['laterality'] or self.typ['laterality'] in typ_criterion['laterality']:
                    if self.typ['mode'] == typ_criterion['mode'] or self.typ['mode'] in typ_criterion['mode']:
                        return True
        return False
    
    def flatten(self):
        df = json_normalize(self.extrinsic, sep='_')
        self.extrinsic = df.to_dict(orient='records')[0]
    
    def choose(self, key):
        dct = self.__dict__
        for key1 in dct.keys():
            if key in dct[key1].keys():
                return dct[key1][key]

def find_extrinsic(steps, dists, dist1s, speeds, turns, directions, fps):
    
    step_count = len(steps)
    l = len(dists)
    
    for i in range(step_count):
        
        startpos = steps[i].peak_datum['startpos']
        endpos = l if i == step_count - 1 else steps[i + 1].peak_datum['startpos']
        steps[i].pos['step_startpos'] = startpos
        steps[i].pos['step_endpos'] = endpos
        steps[i].extrinsic['dist'] = sum(dists[startpos:endpos])
        steps[i].extrinsic['curspeed'] = dist1s[startpos]
        
        height = 0
        peakpos = 0
        maxslope = 0
        maxslopepos = 0
        for j in range(startpos + 1, endpos):
            if speeds[j] > height:
                height = speeds[j]
                peakpos = j
            slope = speeds[j] - speeds[j - 1]
            if slope > maxslope:
                maxslope = slope
                maxslopepos = j
        change = height - speeds[startpos]
        meanslope = (speeds[peakpos] - speeds[startpos]) / (peakpos - startpos)
        steps[i].extrinsic['speed'].update({
            'max': height,
            'max_accel': maxslope,
            'change': change,
            'mean_accel': meanslope})
        steps[i].pos.update({'speed_peakpos': peakpos,
                             'speed_maxslopepos': maxslopepos})
        
        turns_step = list_set(turns[startpos:endpos], 0, endpos - startpos)
        p_height = max(turns_step.p_list)
        n_height = max(turns_step.n_list)
        if p_height > n_height:
            height = -99999999
            peakpos = 0
            maxslope = 0
            maxslopepos = 0
            for j in range(startpos + 1, endpos):
                if directions[j] > height:
                    height = directions[j]
                    peakpos = j
                slope = directions[j] - directions[j - 1]
                if slope > maxslope:
                    maxslope = slope
                    maxslopepos = j
            change = height - directions[startpos]
            meanslope = (directions[peakpos] - directions[startpos]) / (peakpos - startpos)
        else:
            height = 99999999
            peakpos = 0
            maxslope = 0
            maxslopepos = 0
            for j in range(startpos + 1, endpos):
                if directions[j] < height:
                    height = directions[j]
                    peakpos = j
                slope = directions[j] - directions[j - 1]
                if slope < maxslope:
                    maxslope = slope
                    maxslopepos = j
            change = height - directions[startpos]
            meanslope = (directions[peakpos] - directions[startpos]) / (peakpos - startpos)
        steps[i].extrinsic['turn'].update({
            'max_angular_velocity': maxslope * fps,
            'angle': change,
            'mean_angular_velocity': meanslope * fps})
        steps[i].pos.update({'direction_peakpos': peakpos,
                             'turn_peakpos': maxslopepos})
        
        steps[i].flatten()
        
def steps_calculate(steps, typ_criterion, group):
    selected_steps = []
    for step in steps:
        if step.match(typ_criterion):
            selected_steps.append(step)
    calculations = deepcopy(selected_steps[0].__dict__[group])
    for key in calculations.keys():
        calculations[key] = {'count': 0, 'sum': 0, 'mean': 0, 'max': 0, 'stdev': 0}
        values = [step.__dict__[group][key] for step in selected_steps]
        count = len(values)
        calculations[key]['count'] = per_min(count)
        sm = sum(values)
        calculations[key]['sum'] = per_min(sm)
        calculations[key]['mean'] = sm / count
        calculations[key]['max'] = max(values)
        calculations[key]['stdev'] = stdev(values)
    return calculations

def steps_correlate(steps, typ_criterion, sortby, compute, portion):
    selected_steps = []
    for step in steps:
        if step.match(typ_criterion):
            selected_steps.append(step)
    count = len(selected_steps)
    correlations = {}
    selected_steps.sort(key=lambda a: a.choose(sortby))
    for i in range(portion):
        start = round(i * count / portion)
        end = round((i + 1) * count / portion)
        value = sum([selected_steps[j].choose(compute) for j in range(start, end)]) / (end - start)
        value_name = 'value_' + str(i)
        correlations.update({value_name: value})
    return correlations

def search_value(operations, dictionary):
    for operation in operations:
        same = True
        for key in operation.keys():
            if type(operation[key]) == int or type(operation[key]) == float:
                continue
            if operation[key] != dictionary[key]:
                same = False
        if same:
            return operation

def export_data(data, path):
    if data == []:
        print('There are no data to export for ' + path)
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for datum in data:
            writer.writerow(datum)

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
    with open(path + '/' + videoname + '_fishlengths.csv', 'r') as f:
        fish_lengths = [cell for cell in csv.reader(f)]
    fish_length = median([float(length[0]) for length in fish_lengths])
    
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
    
    accels_per_min = per_min(accels_count)
    total_speed_change = sum([datum['change'] for datum in accels_data])
    total_accel_dur = sum([datum['length'] for datum in accels_data]) / metadata['fps']
    mean_speed_change = total_speed_change / accels_count
    mean_peak_accel = sum([datum['maxslope'] for datum in accels_data]) * metadata['fps'] / accels_count
    mean_accel = sum([(datum['meanslope']) for datum in accels_data]) * metadata['fps'] / accels_count
    mean_accel_dur = total_accel_dur / accels_count
    max_speed_change = max([datum['change'] for datum in accels_data])
    max_peak_accel = max([datum['maxslope'] for datum in accels_data]) * metadata['fps']
    max_accel = max([(datum['meanslope']) for datum in accels_data]) * metadata['fps']
    max_accel_dur = max([datum['length'] for datum in accels_data]) / metadata['fps']
    
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
        'max_accel_dur': max_accel_dur})
    
    export_data(accels_data, path + '/' + videoname + '_accels.csv')
    
    if settings['spine_analysis']:
        
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
        directions = [0 for i in range(l)]
        turns = [0 for i in range(l)]
        with open(path + '/' + videoname + '_directions.csv', 'r') as f:
            direction_temp = [[cell for cell in row] for row in csv.reader(f)]
            direction_temp.pop(0)
            for i in range(l):
                directions[i] = float(direction_temp[i][0])
                turns[i] = float(direction_temp[i][1])
        
        if settings['find_s0']:
            s0s = [0 for i in range(l)]
            with open(path + '/' + videoname + '_s0s.csv', 'r') as f:
                s0_temp = [[cell for cell in row] for row in csv.reader(f)]
                s0_temp.pop(0)
                for i in range(l):
                    s0s[i] = [float(s0_temp[i][0]), float(s0_temp[i][1])]
            s0_dists = [0 for i in range(l)]
            for i in range(1, l):
                s0_dists[i] = pyth(s0s[i], s0s[i - 1])
        else:
            s0s = [spines[i][0] for i in range(l)]
        
        trunk_amplitudes = [[0 for j in range(spine_lens[i] - 2)] for i in range(l)]
        s1_amplitudes = [0 for i in range(l)]
        s0_amplitudes = [0 for i in range(l)]
        for i in range(l):
            if spines[i][spine_lens[i] - 1][0] == spines[i][spine_lens[i] - 2][0]:
                for j in range(spine_lens[i] - 2):
                    trunk_amplitudes[i][j] = abs(spines[i][j][0] - spines[i][spine_lens[i] - 1][0]) * ratio
                s0_amplitudes[i] = abs(s0s[i][0] - spines[i][spine_lens[i] - 1][0]) * ratio
            else:
                m = (spines[i][spine_lens[i] - 1][1] - spines[i][spine_lens[i] - 2][1]) / (spines[i][spine_lens[i] - 1][0] - spines[i][spine_lens[i] - 2][0])
                c = spines[i][spine_lens[i] - 1][1] - m * spines[i][spine_lens[i] - 1][0]
                for j in range(spine_lens[i] - 2):
                    trunk_amplitudes[i][j] = abs(m * spines[i][j][0] - spines[i][j][1] + c) / math.sqrt(m ** 2 + 1) * ratio
                s0_amplitudes[i] = abs(m * s0s[i][0] - s0s[i][1] + c) / math.sqrt(m ** 2 + 1) * ratio
            if spine_lens[i] > 2:
                s1_amplitudes[i] = trunk_amplitudes[i][0]
            
        error_frames = []
        with open(path + '/' + videoname + '_errors.csv', 'r') as f:
            for row in csv.reader(f):
                for cell in row:
                    if cell.isnumeric():
                        error_frames.append(int(cell))
        
        # calculate bend angles, amplitudes
        spine_angles = [[] for i in range(l)]
        s1_angles = [0 for i in range(l)]
        s0_angles = [0 for i in range(l)]
        
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
                s1_angles[i] += spine_angles[i][j - 2]
            
            # calculate s0 angles. left is +, right is -
            head_len = max(2, spine_lens[i] // 3)
            for j in range(spine_lens[i] - head_len - 2):
                s0_angles[i] += spine_angles[i][j]
            s0_dir = cal_direction(s0s[i], spines[i][0])
            s0_angles[i] += cal_direction_change(spine_dirs[0], s0_dir)
            directiono = cal_direction(spines[i][spine_lens[i] - head_len], spines[i][spine_lens[i] - 1])
            s0_angles[i] += cal_direction_change(directiono, spine_dirs[spine_lens[i] - head_len - 2])
            
        s1_angles = errors_correct(s1_angles, error_frames)
        s0_angles = errors_correct(s0_angles, error_frames)
        s1_amplitudes = errors_correct(s1_amplitudes, error_frames)
        s0_amplitudes = errors_correct(s0_amplitudes, error_frames)
        
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
        
        steps = []
        
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
            if datum['length'] >= int(settings['min_turn_dur'] * metadata['fps']):
                if datum['height'] * metadata['fps'] >= settings['min_max_turn_velocity']:
                    if datum['auc'] >= settings['min_turn_angle']:
                        return True
            return False
        turns_p_data = get_peaks(turns.p_list, turns.p_list, settings['turn_cutoff'] / metadata['fps'],
                                 turns_criteria_f, turns_criteria_b, turns_criteria_peak)
        turns_n_data = get_peaks(turns.n_list, turns.n_list, settings['turn_cutoff'] / metadata['fps'],
                                 turns_criteria_f, turns_criteria_b, turns_criteria_peak)
        
        if settings['plot_figure']:
            fig, ax = plt.subplots()
            #ax.plot(turns.original_list, c='y')
            ax.plot(turns.list)
            for datum in turns_p_data:
                plot_data(ax, turns.list, datum['startpos'], datum['endpos'] + 1, 'b')
            scatter_x = [datum['peakpos'] for datum in turns_p_data]
            scatter_y = [turns.list[datum['peakpos']] for datum in turns_p_data]
            ax.scatter(scatter_x, scatter_y, c='b', marker='o')
            for datum in turns_n_data:
                plot_data(ax, turns.list, datum['startpos'], datum['endpos'] + 1, 'r')
            x = [datum['peakpos'] for datum in turns_n_data]
            y = [turns.list[datum['peakpos']] for datum in turns_n_data]
            ax.scatter(x, y, c='r', marker='o')
        
        turns_steps = []
        turns_p_steps = [step_datum(peak_datum) for peak_datum in turns_p_data]
        for step in turns_p_steps:
            step.typ['measure'] = 'turn'
            step.typ['laterality'] = 'right'
            step.intrinsic['dur'] = step.peak_datum['length'] / metadata['fps']
            step.intrinsic['angle'] = step.peak_datum['auc']
            del step.intrinsic['amplitude']
            step.intrinsic['angular_velocity'] = step.intrinsic['angle'] / step.intrinsic['dur']
            step.intrinsic['max_angular_velocity'] = step.peak_datum['height'] * metadata['fps']
            turns_steps.append(step)
        turns_n_steps = [step_datum(peak_datum) for peak_datum in turns_n_data]
        for step in turns_n_steps:
            step.typ['measure'] = 'turn'
            step.typ['laterality'] = 'left'
            step.intrinsic['dur'] = step.peak_datum['length'] / metadata['fps']
            step.intrinsic['angle'] = step.peak_datum['auc']
            del step.intrinsic['amplitude']
            step.intrinsic['angular_velocity'] = step.intrinsic['angle'] / step.intrinsic['dur']
            step.intrinsic['max_angular_velocity'] = step.peak_datum['height'] * metadata['fps']
            turns_steps.append(step)
        turns_steps.sort(key=lambda a: a.choose('startpos'))
        
        turn_typ_criteria = []
        for laterality in ['left', 'right', ['left', 'right']]:
            turn_typ_criteria.append({
                'measure': 'turn',
                'method': '',
                'laterality': laterality,
                'mode': ''})
        
        turns_calculations = []
        for typ_criterion in turn_typ_criteria:
            calculations = steps_calculate(turns_steps, typ_criterion, 'intrinsic')
            for properti in calculations.keys():
                count = calculations[properti]['count']
                calculations[properti].pop('count')
                for calculation in calculations[properti].keys():
                    turns_calculation = {'operation': 'calculation'}
                    turns_calculation.update(typ_criterion)
                    turns_calculation.update({
                        'property': properti,
                        'calculation': calculation,
                        'value': calculations[properti][calculation]})
                    turns_calculations.append(turns_calculation)
            turns_calculation = {'operation': 'calculation'}
            turns_calculation.update(typ_criterion)
            turns_calculation.update({
                'property': '',
                'calculation': 'count',
                'value': count})
            turns_calculations.append(turns_calculation)
        
        turns_comparisons = []
        for turns_calculation in turns_calculations:
            if turns_calculation['laterality'] == 'left':
                turns_calculation_temp = copy(turns_calculation)
                turns_calculation_temp['laterality'] = 'right'
                turns_calculation_temp.pop('value')
                for turns_calculation_j in turns_calculations:
                    turns_calculation_j_temp = copy(turns_calculation_j)
                    turns_calculation_j_temp.pop('value')
                    if turns_calculation_temp == turns_calculation_j_temp:
                        turns_comparison = {'operation': 'comparison'}
                        turns_comparison.update(turns_calculation_temp)
                        comparison = cal_preference(turns_calculation['value'],
                                                    turns_calculation_j['value'])
                        turns_comparison.update({'value': comparison})
                        turns_comparisons.append(turns_comparison)
                        break
        
        export_data(turns_calculations, path + '/' + videoname + '_turns_calculations.csv')
        export_data(turns_comparisons, path + '/' + videoname + '_turns_comparisons.csv')
            
        for turns_calculation in turns_calculations:
            name = ''
            for key in turns_calculation.keys():
                if key != 'value':
                    name_part = str(turns_calculation[key])
                    name += (name_part + '_')
            name = name.replace(',', '')
            analysis.update({name: turns_calculation['value']})
        
        for turns_comparison in turns_comparisons:
            name = ''
            for key in turns_comparison.keys():
                if key != 'value':
                    name_part = str(turns_comparison[key])
                    name += (name_part + '_')
            name = name.replace(',', '')
            analysis.update({name: turns_comparison['value']})
        
        # unit is rad/frame. turning left is +, turning right is -
        s1_angles = list_set(s1_angles, start=0, end=l, window=5)
        s1_angles_ddt = [0 for i in range(l)]
        for i in range(1, l):
            s1_angles_ddt[i] = s1_angles.list[i] - s1_angles.list[i - 1]
        s1_angles_ddt = list_set(s1_angles_ddt, start=1, end=l, window=5)
        
        def s1_angles_p_criteria_f(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def s1_angles_p_criteria_b(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def s1_angles_p_criteria_peak(datum):
            if datum['length'] >= int(settings['min_bend_dur'] * metadata['fps']):
                if datum['maxslope'] * metadata['fps'] >= settings['min_bend_speed']:
                    if datum['change'] >= settings['min_bend_angle']:
                        return True
            return False
        s1_angles_p_data = get_peaks(s1_angles_ddt.p_list, s1_angles.list,
                                     settings['bend_cutoff'] / metadata['fps'],
                                     s1_angles_p_criteria_f,
                                     s1_angles_p_criteria_b,
                                     s1_angles_p_criteria_peak)
        s1_angles_p_data = remove_duplicates(s1_angles_p_data)
        def s1_angles_n_criteria_f(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def s1_angles_n_criteria_b(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def s1_angles_n_criteria_peak(datum):
            if datum['length'] >= int(settings['min_bend_dur'] * metadata['fps']):
                if abs(datum['maxslope']) * metadata['fps'] >= settings['min_bend_speed']:
                    if abs(datum['change']) >= settings['min_bend_angle']:
                        return True
            return False
        s1_angles_n_data = get_peaks(s1_angles_ddt.n_list, s1_angles.list,
                                     settings['bend_cutoff'] / metadata['fps'],
                                     s1_angles_n_criteria_f,
                                     s1_angles_n_criteria_b,
                                     s1_angles_n_criteria_peak)
        s1_angles_n_data = remove_duplicates(s1_angles_n_data)
        
        s1_angles_steps = []
        
        s1_angles_p_steps = [step_datum(peak_datum) for peak_datum in s1_angles_p_data]
        # define typ and find intrinsic properties
        for step in s1_angles_p_steps:
            step.typ['measure'] = 'bend'
            step.typ['method'] = 's1'
            step.typ['laterality'] = 'left'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third > 0:
                step.typ['mode'] = 'unilateral'
            elif second_third > 0:
                step.typ['mode'] = 'bilateral'
            else:
                step.typ['mode'] = 'recoil'
            step.intrinsic['dur'] = step.peak_datum['length'] / metadata['fps']
            step.intrinsic['angle'] = step.peak_datum['change']
            step.intrinsic['amplitude'] = s1_amplitudes[step.peak_datum['peakpos']] - s1_amplitudes[step.peak_datum['startpos']]
            i = step.peak_datum['maxslopepos']
            for ii in range(1, spine_lens[i] - 2):
                if trunk_amplitudes[i][ii] < step.intrinsic['amplitude'] / 5:
                    step.pos['bend_pos'] = ii / spine_lens[i]
                    break
            step.intrinsic['angular_velocity'] = step.peak_datum['meanslope'] * metadata['fps']
            step.intrinsic['max_angular_velocity'] = step.peak_datum['maxslope'] * metadata['fps']
            if step.typ['mode'] != 'recoil':
                s1_angles_steps.append(step)
        s1_angles_n_steps = [step_datum(peak_datum) for peak_datum in s1_angles_n_data]
        # define typ and find intrinsic properties
        for step in s1_angles_n_steps:
            step.typ['measure'] = 'bend'
            step.typ['method'] = 's1'
            step.typ['laterality'] = 'right'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third < 0:
                step.typ['mode'] = 'unilateral'
            elif second_third < 0:
                step.typ['mode'] = 'bilateral'
            else:
                step.typ['mode'] = 'recoil'
            step.intrinsic['dur'] = step.peak_datum['length'] / metadata['fps']
            step.intrinsic['angle'] = -step.peak_datum['change']
            step.intrinsic['amplitude'] = s1_amplitudes[step.peak_datum['peakpos']] - s1_amplitudes[step.peak_datum['startpos']]
            i = step.peak_datum['maxslopepos']
            for ii in range(1, spine_lens[i] - 2):
                if trunk_amplitudes[i][ii] < step.intrinsic['amplitude'] / 5:
                    step.pos['bend_pos'] = ii / spine_lens[i]
                    break
            step.intrinsic['angular_velocity'] = -step.peak_datum['meanslope'] * metadata['fps']
            step.intrinsic['max_angular_velocity'] = step.peak_datum['maxslope'] * metadata['fps']
            if step.typ['mode'] != 'recoil':
                s1_angles_steps.append(step)
        
        s1_angles_steps.sort(key=lambda a: a.choose('startpos'))
        find_extrinsic(s1_angles_steps, cen_dists, cdist1s, speeds.list, turns.list, directions_free.list, metadata['fps'])
        steps.extend(s1_angles_steps)
        
        if settings['plot_figure']:
            
            fig, ax = plt.subplots()
            ax.plot(s1_angles.list)
            ax.plot(s1_angles_ddt.list, c='y')
            for datum in s1_angles_p_data:
                plot_data(ax, s1_angles.list, datum['startpos'], datum['endpos'] + 1, 'b')
            for datum in s1_angles_n_data:
                plot_data(ax, s1_angles.list, datum['startpos'], datum['endpos'] + 1, 'r')
            
            fig, ax = plt.subplots()
            ax.plot(s1_angles.list)
            speeds_scaled = [(speed / 100) for speed in speeds.list]
            turns_scaled = [(turn * 2) for turn in turns.list]
            ax.plot(speeds_scaled, c='y')
            ax.plot(turns.list, c='g')
            for step in s1_angles_steps:
                plot_data(ax, s1_angles.list, step.peak_datum['startpos'], step.peak_datum['endpos'] + 1, 'r')
                plot_data(ax, speeds_scaled, step.pos['step_startpos'], step.pos['speed_peakpos'] + 1, 'r')
                plot_data(ax, turns_scaled, step.pos['step_startpos'], step.pos['direction_peakpos'] + 1, 'r')
            plot_scatter(ax, s1_angles_steps, 'speed_maxslopepos', speeds_scaled, 'r')
            plot_scatter(ax, s1_angles_steps, 'turn_peakpos', turns.list, 'r')
        
        # unit is rad/frame. turning left is +, turning right is -
        s0_angles = list_set(s0_angles, start=0, end=l, window=5)
        s0_angles_ddt = [0 for i in range(l)]
        for i in range(1, l):
            s0_angles_ddt[i] = s0_angles.list[i] - s0_angles.list[i - 1]
        s0_angles_ddt = list_set(s0_angles_ddt, start=1, end=l, window=5)
        
        def s0_angles_p_criteria_f(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def s0_angles_p_criteria_b(slope, lst):
            if slope < 0.02:
                return True
            else:
                return False
        def s0_angles_p_criteria_peak(datum):
            if datum['length'] >= int(settings['min_bend_dur'] * metadata['fps']):
                if datum['maxslope'] * metadata['fps'] >= settings['min_bend_speed']:
                    if datum['change'] >= settings['min_bend_angle']:
                        return True
            return False
        s0_angles_p_data = get_peaks(s0_angles_ddt.p_list, s0_angles.list,
                                     settings['bend_cutoff'] / metadata['fps'],
                                     s0_angles_p_criteria_f,
                                     s0_angles_p_criteria_b,
                                     s0_angles_p_criteria_peak)
        s0_angles_p_data = remove_duplicates(s0_angles_p_data)
        def s0_angles_n_criteria_f(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def s0_angles_n_criteria_b(slope, lst):
            if slope > -0.02:
                return True
            else:
                return False
        def s0_angles_n_criteria_peak(datum):
            if datum['length'] >= int(settings['min_bend_dur'] * metadata['fps']):
                if abs(datum['maxslope']) * metadata['fps'] >= settings['min_bend_speed']:
                    if abs(datum['change']) >= settings['min_bend_angle']:
                        return True
            return False
        s0_angles_n_data = get_peaks(s0_angles_ddt.n_list, s0_angles.list,
                                     settings['bend_cutoff'] / metadata['fps'],
                                     s0_angles_n_criteria_f,
                                     s0_angles_n_criteria_b,
                                     s0_angles_n_criteria_peak)
        s0_angles_n_data = remove_duplicates(s0_angles_n_data)
        
        s0_angles_steps = []
        
        s0_angles_p_steps = [step_datum(peak_datum) for peak_datum in s0_angles_p_data]
        # define typ and find intrinsic properties
        for step in s0_angles_p_steps:
            step.typ['measure'] = 'bend'
            step.typ['method'] = 's0'
            step.typ['laterality'] = 'left'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third > 0:
                step.typ['mode'] = 'unilateral'
            elif second_third > 0:
                step.typ['mode'] = 'bilateral'
            else:
                step.typ['mode'] = 'recoil'
            step.intrinsic['dur'] = step.peak_datum['length'] / metadata['fps']
            step.intrinsic['angle'] = step.peak_datum['change']
            step.intrinsic['amplitude'] = s0_amplitudes[step.peak_datum['peakpos']] - s0_amplitudes[step.peak_datum['startpos']]
            i = step.peak_datum['maxslopepos']
            for ii in range(1, spine_lens[i] - 2):
                if trunk_amplitudes[i][ii] < step.intrinsic['amplitude'] / 5:
                    step.pos['bend_pos'] = ii / spine_lens[i]
                    break
            step.intrinsic['angular_velocity'] = step.peak_datum['meanslope'] * metadata['fps']
            step.intrinsic['max_angular_velocity'] = step.peak_datum['maxslope'] * metadata['fps']
            if step.typ['mode'] != 'recoil':
                s0_angles_steps.append(step)
        s0_angles_n_steps = [step_datum(peak_datum) for peak_datum in s0_angles_n_data]
        # define typ and find intrinsic properties
        for step in s0_angles_n_steps:
            step.typ['measure'] = 'bend'
            step.typ['method'] = 's0'
            step.typ['laterality'] = 'right'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third < 0:
                step.typ['mode'] = 'unilateral'
            elif second_third < 0:
                step.typ['mode'] = 'bilateral'
            else:
                step.typ['mode'] = 'recoil'
            step.intrinsic['dur'] = step.peak_datum['length'] / metadata['fps']
            step.intrinsic['angle'] = -step.peak_datum['change']
            step.intrinsic['amplitude'] = s0_amplitudes[step.peak_datum['peakpos']] - s0_amplitudes[step.peak_datum['startpos']]
            i = step.peak_datum['maxslopepos']
            for ii in range(1, spine_lens[i] - 2):
                if trunk_amplitudes[i][ii] < step.intrinsic['amplitude'] / 5:
                    step.pos['bend_pos'] = ii / spine_lens[i]
                    break
            step.intrinsic['angular_velocity'] = -step.peak_datum['meanslope'] * metadata['fps']
            step.intrinsic['max_angular_velocity'] = step.peak_datum['maxslope'] * metadata['fps']
            if step.typ['mode'] != 'recoil':
                s0_angles_steps.append(step)
        
        s0_angles_steps.sort(key=lambda a: a.choose('startpos'))
        find_extrinsic(s0_angles_steps, cen_dists, cdist1s, speeds.list, turns.list, directions_free.list, metadata['fps'])
        steps.extend(s0_angles_steps)
        
        if settings['plot_figure']:
            
            fig, ax = plt.subplots()
            ax.plot(s0_angles.list)
            ax.plot(s0_angles_ddt.list, c='y')
            for datum in s0_angles_p_data:
                plot_data(ax, s0_angles.list, datum['startpos'], datum['endpos'] + 1, 'b')
            for datum in s0_angles_n_data:
                plot_data(ax, s0_angles.list, datum['startpos'], datum['endpos'] + 1, 'r')
            
            fig, ax = plt.subplots()
            ax.plot(s0_angles.list)
            ax.plot(speeds_scaled, c='y')
            ax.plot(turns_scaled, c='g')
            for step in s0_angles_steps:
                plot_data(ax, s0_angles.list, step.peak_datum['startpos'], step.peak_datum['endpos'] + 1, 'r')
                plot_data(ax, speeds_scaled, step.pos['step_startpos'], step.pos['speed_peakpos'] + 1, 'r')
                plot_data(ax, turns_scaled, step.pos['step_startpos'], step.pos['direction_peakpos'] + 1, 'r')
            plot_scatter(ax, s0_angles_steps, 'speed_maxslopepos', speeds_scaled, 'r')
            plot_scatter(ax, s0_angles_steps, 'turn_peakpos', turns.list, 'r')
            
        typ_criteria = []
        for method in ['s1', 's0']:
            for laterality in ['left', 'right', ['left', 'right']]:
                for mode in ['unilateral', 'bilateral', ['unilateral', 'bilateral']]:
                    typ_criteria.append({
                        'measure': 'bend',
                        'method': method,
                        'laterality': laterality,
                        'mode': mode})
        
        steps_calculations = []
        for typ_criterion in typ_criteria:
            calculations = steps_calculate(steps, typ_criterion, 'intrinsic')
            for properti in calculations.keys():
                count = calculations[properti]['count']
                calculations[properti].pop('count')
                for calculation in calculations[properti].keys():
                    steps_calculation = {'operation': 'calculation'}
                    steps_calculation.update(typ_criterion)
                    steps_calculation.update({
                        'property': properti,
                        'calculation': calculation,
                        'value': calculations[properti][calculation]})
                    steps_calculations.append(steps_calculation)
            steps_calculation = {'operation': 'calculation'}
            steps_calculation.update(typ_criterion)
            steps_calculation.update({
                'property': '',
                'calculation': 'count',
                'value': count})
            steps_calculations.append(steps_calculation)
            calculations = steps_calculate(steps, typ_criterion, 'extrinsic')
            for properti in calculations.keys():
                count = calculations[properti]['count']
                calculations[properti].pop('count')
                for calculation in calculations[properti].keys():
                    steps_calculation = {'operation': 'calculation'}
                    steps_calculation.update(typ_criterion)
                    steps_calculation.update({
                        'property': properti,
                        'calculation': calculation,
                        'value': calculations[properti][calculation]})
                    steps_calculations.append(steps_calculation)
        
        steps_comparisons = []
        for steps_calculation in steps_calculations:
            if steps_calculation['laterality'] == 'left':
                steps_calculation_temp = copy(steps_calculation)
                steps_calculation_temp['laterality'] = 'right'
                steps_calculation_temp.pop('value')
                for steps_calculation_j in steps_calculations:
                    steps_calculation_j_temp = copy(steps_calculation_j)
                    steps_calculation_j_temp.pop('value')
                    if steps_calculation_temp == steps_calculation_j_temp:
                        steps_comparison = {'operation': 'comparison'}
                        steps_comparison.update(steps_calculation_temp)
                        comparison = cal_preference(steps_calculation['value'],
                                                    steps_calculation_j['value'])
                        steps_comparison.update({'value': comparison})
                        steps_comparison.update({'laterality': 'left vs right'})
                        steps_comparisons.append(steps_comparison)
                        break
        for steps_calculation in steps_calculations:
            if steps_calculation['mode'] == 'unilateral':
                steps_calculation_temp = copy(steps_calculation)
                steps_calculation_temp['mode'] = 'bilateral'
                steps_calculation_temp.pop('value')
                for steps_calculation_j in steps_calculations:
                    steps_calculation_j_temp = copy(steps_calculation_j)
                    steps_calculation_j_temp.pop('value')
                    if steps_calculation_temp == steps_calculation_j_temp:
                        steps_comparison = {'operation': 'comparison'}
                        steps_comparison.update(steps_calculation_temp)
                        comparison = cal_preference(steps_calculation['value'],
                                                    steps_calculation_j['value'])
                        steps_comparison.update({'value': comparison})
                        steps_comparison.update({'mode': 'unilateral vs bilateral'})
                        steps_comparisons.append(steps_comparison)
                        break
        
        steps_correlations = []
        for typ_criterion in typ_criteria:
            dct = steps[0].__dict__
            for intrinsic in dct['intrinsic'].keys():
                for extrinsic in dct['extrinsic'].keys():
                    correlation = steps_correlate(steps, typ_criterion, intrinsic,
                                                  extrinsic, settings['correlation_portion'])
                    steps_correlation = {'operation': 'correlation'}
                    steps_correlation.update(typ_criterion)
                    steps_correlation.update({
                        'sortby': intrinsic,
                        'compute': extrinsic})
                    steps_correlation.update(correlation)
                    steps_correlations.append(steps_correlation)
                    correlation = steps_correlate(steps, typ_criterion, extrinsic,
                                                  intrinsic, settings['correlation_portion'])
                    steps_correlation = {'operation': 'correlation'}
                    steps_correlation.update(typ_criterion)
                    steps_correlation.update({
                        'sortby': extrinsic,
                        'compute': intrinsic})
                    steps_correlation.update(correlation)
                    steps_correlations.append(steps_correlation)
        
        export_data(steps_calculations, path + '/' + videoname + '_steps_calculations.csv')
        export_data(steps_comparisons, path + '/' + videoname + '_steps_comparisons.csv')
        export_data(steps_correlations, path + '/' + videoname + '_steps_correlations.csv')
        
        for steps_calculation in steps_calculations:
            
            name = ''
            for key in steps_calculation.keys():
                if key != 'value':
                    name_part = str(steps_calculation[key])
                    name += (name_part + '_')
            name = name.replace(',', '')
            analysis.update({name: steps_calculation['value']})
        
        for steps_comparison in steps_comparisons:
            name = ''
            for key in steps_comparison.keys():
                if key != 'value':
                    name_part = str(steps_calculation[key])
                    name += (name_part + '_')
            name = name.replace(',', '')
            analysis.update({name: steps_comparison['value']})
        
        for steps_correlation in steps_correlations:
            name = ''
            for key in steps_correlation.keys():
                if 'value' not in key:
                    name_part = str(steps_correlation[key])
                    name += (name_part + '_')
            name = name.replace(',', '')
            for key in steps_correlation.keys():
                if 'value' in key:
                    value_name = name + key
                    analysis.update({value_name: steps_correlation[key]})
        
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
