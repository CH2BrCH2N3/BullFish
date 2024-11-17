import os
import csv
import matplotlib.pyplot as plt
import math
from sys import exit
from scipy.signal import find_peaks
from copy import copy, deepcopy
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
                   'min_amplitude_dt', 'min_amplitude', 'correlation_portion']
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
        correlation_portion = int(settings['correlation_portion'])

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
            meanslope = upheight / Decimal(length) if length > 0 else 0
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
            meanslope = change / Decimal(length) if length > 0 else 0
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

class step_datum:
    
    def __init__(self, peak_datum):
        self.peak_datum = peak_datum
        self.kind = {
            'measure': '',
            'method': '',
            'laterality': '',
            'mode': ''
            }
        self.properties = {
            'dur': 0,
            'angle': 0,
            'angular_velocity': 0,
            'max_angular_velocity': 0
            }
        self.kinematics = {
            'curspeed': 0,
            'dist': 0,
            'speed_change': 0,
            'max_speed': 0,
            'max_accel': 0
            }
    
    def match(self, kind_criterion):
        if self.kind['measure'] == kind_criterion['measure']:
            if self.kind['method'] == kind_criterion['method']:
                if self.kind['laterality'] == kind_criterion['laterality'] or self.kind['laterality'] in kind_criterion['laterality']:
                    if self.kind['mode'] == kind_criterion['mode'] or self.kind['mode'] in kind_criterion['mode']:
                        return True
        return False
    
    def choose(self, key):
        if key in self.properties.keys():
            return self.properties[key]
        elif key in self.kinematics.keys():
            return self.kinematics[key]

def find_steps(data, dists, speeds, dist1s):
    data_count = len(data)
    steps = [step_datum(data[i]) for i in range(data_count)]
    l = len(dists)
    for i in range(data_count):
        startpos = data[i]['startpos']
        endpos = l if i == data_count - 1 else data[i + 1]['startpos']
        steps[i].kinematics['curspeed'] = dist1s[startpos]
        steps[i].kinematics['dist'] = sum(dists[startpos:endpos])
        step_max_speed = 0
        step_max_speed_pos = 0
        for j in range(startpos, endpos):
            if speeds[j] > step_max_speed:
                step_max_speed = speeds[j]
                step_max_speed_pos = j
        steps[i].kinematics['max_speed'] = step_max_speed
        steps[i].kinematics['speed_change'] = step_max_speed - speeds[startpos]
        step_max_accel = 0
        for j in range(startpos + 1, step_max_speed_pos):
            step_accel = (speeds[j] - speeds[j - 1]) * fps
            if step_accel > step_max_accel:
                step_max_accel = step_accel
        steps[i].kinematics['max_accel'] = step_max_accel
    return steps

def steps_calculate(steps, kind_criterion):
    selected_steps = []
    for step in steps:
        if step.match(kind_criterion):
            selected_steps.append(step)
    calculations = step_datum(None).properties
    for key in calculations.keys():
        calculations[key] = {'count': 0, 'sum': 0, 'mean': 0, 'max': 0}
        values = [step.properties[key] for step in selected_steps]
        calculations[key]['count'] = len(values)
        calculations[key]['sum'] = sum(values)
        calculations[key]['mean'] = calculations[key]['sum'] / Decimal(calculations[key]['count'])
        calculations[key]['max'] = max(values)
    return calculations

def steps_correlate(steps, kind_criterion, sortby, compute, portion):
    selected_steps = []
    for step in steps:
        if step.match(kind_criterion):
            selected_steps.append(step)
    count = len(selected_steps)
    correlations = {}
    selected_steps.sort(key=lambda a: a.choose(sortby))
    for i in range(portion):
        start = round(i * count / portion)
        end = round((i + 1) * count / portion)
        value = sum([selected_steps[j].choose(compute) for j in range(start, end)]) / Decimal(end - start)
        value_name = 'value_' + str(i)
        correlations.update({value_name: value})
    return correlations

def search_value(operations, dictionary):
    for operation in operations:
        same = True
        for key in operation.keys():
            if type(operation[key]) == int or type(operation[key]) == Decimal:
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
        if datum['length'] >= int(min_accel_dur * fps) and datum['maxslope'] * fps >= min_max_accel and datum['change'] >= min_speed_change:
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
        ke[i] = speeds_data[i]['height'] ** 2 - speeds.list[speeds_data[i]['startpos']] ** 2
    '''
    if plot_figure:
        fig, ax = plt.subplots()
        ax.plot([i * fps for i in accels.p_list])
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
    total_accel_dur = Decimal(sum([datum['length'] for datum in accels_data])) / fps
    mean_speed_change = total_speed_change / Decimal(accels_count)
    mean_peak_accel = sum([datum['maxslope'] for datum in accels_data]) * fps / Decimal(accels_count)
    mean_accel = sum([(datum['meanslope']) for datum in accels_data]) * fps / Decimal(accels_count)
    mean_accel_dur = total_accel_dur / Decimal(accels_count)
    max_speed_change = max([datum['change'] for datum in accels_data])
    max_peak_accel = max([datum['maxslope'] for datum in accels_data]) * fps
    max_accel = max([(datum['meanslope']) for datum in accels_data]) * fps
    max_accel_dur = Decimal(max([datum['length'] for datum in accels_data])) / fps
    
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
            if datum['length'] >= int(min_turn_dur * fps) and datum['height'] * fps >= min_max_turn_velocity and datum['auc'] >= min_turn_angle:
                return True
            else:
                return False
        turns_p_data = get_peaks(turns.p_list, turns.p_list, turn_cutoff / fps,
                                 turns_criteria_f, turns_criteria_b, turns_criteria_peak)
        turns_n_data = get_peaks(turns.n_list, turns.n_list, turn_cutoff / fps,
                                 turns_criteria_f, turns_criteria_b, turns_criteria_peak)
        
        if plot_figure:
            fig, ax = plt.subplots()
            #ax.plot(turns.original_list, c='y')
            ax.plot(turns.list)
            scatter_x = [datum['peakpos'] for datum in turns_p_data]
            scatter_y = [turns.list[datum['peakpos']] for datum in turns_p_data]
            ax.scatter(scatter_x, scatter_y, c='b', marker='o')
            for datum in turns_p_data:
                y = [turns.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='b')
            x = [datum['peakpos'] for datum in turns_n_data]
            y = [turns.list[datum['peakpos']] for datum in turns_n_data]
            ax.scatter(x, y, c='r', marker='o')
            for datum in turns_n_data:
                y = [turns.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='r')
        
        turns_p_steps = find_steps(turns_p_data, cen_dists, speeds.list, cdist1s)
        for step in turns_p_steps:
            step.kind['measure'] = 'turn'
            step.kind['laterality'] = 'right'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['auc'] * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['height'] * fps * 180 / pi
            steps.append(step)
        turns_n_steps = find_steps(turns_n_data, cen_dists, speeds.list, cdist1s)
        for step in turns_n_steps:
            step.kind['measure'] = 'turn'
            step.kind['laterality'] = 'left'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['auc'] * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['height'] * fps * 180 / pi
            steps.append(step)
        
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
            if datum['length'] >= int(min_bend_dur * fps) and datum['maxslope'] * fps >= min_bend_speed and datum['change'] >= min_bend_angle:
                return True
            else:
                return False
        trunk_angles_p_data = get_peaks(trunk_angles_ddt.p_list, trunk_angles.list,
                                        bend_cutoff / fps,
                                        trunk_angles_p_criteria_f,
                                        trunk_angles_p_criteria_b,
                                        trunk_angles_p_criteria_peak)
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
            if datum['length'] >= int(min_bend_dur * fps) and abs(datum['maxslope']) * fps >= min_bend_speed and abs(datum['change']) >= min_bend_angle:
                return True
            else:
                return False
        trunk_angles_n_data = get_peaks(trunk_angles_ddt.n_list, trunk_angles.list,
                                        bend_cutoff / fps,
                                        trunk_angles_n_criteria_f,
                                        trunk_angles_n_criteria_b,
                                        trunk_angles_n_criteria_peak)
        
        trunk_angles_p_steps = find_steps(trunk_angles_p_data, cen_dists, speeds.list, cdist1s)
        for step in trunk_angles_p_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'trunk'
            step.kind['laterality'] = 'left'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third > 0:
                step.kind['mode'] = 'unilateral'
            elif second_third > 0:
                step.kind['mode'] = 'bilateral'
            else:
                step.kind['mode'] = 'recoil'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['change'] * 180 / pi
            step.properties['angular_velocity'] = step.peak_datum['meanslope'] * fps * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps * 180 / pi
            steps.append(step)
        trunk_angles_n_steps = find_steps(trunk_angles_n_data, cen_dists, speeds.list, cdist1s)
        for step in trunk_angles_n_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'trunk'
            step.kind['laterality'] = 'right'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third < 0:
                step.kind['mode'] = 'unilateral'
            elif second_third < 0:
                step.kind['mode'] = 'bilateral'
            else:
                step.kind['mode'] = 'recoil'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = -step.peak_datum['change'] * 180 / pi
            step.properties['angular_velocity'] = -step.peak_datum['meanslope'] * fps * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps * 180 / pi
            steps.append(step)
        
        if plot_figure:
            
            fig, ax = plt.subplots()
            ax.plot(trunk_angles.list)
            ax.plot(trunk_angles_ddt.list, c='y')
            for datum in trunk_angles_p_data:
                y = [trunk_angles.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='b')
            for datum in trunk_angles_n_data:
                y = [trunk_angles.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='r')
        
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
            if datum['length'] >= int(min_bend_dur * fps) and datum['maxslope'] * fps >= min_bend_speed and datum['change'] >= min_bend_angle:
                return True
            else:
                return False
        tail_angles_p_data = get_peaks(tail_angles_ddt.p_list, tail_angles.list,
                                       bend_cutoff / fps,
                                       tail_angles_p_criteria_f, tail_angles_p_criteria_b,
                                       tail_angles_p_criteria_peak)
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
            if datum['length'] >= int(min_bend_dur * fps) and abs(datum['maxslope']) * fps >= min_bend_speed and abs(datum['change']) >= min_bend_angle:
                return True
            else:
                return False
        tail_angles_n_data = get_peaks(tail_angles_ddt.n_list, tail_angles.list,
                                       bend_cutoff / fps,
                                       tail_angles_n_criteria_f, tail_angles_n_criteria_b,
                                       tail_angles_n_criteria_peak)
        
        tail_angles_p_steps = find_steps(tail_angles_p_data, cen_dists, speeds.list, cdist1s)
        for step in tail_angles_p_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'tail'
            step.kind['laterality'] = 'left'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third > 0:
                step.kind['mode'] = 'unilateral'
            elif second_third > 0:
                step.kind['mode'] = 'bilateral'
            else:
                step.kind['mode'] = 'recoil'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['change'] * 180 / pi
            step.properties['angular_velocity'] = step.peak_datum['meanslope'] * fps * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps * 180 / pi
            steps.append(step)
        tail_angles_n_steps = find_steps(tail_angles_n_data, cen_dists, speeds.list, cdist1s)
        for step in tail_angles_n_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'tail'
            step.kind['laterality'] = 'right'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third < 0:
                step.kind['mode'] = 'unilateral'
            elif second_third < 0:
                step.kind['mode'] = 'bilateral'
            else:
                step.kind['mode'] = 'recoil'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = -step.peak_datum['change'] * 180 / pi
            step.properties['angular_velocity'] = -step.peak_datum['meanslope'] * fps * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps * 180 / pi
            steps.append(step)
        
        if plot_figure:
            
            fig, ax = plt.subplots()
            ax.plot(tail_angles.list)
            ax.plot(tail_angles_ddt.list, c='y')
            for datum in tail_angles_p_data:
                y = [tail_angles.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='b')
            for datum in tail_angles_n_data:
                y = [tail_angles.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='r')
        
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
            if datum['length'] >= int(min_amplitude_dur * fps) and datum['maxslope'] * fps >= min_amplitude_dt and datum['change'] >= min_amplitude:
                return True
            else:
                return False
        amplitudes_p_data = get_peaks(amplitudes_ddt.p_list, amplitudes.list,
                                      amplitude_cutoff / fps,
                                      amplitudes_p_criteria_f, amplitudes_p_criteria_b,
                                      amplitudes_p_criteria_peak)
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
            if datum['length'] >= int(min_amplitude_dur * fps) and abs(datum['maxslope']) * fps >= min_amplitude_dt and abs(datum['change']) >= min_amplitude:
                return True
            else:
                return False
        amplitudes_n_data = get_peaks(amplitudes_ddt.n_list, amplitudes.list,
                                      amplitude_cutoff / fps,
                                      amplitudes_n_criteria_f, amplitudes_n_criteria_b,
                                      amplitudes_n_criteria_peak)
        
        amplitudes_p_steps = find_steps(amplitudes_p_data, cen_dists, speeds.list, cdist1s)
        for step in amplitudes_p_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'amplitude'
            step.kind['laterality'] = 'left'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third > 0:
                step.kind['mode'] = 'unilateral'
            elif second_third > 0:
                step.kind['mode'] = 'bilateral'
            else:
                step.kind['mode'] = 'recoil'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['change']
            step.properties['angular_velocity'] = step.peak_datum['meanslope'] * fps
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps
            steps.append(step)
        amplitudes_n_steps = find_steps(amplitudes_n_data, cen_dists, speeds.list, cdist1s)
        for step in amplitudes_n_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'amplitude'
            step.kind['laterality'] = 'right'
            first_third = (step.peak_datum['start'] * 2 + step.peak_datum['end']) / 3
            second_third = (step.peak_datum['start'] + step.peak_datum['end'] * 2) / 3
            if first_third < 0:
                step.kind['mode'] = 'unilateral'
            elif second_third < 0:
                step.kind['mode'] = 'bilateral'
            else:
                step.kind['mode'] = 'recoil'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = -step.peak_datum['change']
            step.properties['angular_velocity'] = -step.peak_datum['meanslope'] * fps
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps
            steps.append(step)
        
        if plot_figure:
        
            fig, ax = plt.subplots()
            ax.plot(amplitudes.list)
            ax.plot(amplitudes_ddt.list, c='y')
            for datum in amplitudes_p_data:
                y = [amplitudes.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='b')
            for datum in amplitudes_n_data:
                y = [amplitudes.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='r')
        
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
            if datum['length'] >= int(min_bend_dur * fps) and datum['maxslope'] * fps >= min_bend_speed and datum['upheight'] >= min_bend_angle:
                return True
            else:
                return False
        trunk_curvs_data = get_peaks(trunk_curvs_filtered.list, trunk_curvs_filtered.list,
                                     bend_cutoff / fps, trunk_curvs_criteria_f,
                                     trunk_curvs_criteria_b, trunk_curvs_criteria_peak)
        
        trunk_curvs_steps = find_steps(trunk_curvs_data, cen_dists, speeds.list, cdist1s)
        for step in trunk_curvs_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'trunk curvature'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['upheight'] * 180 / pi
            step.properties['angular_velocity'] = step.peak_datum['meanslope'] * fps * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps * 180 / pi
            steps.append(step)
        
        if plot_figure:
            fig, ax = plt.subplots()
            ax.plot(trunk_curvs_filtered.list)
            for datum in trunk_curvs_data:
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                y = [trunk_curvs_filtered.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='r')
            x = [datum['peakpos'] for datum in trunk_curvs_data]
            y = [trunk_curvs_filtered.list[datum['peakpos']] for datum in trunk_curvs_data]
            ax.scatter(x, y, c='r', marker='o')
        
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
            if datum['length'] >= int(min_bend_dur * fps) and datum['maxslope'] * fps >= min_bend_speed and datum['upheight'] >= min_bend_angle:
                return True
            else:
                return False
        total_curvs_data = get_peaks(total_curvs_filtered.list, total_curvs_filtered.list,
                                     bend_cutoff / fps, total_curvs_criteria_f,
                                     total_curvs_criteria_b, total_curvs_criteria_peak)
        
        total_curvs_steps = find_steps(total_curvs_data, cen_dists, speeds.list, cdist1s)
        for step in total_curvs_steps:
            step.kind['measure'] = 'tail bend'
            step.kind['method'] = 'total curvature'
            step.properties['dur'] = step.peak_datum['length'] / fps
            step.properties['angle'] = step.peak_datum['upheight'] * 180 / pi
            step.properties['angular_velocity'] = step.peak_datum['meanslope'] * fps * 180 / pi
            step.properties['max_angular_velocity'] = step.peak_datum['maxslope'] * fps * 180 / pi
            steps.append(step)
        
        if plot_figure:
            fig, ax = plt.subplots()
            ax.plot(total_curvs_filtered.list)
            for datum in total_curvs_data:
                x = [i for i in range(datum['startpos'], datum['endpos'] + 1)]
                y = [total_curvs_filtered.list[i] for i in range(datum['startpos'], datum['endpos'] + 1)]
                ax.plot(x, y, c='r')
            x = [datum['peakpos'] for datum in total_curvs_data]
            y = [total_curvs_filtered.list[datum['peakpos']] for datum in total_curvs_data]
            ax.scatter(x, y, c='r', marker='o')
            
        kind_criteria = []
        for laterality in ['left', 'right', ['left', 'right']]:
            kind_criteria.append({
                'measure': 'turn',
                'method': '',
                'laterality': laterality,
                'mode': ''
                })
        for method in ['trunk', 'tail', 'amplitude']:
            for laterality in ['left', 'right', ['left', 'right']]:
                for mode in ['unilateral', 'bilateral', 'recoil', ['unilateral', 'bilateral']]:
                    kind_criteria.append({
                        'measure': 'tail bend',
                        'method': method,
                        'laterality': laterality,
                        'mode': mode
                        })
        for method in ['trunk curvature', 'total curvature']:
            kind_criteria.append({
                'measure': 'tail bend',
                'method': method,
                'laterality': '',
                'mode': ''
                })
        
        steps_calculations = []
        for kind_criterion in kind_criteria:
            calculations = steps_calculate(steps, kind_criterion)
            for properti in calculations.keys():
                for calculation in calculations[properti].keys():
                    steps_calculation = {'operation': 'calculation'}
                    steps_calculation.update(kind_criterion)
                    steps_calculation.update({
                        'property': properti,
                        'calculation': calculation,
                        'value': calculations[properti][calculation]
                        })
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
                        steps_comparisons.append(steps_comparison)
                        break
        
        steps_correlations = []
        for kind_criterion in kind_criteria:
            for properti in step_datum(None).properties.keys():
                for kinematic in step_datum(None).kinematics.keys():
                    correlation = steps_correlate(steps, kind_criterion, properti,
                                                  kinematic, correlation_portion)
                    steps_correlation = {'operation': 'correlation'}
                    steps_correlation.update(kind_criterion)
                    steps_correlation.update({
                        'sortby': properti,
                        'compute': kinematic
                        })
                    steps_correlation.update(correlation)
                    steps_correlations.append(steps_correlation)
                    correlation = steps_correlate(steps, kind_criterion, kinematic,
                                                  properti, correlation_portion)
                    steps_correlation = {'operation': 'correlation'}
                    steps_correlation.update(kind_criterion)
                    steps_correlation.update({
                        'sortby': kinematic,
                        'compute': properti
                        })
                    steps_correlation.update(correlation)
                    steps_correlations.append(steps_correlation)
        
        meandering_dict = {
            'operation': 'calculation',
            'measure': 'turn',
            'method': '',
            'laterality': ['left', 'right'],
            'mode': '',
            'property': 'angle',
            'calculation': 'sum'
            }
        meandering_operation = search_value(steps_calculations, meandering_dict)
        meandering = meandering_operation['value'] / total_distance
        
        analysis.update({
            'meandering': meandering
        })
        
        export_data(steps_calculations, path + '/' + videoname + '_steps_calculations.csv')
        export_data(steps_comparisons, path + '/' + videoname + '_steps_comparisons.csv')
        export_data(steps_correlations, path + '/' + videoname + '_steps_correlations.csv')
        
        for steps_calculation in steps_calculations:
            name = ''
            for key in steps_calculation.keys():
                if key != 'value':
                    name_part = str(steps_calculation[key])
                    if type(steps_calculation[key]) == list:
                        name_part = name_part.replace(',', '')
                    name += (name_part + '_')
            analysis.update({name: steps_calculation['value']})
        
        for steps_comparison in steps_comparisons:
            name = ''
            for key in steps_comparison.keys():
                if key != 'value':
                    name_part = str(steps_comparison[key])
                    if type(steps_comparison[key]) == list:
                        name_part = name_part.replace(',', '')
                    name += (name_part + '_')
            analysis.update({name: steps_comparison['value']})
        
        for steps_correlation in steps_correlations:
            name = ''
            for key in steps_correlation.keys():
                if 'value' not in key:
                    name_part = str(steps_correlation[key])
                    if type(steps_correlation[key]) == list:
                        name_part = name_part.replace(',', '')
                    name += (name_part + '_')
            for key in steps_correlation.keys():
                if 'value' in key:
                    analysis.update({name: steps_correlation[key]})
        
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
