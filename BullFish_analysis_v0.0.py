import os
import csv
import matplotlib.pyplot as plt
import math
from decimal import Decimal
from statistics import median
pi = Decimal(math.pi)

if not os.path.exists('bullfish_analysis_settings.csv'):
    with open('bullfish_analysis_settings.csv', 'w') as f:
        headers = ['tank_x', 'tank_y', 'sampling_ratio', 'accel_cutoff', 'accel_min_dur', 'accel_min_total',
                   'accel_min_max', 'accel_min_mean', 'spine_analysis', 'turn_cutoff', 'turn_min_dur', 'turn_min_total',
                   'turn_min_max', 'turn_min_mean', 'angle_cutoff', 'angle_min_dur', 'angle_min_total', 'angle_min_max',
                   'angle_min_mean', 'amplitude_cutoff', 'amplitude_min_dur', 'amplitude_min_total', 'amplitude_min_max',
                   'amplitude_min_mean']
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
    sampling_ratio = int(settings['sampling_ratio'])
    accel_cutoff = Decimal(settings['accel_cutoff'])
    accel_min_dur = Decimal(settings['accel_min_dur'])
    accel_min_total = Decimal(settings['accel_min_total'])
    accel_min_max = Decimal(settings['accel_min_max'])
    accel_min_mean = Decimal(settings['accel_min_mean'])
    spine_analysis = bool(int(settings['spine_analysis']))
    if spine_analysis:
        turn_cutoff = Decimal(settings['turn_cutoff']) * pi / 180
        turn_min_dur = Decimal(settings['turn_min_dur'])
        turn_min_total = Decimal(settings['turn_min_total']) * pi / 180
        turn_min_max = Decimal(settings['turn_min_max']) * pi / 180
        turn_min_mean = Decimal(settings['turn_min_mean']) * pi / 180
        angle_cutoff = Decimal(settings['angle_cutoff']) * pi / 180
        angle_min_dur = Decimal(settings['angle_min_dur'])
        angle_min_total = Decimal(settings['angle_min_total']) * pi / 180
        angle_min_max = Decimal(settings['angle_min_max']) * pi / 180
        angle_min_mean = Decimal(settings['angle_min_mean']) * pi / 180
        amplitude_cutoff = Decimal(settings['amplitude_cutoff'])
        amplitude_min_dur = Decimal(settings['amplitude_min_dur'])
        amplitude_min_total = Decimal(settings['amplitude_min_total'])
        amplitude_min_max = Decimal(settings['amplitude_min_max'])
        amplitude_min_mean = Decimal(settings['amplitude_min_mean'])

def pyth(x1, y1, x2, y2):
    try:
        return Decimal.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def runavg(inputlist, start, end):
    outputlist = [0 for index in range(start)]
    outputlist.append(sum(inputlist[1:4]) / 3)
    outputlist.append(sum(inputlist[1:5]) / 4)
    outputlist.append(sum(inputlist[1:6]) / 5)
    for index in range(start + 3, end - 2):
        outputlist.append((outputlist[index - 1] * 5 + inputlist[index + 2] - inputlist[index - 3]) / 5)
    outputlist.append(sum(inputlist[(end - 4):end]) / 4)
    outputlist.append(sum(inputlist[(end - 3):end]) / 3)
    return outputlist

def cal_direction(x1, y1, x2, y2): #from (x1, y1) to (x2, y2)
    if x1 == x2 and y2 > y1:
        return pi / 2
    elif x1 == x2 and y2 < y1:
        return -pi / 2
    elif x1 == x2 and y1 == y2:
        print('cal_direction_Error')
        return Decimal(0)
    else:
        return Decimal(math.atan2(y2 - y1, x2 - x1))

def cal_direction_change(s1, s2): #from s1 to s2
    direction_change = Decimal(s2) - Decimal(s1)
    if direction_change > pi:
        return direction_change - pi * 2
    elif direction_change <= -pi:
        return direction_change + pi * 2
    else:
        return direction_change

def cal_bias(p, n):
    try:
        return (p - n) / (p + n)
    except:
        print('cal_bias Error')
        return 0

class peak_data:
    def __init__(self, pos, length, height, maxslope, meanslope, curspeed, freeze):
        self.pos = pos
        self.length = length
        self.height = height
        self.maxslope = maxslope
        self.meanslope = meanslope
        self.curspeed = curspeed
        self.freeze = freeze

class gen_peaks:
    
    def __init__(self, inputlist, cutoff, criteria):
        
        self.b = inputlist
        self.peaks = []
        i = 1
        while i < l:
            if inputlist[i] > cutoff:
                j = i
                max_slope = 0
                total_change = 0
                while j < l and inputlist[j] > cutoff:
                    total_change += inputlist[j]
                    if inputlist[j] > max_slope:
                        max_slope = inputlist[j]
                    j += 1
                if j >= l:
                    break
                total_change /= Decimal(fps)
                self.peaks.append(peak_data(i, j - i, total_change, max_slope,
                                            Decimal(total_change) / Decimal(j - i) * Decimal(fps), cdist1[i], freeze[i]))
                i = j
            i += 1
        
        i = 0
        while i < len(self.peaks):
            if (self.peaks[i].length < criteria.length
                or self.peaks[i].height < criteria.height
                or self.peaks[i].maxslope < criteria.maxslope
                or self.peaks[i].meanslope < criteria.meanslope
                or self.peaks[i].curspeed < criteria.curspeed):
                self.peaks.pop(i)
                i -= 1
            i += 1
        
        self.freq = len(self.peaks)
        self.length_sum = sum([self.peaks[i].length for i in range(self.freq)])
        self.length_mean = self.length_sum / Decimal(self.freq) if self.freq > 0 else 0
        self.height_sum = sum([self.peaks[i].height for i in range(self.freq)])
        self.height_mean = self.height_sum / Decimal(self.freq) if self.freq > 0 else 0
        self.maxslope_sum = sum([self.peaks[i].maxslope for i in range(self.freq)])
        self.maxslope_mean = self.maxslope_sum / Decimal(self.freq) if self.freq > 0 else 0
        self.meanslope_sum = sum([self.peaks[i].meanslope for i in range(self.freq)])
        self.meanslope_mean = self.meanslope_sum / Decimal(self.freq) if self.freq > 0 else 0
        
        self.freq_a = 0
        self.length_sum_a = 0
        self.height_sum_a = 0
        self.maxslope_sum_a = 0
        self.meanslope_sum_a = 0
        for i in range(self.freq):
            if self.peaks[i].freeze == 0:
                self.freq_a += 1
                self.length_sum_a += self.peaks[i].length
                self.height_sum_a += self.peaks[i].height
                self.maxslope_sum_a += self.peaks[i].maxslope
                self.meanslope_sum_a += self.peaks[i].meanslope
        self.length_mean_a = self.length_sum_a / Decimal(self.freq_a) if self.freq_a > 0 else 0
        self.height_mean_a = self.height_sum_a / Decimal(self.freq_a) if self.freq_a > 0 else 0
        self.maxslope_mean_a = self.maxslope_sum_a / Decimal(self.freq_a) if self.freq_a > 0 else 0
        self.meanslope_mean_a = self.meanslope_sum_a / Decimal(self.freq_a) if self.freq_a > 0 else 0

    def write_peaks(self, name, header):
        with open(name, 'w') as csvfile:
            for word in header:
                csvfile.write(word + ', ')
            csvfile.write('\n')
            for i in range(self.freq):
                data = [str(self.peaks[i].pos), str(self.peaks[i].length),
                        str(self.peaks[i].height), str(self.peaks[i].maxslope),
                        str(self.peaks[i].meanslope), str(self.peaks[i].curspeed),
                        str(self.peaks[i].freeze)]
                for datum in data:
                    csvfile.write(datum + ', ')
                csvfile.write('\n')
    
    def plot_peaks(self, axes, list_toplot, color_b, color_peak, factor=1):
        for item in list_toplot:
            item *= Decimal(factor)
        if color_b != None:
            axes.plot(list_toplot, color=color_b)
        if color_peak != None:
            for peak in self.peaks:
                axes.plot([i for i in range(peak.pos, peak.pos + peak.length)], list_toplot[peak.pos:(peak.pos + peak.length)], color=color_peak)

class splitpn:
    def __init__(self, inputlist):
        self.b = inputlist
        self.p = []
        self.n = []
        l = len(inputlist)
        for i in range(l):
            if inputlist[i] > 0:
                self.p.append(Decimal(inputlist[i]))
                self.n.append(Decimal(0))
            else:
                self.p.append(Decimal(0))
                self.n.append(Decimal(abs(inputlist[i])))

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
        fps = round(float(metadata['fps']))
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
    with open(path + '/' + videoname + '_trackdata(fishlength).csv', 'r') as f:
        fish_lengths = [cell for cell in csv.reader(f)]
    fish_length = median([Decimal(length[0]) for length in fish_lengths])
    
    with open(path + '/' + videoname + '_cen.csv', 'r') as f:
        cen = [[cell for cell in row] for row in csv.reader(f)]
        cen.pop(0)
        for i in range(l):
            cen[i][0] = Decimal(cen[i][0]) * ratio
            cen[i][1] = Decimal(cen[i][1]) * ratio
    
    cdist = [0 for i in range(l)]
    speed = [0 for i in range(l)]
    for i in range(1, l):
        cdist[i] = pyth(cen[i][0], cen[i][1], cen[i - 1][0], cen[i - 1][1])
        speed[i] = cdist[i] * fps

    speed_avg = runavg(speed, 1, l)

    freeze = [0 for i in range(l)]
    for i in range(fps * 3, l):
        cdist1 = pyth(cen[i - fps * 2][0], cen[i - fps * 2][1], cen[i - fps * 3][0], cen[i - fps * 3][1])
        cdist2 = pyth(cen[i - fps][0], cen[i - fps][1], cen[i - fps * 2][0], cen[i - fps * 2][1])
        cdist3 = pyth(cen[i][0], cen[i][1], cen[i - fps][0], cen[i - fps][1])
        if cdist1 < 1 and cdist2 < 1 and cdist3 < 1:
            for j in range(i - fps * 3 + 1, i + 1):
                freeze[j] = 1
        elif cdist1 > 1 and cdist2 > 1 and cdist3 > 1:
            for j in range(i - fps * 2 + 1, i + 1):
                freeze[j] = 0

    total_time = Decimal(l / fps)
    total_freeze_time = Decimal(sum(freeze)) / Decimal(fps)
    active_time = total_time - total_freeze_time
    freeze_count = 0
    for i in range(2, l):
        if freeze[i] - freeze[i - 1] == 1:
            freeze_count += 1

    total_distance = sum(cdist)

    max_distance_1s = 0
    cdist1 = [0 for i in range(l)]
    for i in range(1, l):
        start = max(1, i + 1 - fps // 2)
        end = min(l, i + 1 + fps // 2)
        cdist1[i] = sum([cdist[j] for j in range(start, end)]) * fps / Decimal(end - start)
        if cdist1[i] > max_distance_1s:
            max_distance_1s = cdist1[i]
    
    analysis = {
        'Speed (mm/s)': total_distance / total_time,
        'Speed during active mvt (mm/s)': total_distance / active_time,
        'Max speed (mm/s)': max(speed_avg),
        'Max speed in 1s (mm/s)': max_distance_1s,
        '% of time freezing': total_freeze_time / total_time * 100,
        'Number of freezing episodes per min': Decimal(freeze_count) / total_time * 60
    }
    
    accel = [0 for i in range(l)]
    
    for i in range(sampling_ratio + 1, l, sampling_ratio):
        accel[i] = (speed_avg[i] - speed_avg[i - sampling_ratio]) * fps / Decimal(sampling_ratio)
    for i in range(sampling_ratio, l):
        
        if accel[i] == 0:
            accel[i] = (accel[i + 1] + accel[i - 1]) / 2
    accel_avg = runavg(accel, 2, l)
    
    with open(path + '/' + videoname + '_accel.csv', 'w') as f:
        header = ['accel', 'accel_avg']
        for word in header:
            f.write(str(word) + ', ')
        f.write('\n')
        for i in range(l):
            row = [accel[i], accel_avg[i]]
            for cell in row:
                f.write(str(cell) + ', ')
            f.write('\n')
    
    accel_criteria = peak_data(0, accel_min_dur, accel_min_total, accel_min_max, accel_min_mean, 0, 1)
    accel_avg = gen_peaks(accel_avg, accel_cutoff, accel_criteria)
    
    header = ['Position (frame)', 'Duration (frame(s))', 'Speed change (mm/s)', 'Maximum accleration (mm/s2)',
              'Mean acceleration (mm/s2)', 'Current speed (mm/s)', 'Freeze (0/1)']
    accel_avg.write_peaks(path + '/' + videoname + '_accel_peaks_data.csv', header)
    
    analysis.update({
        'Number of accelerations per sec': Decimal(accel_avg.freq) / total_time,
        'Mean acceleration duration (s)': Decimal(accel_avg.length_mean) / Decimal(fps),
        'Mean peak acceleration (mm/s2)': accel_avg.maxslope_mean,
        'Mean acceleration (mm/s2)': accel_avg.meanslope_mean,
    })
    
    if total_freeze_time > 0:
        analysis.update({
            'Number of accelerations per sec during active mvt': Decimal(accel_avg.freq_a) / active_time,
            'Mean acceleration duration during active mvt (s)': Decimal(accel_avg.length_mean_a) / Decimal(fps),
            'Mean peak acceleration during active mvt (mm/s2)': accel_avg.maxslope_mean_a,
            'Mean acceleration during active mvt (mm/s2)': accel_avg.meanslope_mean_a
        })
        
    if spine_analysis:
        
        spine_len = [0 for i in range(l)]
        with open(path + '/' + videoname + '_spine.csv', 'r') as f:
            spine = [[cell for cell in row] for row in csv.reader(f)]
            spine.pop(0)
            for i in range(l):
                spine_len[i] = int(spine[i][0])
                spine_temp = []
                for j in range(1, spine_len[i] + 1):
                    spine_temp.append([Decimal(spine[i][j * 2 - 1]), Decimal(spine[i][j * 2])])
                spine[i] = spine_temp
        
        directions = [0 for i in range(l)]
        turn = [0 for i in range(l)]
        amplitudes = [0 for i in range(l)]
        with open(path + '/' + videoname + '_direction.csv', 'r') as f:
            direction_temp = [[cell for cell in row] for row in csv.reader(f)]
            direction_temp.pop(0)
            for i in range(l):
                directions[i] = Decimal(direction_temp[i][0])
                turn[i] = Decimal(direction_temp[i][1])
                amplitudes[i] = Decimal(direction_temp[i][2]) * ratio
        spine_angles = [[] for i in range(l)]
        tail_angles = [0 for i in range(l)]
        total_body_curv = [0 for i in range(l)]
        for i in range(l):
            spine_dir = []
            for j in range(1, spine_len[i]):
                spine_dir.append(cal_direction(spine[i][j - 1][0], spine[i][j - 1][1], spine[i][j][0], spine[i][j][1]))
            spine_angle = []
            for j in range(2, spine_len[i]):
                spine_angle.append(cal_direction_change(spine_dir[j - 1], spine_dir[j - 2]))
            spine_angles[i] = spine_angle
            head_len = max(2, (spine_len[i] - 1) // 3)
            for j in range(spine_len[i] - head_len - 3):
                tail_angles[i] += spine_angle[j]
            tail_angles[i] += cal_direction_change(directions[i], spine_dir[spine_len[i] - head_len - 2])
            if tail_angles[i] < 0:
                amplitudes[i] = -amplitudes[i]
            for j in range(spine_len[i] - 2):
                total_body_curv[i] += abs(spine_angle[j])
        
        turn_avg = runavg(turn, 1, l)
        tail_angles_avg = runavg(tail_angles, 0, l)
        tail_angles_change = [0 for i in range(l)]
        amplitudes_avg = runavg(amplitudes, 0, l)
        amplitudes_change = [0 for i in range(l)]
        total_body_curv_avg = runavg(total_body_curv, 0, l)
        total_body_curv_change = [0 for i in range(l)]
        for i in range(1, l):
            tail_angles_change[i] = (tail_angles_avg[i] - tail_angles_avg[i - 1]) * Decimal(fps)
            amplitudes_change[i] = (amplitudes_avg[i] - amplitudes_avg[i - 1]) * Decimal(fps)
            total_body_curv_change[i] = (total_body_curv_avg[i] - total_body_curv_avg[i - 1]) * Decimal(fps)
        tail_angles_change_avg = runavg(tail_angles_change, 1, l)
        amplitudes_change_avg = runavg(amplitudes_change, 1, l)
        total_body_curv_change_avg = runavg(total_body_curv_change, 1, l)
        
        turn_criteria = peak_data(0, turn_min_dur, turn_min_total, turn_min_max, turn_min_mean, 0, 0)
        turn_avg = splitpn(turn_avg)
        turn_avg_p = gen_peaks(turn_avg.p, turn_cutoff, turn_criteria)
        turn_avg_n = gen_peaks(turn_avg.n, turn_cutoff, turn_criteria)
        
        header = ['Position (frame)', 'Duration (frame(s))', 'Direction change (rad)', 'Maximum angular velocity (rad/s)',
                  'Mean angular velocity (rad/s)', 'Current speed (mm/s)', 'Freeze (0/1)']
        turn_avg_p.write_peaks(path + '/' + videoname + '_turn_clockwise_peaks_data.csv', header)
        turn_avg_n.write_peaks(path + '/' + videoname + '_turn_anticlockwise_peaks_data.csv', header)
        
        analysis.update({
            'Number of turn per sec during active mvt': Decimal(turn_avg_p.freq_a + turn_avg_n.freq_a) / active_time,
            'Number of clockwise turn per sec during active mvt': Decimal(turn_avg_p.freq_a) / active_time,
            'Number of anticlockwise turn per sec during active mvt': Decimal(turn_avg_n.freq_a) / active_time,
            'Turn number preference (positive for clockwise)': cal_bias(turn_avg_p.freq_a, turn_avg_n.freq_a),
            'Turn amplitude per sec during active mvt': (turn_avg_p.height_sum_a + turn_avg_n.height_sum_a) * 180 / pi / active_time,
            'Clockwise turn amplitude per sec during active mvt': turn_avg_p.height_sum_a * 180 / pi / active_time,
            'Anticlockwise turn amplitude per sec during active mvt': turn_avg_n.height_sum_a * 180 / pi / active_time,
            'Turn amplitude per sec preference during active mvt (positive for clockwise)': cal_bias(turn_avg_p.height_sum_a, turn_avg_n.height_sum_a),
            'Mean turn amplitude during active mvt (deg)': (turn_avg_p.height_sum_a + turn_avg_n.height_sum_a) / Decimal(turn_avg_p.freq_a + turn_avg_n.freq_a) * 180 / pi,
            'Mean clockwise turn amplitude during active mvt (deg)': turn_avg_p.height_mean_a * 180 / pi,
            'Mean anticlockwise turn amplitude during active mvt (deg)': turn_avg_n.height_mean_a * 180 / pi,
            'Turn amplitude preference during active mvt (positive for clockwise)': cal_bias(turn_avg_p.height_mean_a, turn_avg_n.height_mean_a),
            'Mean angular velocity during active mvt (deg/s)': (turn_avg_p.meanslope_sum_a + turn_avg_n.meanslope_sum_a) / Decimal(turn_avg_p.freq_a + turn_avg_n.freq_a) * 180 / pi,
            'Mean clockwise angular velocity during active mvt (deg/s)': turn_avg_p.meanslope_mean_a * 180 / pi,
            'Mean anticlockwise angular velocity during active mvt (deg/s)': turn_avg_n.meanslope_mean_a * 180 / pi,
            'Turn angular velocity preference during active mvt (positive for clockwise)': cal_bias(turn_avg_p.meanslope_mean_a, turn_avg_n.meanslope_mean_a),
            'Meandering during active mvt (deg/mm)': Decimal(turn_avg_p.height_sum_a + turn_avg_n.height_sum_a) * 180 / pi / total_distance
        })
        
        angle_criteria = peak_data(0, angle_min_dur, angle_min_total, angle_min_max, angle_min_mean, 0, 1)
        tail_angles_avg = splitpn(tail_angles_avg)
        tail_angles_change_p = [0]
        tail_angles_change_n = [0]
        for i in range(1, l):
            tail_angles_change_p.append((tail_angles_avg.p[i] - tail_angles_avg.p[i - 1]) * fps)
            tail_angles_change_n.append((tail_angles_avg.n[i] - tail_angles_avg.n[i - 1]) * fps)
        tail_angles_change_avg_p = runavg(tail_angles_change_p, 1, l)
        tail_angles_change_avg_n = runavg(tail_angles_change_n, 1, l)
        
        tail_angles_change_avg_p = gen_peaks(tail_angles_change_avg_p, angle_cutoff, angle_criteria)
        tail_angles_change_avg_n = gen_peaks(tail_angles_change_avg_n, angle_cutoff, angle_criteria)
        
        tail_angles_change_avg_p.write_peaks(path + '/' + videoname + '_tail_angles_clockwise_peaks_data.csv', header)
        tail_angles_change_avg_n.write_peaks(path + '/' + videoname + '_tail_angles_anticlockwise_peaks_data.csv', header)
        
        tail_angles_mean = (tail_angles_change_avg_p.height_sum + tail_angles_change_avg_n.height_sum) / (tail_angles_change_avg_p.freq + tail_angles_change_avg_n.freq)
        tail_angles_velocity_mean = (tail_angles_change_avg_p.meanslope_sum + tail_angles_change_avg_n.meanslope_sum) / Decimal(tail_angles_change_avg_p.freq + tail_angles_change_avg_n.freq)
        analysis.update({
            'Tail bend frequency (Hz)': Decimal(tail_angles_change_avg_p.freq + tail_angles_change_avg_n.freq) / total_time,
            'Clockwise tail bend frequency (Hz)': Decimal(tail_angles_change_avg_p.freq) / total_time,
            'Anticlockwise tail bend frequency (Hz)': Decimal(tail_angles_change_avg_n.freq) / total_time,
            'Tail bend frequency preference (positive for clockwise)': cal_bias(tail_angles_change_avg_p.freq, tail_angles_change_avg_n.freq),
            'Tail bend amplitude per sec (deg/s)': (tail_angles_change_avg_p.height_sum + tail_angles_change_avg_n.height_sum) * 180 / pi / total_time,
            'Clockwise tail bend amplitude per sec (deg/s)': tail_angles_change_avg_p.height_sum * 180 / pi / total_time,
            'Anticlockwise tail bend amplitude per sec (deg/s)': tail_angles_change_avg_n.height_sum * 180 / pi / total_time,
            'Tail bend amplitude per sec preference (positive for clockwise)': cal_bias(tail_angles_change_avg_p.height_sum, tail_angles_change_avg_n.height_sum),
            'Mean tail bend amplitude (deg)': tail_angles_mean * 180 / pi,
            'Mean clockwise tail bend amplitude (deg)': tail_angles_change_avg_p.height_mean * 180 / pi,
            'Mean anticlockwise tail bend amplitude (deg)': tail_angles_change_avg_n.height_mean * 180 / pi,
            'Tail bend amplitude preference (positive for clockwise)': cal_bias(tail_angles_change_avg_p.height_mean, tail_angles_change_avg_n.height_mean),
            'Mean tail bend velocity (deg/s)': tail_angles_velocity_mean * 180 / pi,
            'Mean clockwise tail bend velocity (deg/s)': tail_angles_change_avg_p.meanslope_mean * 180 / pi,
            'Mean anticlockwise tail bend velocity (deg/s)': tail_angles_change_avg_n.meanslope_mean * 180 / pi,
            'Tail bend velocity preference (positive for clockwise)': cal_bias(tail_angles_change_avg_p.meanslope_mean, tail_angles_change_avg_n.meanslope_mean)
        })
        if total_freeze_time > 0:
            tail_angles_mean_a = (tail_angles_change_avg_p.height_sum_a + tail_angles_change_avg_n.height_sum_a) / Decimal(tail_angles_change_avg_p.freq_a + tail_angles_change_avg_n.freq_a)
            tail_angles_velocity_mean_a = (tail_angles_change_avg_p.meanslope_sum_a + tail_angles_change_avg_n.meanslope_sum_a) / Decimal(tail_angles_change_avg_p.freq_a + tail_angles_change_avg_n.freq_a)
            analysis.update({
                'Tail bend frequency during active mvt (Hz)': Decimal(tail_angles_change_avg_p.freq_a + tail_angles_change_avg_n.freq_a) / active_time,
                'Clockwise tail bend frequency during active mvt (Hz)': Decimal(tail_angles_change_avg_p.freq_a) / active_time,
                'Anticlockwise tail bend frequency during active mvt (Hz)': Decimal(tail_angles_change_avg_n.freq_a) / active_time,
                'Tail bend frequency preference during active mvt (positive for clockwise)': cal_bias(tail_angles_change_avg_p.freq_a, tail_angles_change_avg_n.freq_a),
                'Tail bend amplitude per sec during active mvt (deg/s)': (tail_angles_change_avg_p.height_sum_a + tail_angles_change_avg_n.height_sum_a) * 180 / pi / active_time,
                'Clockwise tail bend amplitude per sec during active mvt (deg/s)': tail_angles_change_avg_p.height_sum_a * 180 / pi / active_time,
                'Anticlockwise tail bend amplitude per sec during active mvt (deg/s)': tail_angles_change_avg_n.height_sum_a * 180 / pi / active_time,
                'Tail bend amplitude per sec preference during active mvt (positive for clockwise)': cal_bias(tail_angles_change_avg_p.height_sum_a, tail_angles_change_avg_n.height_sum_a),
                'Mean tail bend amplitude during active mvt (deg)': tail_angles_mean_a * 180 / pi,
                'Mean clockwise tail bend amplitude  during active mvt(deg)': tail_angles_change_avg_p.height_mean_a * 180 / pi,
                'Mean anticlockwise tail bend amplitude  during active mvt(deg)': tail_angles_change_avg_n.height_mean_a * 180 / pi,
                'Tail bend amplitude preference during active mvt (positive for clockwise)': cal_bias(tail_angles_change_avg_p.height_mean_a, tail_angles_change_avg_n.height_mean_a),
                'Mean tail bend velocity during active mvt (deg/s)': tail_angles_velocity_mean_a * 180 / pi,
                'Mean clockwise tail bend velocity during active mvt (deg/s)': tail_angles_change_avg_p.meanslope_mean_a * 180 / pi,
                'Mean anticlockwise tail bend velocity during active mvt (deg/s)': tail_angles_change_avg_n.meanslope_mean_a * 180 / pi,
                'Tail bend velocity preference during active mvt (positive for clockwise)': cal_bias(tail_angles_change_avg_p.meanslope_mean_a, tail_angles_change_avg_n.meanslope_mean_a)
            })
        
        amplitude_criteria = peak_data(0, amplitude_min_dur, amplitude_min_total, amplitude_min_max, amplitude_min_mean, 0, 1)
        amplitudes_avg = splitpn(amplitudes_avg)
        amplitudes_change_p = [0]
        amplitudes_change_n = [0]
        for i in range(1, l):
            amplitudes_change_p.append((amplitudes_avg.p[i] - amplitudes_avg.p[i - 1]) * fps)
            amplitudes_change_n.append((amplitudes_avg.n[i] - amplitudes_avg.n[i - 1]) * fps)
        amplitudes_change_avg_p = runavg(amplitudes_change_p, 1, l)
        amplitudes_change_avg_n = runavg(amplitudes_change_n, 1, l)
        
        amplitudes_change_avg_p = gen_peaks(amplitudes_change_avg_p, amplitude_cutoff, amplitude_criteria)
        amplitudes_change_avg_n = gen_peaks(amplitudes_change_avg_n, amplitude_cutoff, amplitude_criteria)
        
        
        fig, (amplitudes_ax, tail_angles_ax, turn_ax, accel_ax) = plt.subplots(4, sharex=True)
        amplitudes_change_avg_p.plot_peaks(amplitudes_ax, amplitudes_avg.b, 'y', 'b')
        amplitudes_change_avg_n.plot_peaks(amplitudes_ax, amplitudes_avg.b, None, 'r')
        tail_angles_change_avg_p.plot_peaks(tail_angles_ax, tail_angles_avg.b, 'y', 'b')
        tail_angles_change_avg_n.plot_peaks(tail_angles_ax, tail_angles_avg.b, None, 'r')
        #ax1.set_xlabel('Frame')
        #ax1.set_ylabel('Turn (' + str(turn_scale) + ' rad/s)', color='y')
        turn_avg_p.plot_peaks(turn_ax, turn_avg.b, 'y', 'b')
        turn_avg_n.plot_peaks(turn_ax, turn_avg.b, None, 'r')
        
        accel_avg.plot_peaks(accel_ax, speed_avg, 'y', 'b')
        
        plt.show()
        
        plt.figure()
        scatter_x = []
        scatter_y = []
        for peaki in amplitudes_change_avg_p.peaks:
            for peakj in accel_avg.peaks:
                if peakj.pos <= peaki.pos + peaki.length and peakj.pos + peakj.length >= peaki.pos:
                    scatter_x.append(peaki.height)
                    scatter_y.append(peakj.height)
        for peaki in amplitudes_change_avg_n.peaks:
            for peakj in accel_avg.peaks:
                if peakj.pos <= peaki.pos + peaki.length and peakj.pos + peakj.length >= peaki.pos:
                    scatter_x.append(peaki.height)
                    scatter_y.append(peakj.height)
        plt.scatter(scatter_x, scatter_y)
        plt.show()
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