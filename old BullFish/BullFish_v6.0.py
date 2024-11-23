import os
import csv
import cv2 as cv
import numpy as np
import math
from decimal import Decimal

pi = Decimal(math.pi)

with open('settings_bullfish.csv', 'r') as f:
    
    settings = {row[0]: row[1] for row in csv.reader(f)}
    
    tank_length = Decimal(settings['tank length (mm)'])
    tank_width = Decimal(settings['tank width (mm)'])
    
    accel_analysis = bool(int(settings['acceleration analysis (0/1)']))
    if accel_analysis:
        accel_cutoff = Decimal(settings['acceleration cutoff (mm/s2)'])
        accel_min_dur = int(settings['minimum duration of an acceleration (frame(s))'])
        accel_min_total = Decimal(settings['minimum speed change of an acceleration (mm/s)'])
        accel_min_max = Decimal(settings['minimum maximum speed slope of an acceleration (mm/s2)'])
        accel_min_mean = Decimal(settings['minimum mean speed slope of an acceleration (mm/s2)'])
        plot_vtgraph = bool(int(settings['plot vt graph (0/1)']))
    spine_analysis = bool(int(settings['angle and turn analysis (0/1)']))
    if spine_analysis:
        turn_cutoff = Decimal(settings['turn cutoff (deg/s)']) * pi / 180
        turn_min_dur = int(settings['minimum duration of a turn (frame(s))'])
        turn_min_amp = Decimal(settings['minimum direction change of a turn (deg)']) * pi / Decimal(180)
        turn_min_max = Decimal(settings['minimum maximum angular velocity of a turn (deg/s)']) * pi / Decimal(180)
        turn_min_mean = Decimal(settings['minimum mean angular velocity of a turn (deg/s)']) * pi / Decimal(180)
        plot_turngraph = bool(int(settings['plot turn graph (0/1)']))
        angle_cutoff = Decimal(settings['spine angular velocity cutoff (deg/s)']) * pi / Decimal(180)
        angle_min_dur = int(settings['minimum duration of a tail beat (frame(s))'])
        angle_min_amp = Decimal(settings['minimum tail bend amplitude of a tail beat (deg)']) * pi / Decimal(180)
        angle_min_max = Decimal(settings['minimum maximum angular velocity of a tail beat (deg/s)']) * pi / Decimal(180)
        angle_min_mean = Decimal(settings['minimum mean angular velocity of a tail beat (deg/s)']) * pi / Decimal(180)
        plot_anglegraph = bool(int(settings['plot spine angle graph (0/1)']))
    
    thresholding = bool(int(settings['Thresholding by Max Entropy? (0/1)']))
    blurr = int(settings['Radius of Gaussian blurring (pixels)'])
    thresholding_sampling = int(settings['Calculate threshold once every ? frame(s)'])
    threshold_reduction = float(settings['Lower threshold by ? %'])
    tail_search_r = int(settings['Length of square for finding the tail point (pixels)']) // 2
    contour_points_dist = int(settings['Approximate distance between spine points (pixels)'])
    head_r = float(settings['Perimeter of head is ? % of the whole fish perimeter']) / 100
    show_binary = bool(int(settings['Show thresholded video? (0/1)']))
    show_annotated = bool(int(settings['Show annotated video? (0/1)']))
    turn_max = Decimal(settings['Maximum acceptable turn (deg/s)']) * pi / 180
    fix = bool(int(settings['Fix abnormal frames? (0/1)']))

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

def cal_direction(x1, y1, x2, y2): #caudal (x2, y2) to cranial (x1, y1)
    if x1 == x2 and y1 > y2:
        return pi / 2
    elif x1 == x2 and y1 < y2:
        return -pi / 2
    elif x1 == x2 and y1 == y2:
        print('cal_direction_Error')
        return Decimal(0)
    inclin = math.atan((y1 - y2) / (x1 - x2))
    if x1 > x2:
        return Decimal(inclin)
    elif x1 < x2 and y1 >= y2:
        return Decimal(inclin) + pi
    elif x1 < x2 and y1 < y2:
        return Decimal(inclin) - pi

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
                self.peaks.append(peak_data(i, j - i, total_change, max_slope, Decimal(total_change) / Decimal(j - i) * Decimal(fps), cdist1[i], freeze[i]))
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
    if filename.endswith("_p.avi"):
        
        gray = cv.VideoCapture(filename)
        if gray.isOpened():
            print('Processing ' + filename + '\n')
        else:
            print('Unable to open ' + filename + '\n')
            continue
        width = int(gray.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(gray.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(gray.get(cv.CAP_PROP_FPS))
        l = int(gray.get(cv.CAP_PROP_FRAME_COUNT))
        split_tup = os.path.splitext(filename)
        videoname = split_tup[0]
        
        path = './' + videoname
        if not os.path.exists(path):
            os.mkdir(path)
        
        if thresholding:
            
            thresholds = [0 for i in range(l)]
            
            for i in range(0, l, thresholding_sampling):
                
                gray.set(cv.CAP_PROP_POS_FRAMES, i)
                ret, frame = gray.read()
                
                if ret:
                    
                    print('\rCalculating threshold for Frame ' + str(i), end = '')
                    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    blurred_frame = cv.GaussianBlur(gray_frame, (blurr, blurr), 0)
                    
                    hist, _ = np.histogram(blurred_frame.ravel(), bins = 256, range=(0, 256))
                    nhist = hist / Decimal(int(hist.sum()))
                    pT = [0 for ii in range(256)]
                    ii = 1
                    while ii <= 255:
                        pT[ii] = pT[ii - 1] + nhist[ii]
                        ii += 1
                    hB = [0 for ii in range(256)]
                    hW = [0 for ii in range(256)]
                    t = 0
                    while t <= 255:
                        if pT[t] > 0:
                            hhB = 0
                            ii = 0
                            while ii <= t:
                                if nhist[ii] > 0:
                                    temp = nhist[ii] / pT[t]
                                    hhB -= temp * temp.log10()
                                ii += 1
                            hB[t] = hhB
                        pTW = 1 - pT[t]
                        if pTW > 0:
                            hhW = 0
                            ii = t + 1
                            while ii <= 255:
                                if nhist[ii] > 0:
                                    temp = nhist[ii] / pTW
                                    hhW -= temp * temp.log10()
                                ii += 1
                            hW[t] = hhW
                        t += 1
                    hmax = hB[0] + hW[0]
                    tmax = 0
                    t = 1
                    while t <= 255:
                        h = hB[t] + hW[t]
                        if h > hmax:
                            hmax = h
                            tmax = t
                        t += 1
                    
                    thresholds[i] = tmax * (1 - threshold_reduction / 100)
                    
            for i in range(thresholding_sampling, l, thresholding_sampling):
                for j in range(i - thresholding_sampling + 1, i):
                    thresholds[j] = (thresholds[i - thresholding_sampling] * (i - j) + thresholds[i] * (j - i + thresholding_sampling)) / thresholding_sampling
            for i in range(l - thresholding_sampling + 1, l):
                thresholds[i] = thresholds[l - thresholding_sampling]
            for i in range(0, l):
                thresholds[i] = round(thresholds[i])
            
            print()
        
        gray.set(cv.CAP_PROP_POS_FRAMES, 0)
        cen = [[0, 0] for i in range(l)]
        spine = [[] for i in range(l)]
        spine_len = [0 for i in range(l)]
        if show_binary:
            binary = cv.VideoWriter(path + '/' + videoname + '_t.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height), 0)
        if show_annotated:
            annotated = cv.VideoWriter(path + '/' + videoname + '_a.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

        i = 0
        while True:
            
            ret, frame = gray.read()
            if ret:
                
                print('\rProcessing frame ' + str(i), end = '')
                gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                blurred_frame = cv.GaussianBlur(gray_frame, (blurr, blurr), 0)
                ret1, tframe = cv.threshold(blurred_frame, thresholds[i], 255, cv.THRESH_BINARY_INV)
                
                contours, hierarchy = cv.findContours(tframe, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                max_contour_d = 0
                for contour in contours:
                    if len(contour) > max_contour_d:
                        max_contour_d = len(contour)
                        fish_contour = contour
                
                sframe = np.zeros((height, width), dtype = np.uint8)
                cv.drawContours(sframe, fish_contour, -1, 255, -1)
                leftmost = tuple(fish_contour[fish_contour[:,:,0].argmin()][0])
                rightmost = tuple(fish_contour[fish_contour[:,:,0].argmax()][0])
                topmost = tuple(fish_contour[fish_contour[:,:,1].argmin()][0])
                bottommost = tuple(fish_contour[fish_contour[:,:,1].argmax()][0])
                for ii in range(topmost[1], bottommost[1] + 1):
                    for jj in range(leftmost[0], rightmost[0] + 1):
                        if cv.pointPolygonTest(fish_contour, (jj, ii), True) >= 0:
                            sframe[ii][jj] = 255
                if show_binary:
                    binary.write(sframe)
                
                moment = cv.moments(fish_contour)
                cen[i][0] = moment['m10'] / moment['m00']
                cen[i][1] = moment['m01'] / moment['m00']
                
                if spine_analysis:
                
                    tail_search_area = 99999999
                    for j in range(max_contour_d):
                        current_area = 0
                        for ii in range(fish_contour[j][0][1] - tail_search_r, fish_contour[j][0][1] + tail_search_r + 1):
                            for jj in range(fish_contour[j][0][0] - tail_search_r, fish_contour[j][0][0] + tail_search_r + 1):
                                if tframe[ii][jj] == 255:
                                    current_area += 1
                        if current_area < tail_search_area:
                            tail_search_area = current_area
                            tail_index = j
                    
                    fish_contour_points = []             
                    j = tail_index
                    loop_terminator = 0
                    while loop_terminator < max_contour_d:
                        fish_contour_points.append([fish_contour[j][0][0], fish_contour[j][0][1]])
                        j += 1
                        loop_terminator += 1
                        if j >= max_contour_d:
                            j -= max_contour_d
                    
                    head_arc = (max_contour_d * head_r)
                    start = contour_points_dist
                    end = max_contour_d - 1
                    while end - start > head_arc:
                        min_dist = 99999999
                        while end - start > head_arc:
                            dist = pyth(fish_contour_points[start][0], fish_contour_points[start][1],
                                        fish_contour_points[end][0], fish_contour_points[end][1])
                            if dist < min_dist:
                                min_dist = dist
                            elif dist > min_dist:
                                spine[i].append([(fish_contour_points[start][0] + fish_contour_points[end + 1][0]) / 2,
                                                 (fish_contour_points[start][1] + fish_contour_points[end + 1][1]) / 2])
                                break
                            end -= 1
                        start += contour_points_dist
                    spine_len[i] = len(spine[i])
                
                if show_annotated:
                    cv.circle(frame, (int(cen[i][0]), int(cen[i][1])), 3, (0, 255, 255), -1)
                    if spine_analysis:
                        for j in range(max_contour_d):
                            colorn = int(j / max_contour_d * 255)
                            cv.circle(frame, (fish_contour_points[j][0], fish_contour_points[j][1]), 1, (0, colorn, 255 - colorn), -1)
                        for j in range(spine_len[i]):
                            cv.circle(frame, (int(spine[i][j][0]), int(spine[i][j][1])), 2, (255, 255, 0), -1)
                    annotated.write(frame)
                
                i += 1
            
            else:
                break
            
        print()
        gray.release()
        if show_binary:
            binary.release()
        if show_annotated:
            annotated.release()
        
        with open(videoname + '_info.csv', 'r') as f:
            info = {row[0]: row[1] for row in csv.reader(f)}
            xl = int(info['x pixel at the left border'])
            xr = int(info['x pixel at the right border'])
            yt = int(info['y pixel at the top border'])
            yb = int(info['y pixel at the bottom border'])
            ratiox = Decimal(tank_length) / Decimal(xr - xl)
            ratioy = Decimal(tank_width) / Decimal(yb - yt)
            
        for i in range(l):
            cen[i][0] = Decimal(cen[i][0]) * ratiox
            cen[i][1] = Decimal(cen[i][1]) * ratioy

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
        
        if accel_analysis:
            
            accel = [0 for i in range(l)]
            for i in range(2, l):
                accel[i] = (speed_avg[i] - speed_avg[i - 1]) * fps
            
            accel_avg = runavg(accel, 2, l)
            
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

            if plot_vtgraph:
                vt_graph = np.zeros((1000, l, 3), np.uint8)
                for i in range(l):
                    cv.circle(vt_graph, (i, 1000 - int(round(speed_avg[i]))), 1, (255, 255, 255), -1)
                for i in range(accel_avg.freq):
                    for j in range(accel_avg.peaks[i].pos, accel_avg.peaks[i].pos + accel_avg.peaks[i].length):
                        cv.circle(vt_graph, (j, 1000 - int(round(speed_avg[j]))), 1, (0, 127, 255), -1)
                cv.imwrite(path + '/' + videoname + '_vt_graph.png', vt_graph)
        
        if spine_analysis:
            
            direction = [0 for i in range(l)]
            for i in range(l):
                head_len = spine_len[i] // 3
                direction[i] = cal_direction(spine[i][spine_len[i] - head_len - 1][0], spine[i][spine_len[i] - head_len - 1][1],
                                             spine[i][spine_len[i] - 1][0], spine[i][spine_len[i] - 1][1])
            turn = [0 for i in range(l)]
            abnormal_frames = []
            for i in range(1, l):
                turn[i] = cal_direction_change(direction[i - 1], direction[i]) * Decimal(fps)
                if turn[i] > turn_max:
                    abnormal_frames.append(i)
            abnormal_frames_n = len(abnormal_frames)
            if abnormal_frames_n > 0 and abnormal_frames[abnormal_frames_n - 1] == l - 1:
                print('Abnormal frames: ', abnormal_frames)
                abnormal_frames.pop(abnormal_frames_n - 1)
                
            total_abs_angle = [0 for i in range(l)]
            headtotail_angle = [0 for i in range(l)]
            tail_angle = [0 for i in range(l)]
            for i in range(l):
                spine_dir = []
                for j in range(1, spine_len[i]):
                    spine_dir.append(cal_direction(spine[i][j][0], spine[i][j][1], spine[i][j - 1][0], spine[i][j - 1][1]))
                spine_angle = []
                for j in range(2, spine_len[i]):
                    spine_angle.append(cal_direction_change(spine_dir[j - 1], spine_dir[j - 2]))
                spine_angle_len = len(spine_angle)
                if spine_angle_len % 2 == 0:
                    for j in range(1, spine_angle_len, 2):
                        total_abs_angle[i] += abs(spine_angle[j] + spine_angle[j - 1])
                else:
                    total_abs_angle[i] += abs(spine_angle[0])
                    for j in range(2, spine_angle_len, 2):
                        total_abs_angle[i] += abs(spine_angle[j] + spine_angle[j - 1])
                headtotail_angle[i] = sum(spine_angle)
                tail_len = (spine_len[i] - 3) // 2
                for j in range(tail_len):
                    tail_angle[i] += spine_angle[j]
            
            if fix and abnormal_frames_n > 0:
                while abnormal_frames_n > 0:
                    i = 0
                    while i < abnormal_frames_n:
                        ii = abnormal_frames[i]
                        direction[ii] = (direction[ii - 1] + direction[ii + 1]) / 2
                        turn[ii] = cal_direction_change(direction[ii - 1], direction[ii]) * Decimal(fps)
                        turn[ii + 1] = cal_direction_change(direction[ii], direction[ii + 1]) * Decimal(fps)
                        total_abs_angle[ii] = (total_abs_angle[ii - 1] + total_abs_angle[ii + 1]) / 2
                        headtotail_angle[ii] = (headtotail_angle[ii - 1] + headtotail_angle[ii + 1]) / 2
                        tail_angle[ii] = (tail_angle[ii - 1] + tail_angle[ii + 1]) / 2
                        if i < abnormal_frames_n - 1 and abnormal_frames[i + 1] == ii + 1 and turn[ii + 1] <= turn_max:
                            abnormal_frames.remove(ii + 1)
                        if turn[ii] <= turn_max:
                            abnormal_frames.remove(ii)
                            i -= 1
                        i += 1
                        abnormal_frames_n = len(abnormal_frames)
            
            turn_avg = runavg(turn, 1, l)
            total_abs_angle_avg = runavg(total_abs_angle, 0, l)
            headtotail_angle_avg = runavg(headtotail_angle, 0, l)
            tail_angle_avg = runavg(tail_angle, 0, l)
            total_abs_angle_change = [0 for i in range(l)]
            headtotail_angle_change = [0 for i in range(l)]
            tail_angle_change = [0 for i in range(l)]
            for i in range(1, l):
                headtotail_angle_change[i] = (headtotail_angle_avg[i] - headtotail_angle_avg[i - 1]) * Decimal(fps)
                total_abs_angle_change[i] = (total_abs_angle_avg[i] - total_abs_angle_avg[i - 1]) * Decimal(fps)
                tail_angle_change[i] = (tail_angle_avg[i] - tail_angle_avg[i - 1]) * Decimal(fps)
            headtotail_angle_change_avg = runavg(headtotail_angle_change, 1, l)
            total_abs_angle_change_avg = runavg(total_abs_angle_change, 1, l)
            tail_angle_change_avg = runavg(tail_angle_change, 1, l)
            
            with open(path + '/' + videoname + '_converted_spine_turns_and_angles.csv', 'w') as file:
                header = ['Direction', 'Turn', 'Turn avg',
                          'headtotail_angle', 'headtotail_angle_avg', 'headtotail_angle_change', 'headtotail_angle_change_avg',
                          'total_abs_angle', 'total_abs_angle_avg', 'total_abs_angle_change', 'total_abs_angle_change_avg',
                          'tail_angle', 'tail_angle_avg', 'tail_angle_change', 'tail_angle_change_avg']
                for word in header:
                    file.write(str(word) + ', ')
                file.write('\n')
                for i in range(l):
                    file.write(str(direction[i]) + ', ' + str(turn[i]) + ', ' + str(turn_avg[i]) + ', '
                               + str(headtotail_angle[i]) + ', ' + str(headtotail_angle_avg[i]) + ', '
                               + str(headtotail_angle_change[i]) + ', ' + str(headtotail_angle_change_avg[i]) + ', '
                               + str(total_abs_angle[i]) + ', ' + str(total_abs_angle_avg[i]) + ', '
                               + str(total_abs_angle_change[i]) + ', ' + str(total_abs_angle_change_avg[i]) + ', '
                               + str(tail_angle[i]) + ', ' + str(tail_angle_avg[i]) + ', '
                               + str(tail_angle_change[i]) + ', ' + str(tail_angle_change_avg[i]) + '\n')
            
            turn_criteria = peak_data(0, turn_min_dur, turn_min_amp, turn_min_max, turn_min_mean, 0, 0)
            turn_avg = splitpn(turn_avg)
            turn_avg_p = gen_peaks(turn_avg.p, turn_cutoff, turn_criteria)
            turn_avg_n = gen_peaks(turn_avg.n, turn_cutoff, turn_criteria)
            
            header = ['Position (frame)', 'Duration (frame(s))', 'Direction change (rad)', 'Maximum angular velocity (rad/s)',
                      'Mean angular velocity (rad/s)', 'Current speed (mm/s)', 'Freeze (0/1)']
            turn_avg_p.write_peaks(path + '/' + videoname + '_turn_clockwise_peaks_data.csv', header)
            turn_avg_n.write_peaks(path + '/' + videoname + '_turn_anticlockwise_peaks_data.csv', header)
            
            if plot_turngraph:
                turn_graph = np.zeros((2000, l, 3), np.uint8)
                for i in range(l):
                    cv.circle(turn_graph, (i, 1000 - int(round(turn_avg.b[i] * 10))), 1, (255, 255, 255), -1)
                for i in range(turn_avg_p.freq):
                    for j in range(turn_avg_p.peaks[i].pos, turn_avg_p.peaks[i].pos + turn_avg_p.peaks[i].length):
                        cv.circle(turn_graph, (j, 1000 - int(round(turn_avg.b[j] * 10))), 1, (0, 255, 0), -1)
                for i in range(turn_avg_n.freq):
                    for j in range(turn_avg_n.peaks[i].pos, turn_avg_n.peaks[i].pos + turn_avg_n.peaks[i].length):
                        cv.circle(turn_graph, (j, 1000 - int(round(turn_avg.b[j] * 10))), 1, (0, 127, 255), -1)
                cv.imwrite(path + '/' + videoname + '_turn_graph.png', turn_graph)
            
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
            
            angle_criteria = peak_data(0, angle_min_dur, angle_min_amp, angle_min_max, angle_min_mean, 0, 1)
            total_abs_angle_change_avg = gen_peaks(total_abs_angle_change_avg, angle_cutoff, angle_criteria)
            
            header = ['Position (frame)', 'Duration (frame(s))', 'Tail bend amplitude (rad)', 'Maximum angular velocity (rad/s)',
                      'Mean angular velocity (rad/s)', 'Current speed (mm/s)', 'Freeze (0/1)']
            total_abs_angle_change_avg.write_peaks(path + '/' + videoname + '_total_abs_angle_peaks_data.csv', header)
            
            headtotail_angle_avg = splitpn(headtotail_angle_avg)
            headtotail_angle_change_p = [0]
            headtotail_angle_change_n = [0]
            for i in range(1, l):
                headtotail_angle_change_p.append((headtotail_angle_avg.p[i] - headtotail_angle_avg.p[i - 1]) * fps)
                headtotail_angle_change_n.append((headtotail_angle_avg.n[i] - headtotail_angle_avg.n[i - 1]) * fps)
            headtotail_angle_change_avg_p = runavg(headtotail_angle_change_p, 1, l)
            headtotail_angle_change_avg_n = runavg(headtotail_angle_change_n, 1, l)
            
            headtotail_angle_change_avg_p = gen_peaks(headtotail_angle_change_avg_p, angle_cutoff, angle_criteria)
            headtotail_angle_change_avg_n = gen_peaks(headtotail_angle_change_avg_n, angle_cutoff, angle_criteria)
            
            headtotail_angle_change_avg_p.write_peaks(path + '/' + videoname + '_headtotail_angle_clockwise_peaks_data.csv', header)
            headtotail_angle_change_avg_n.write_peaks(path + '/' + videoname + '_headtotail_angle_anticlockwise_peaks_data.csv', header)
            
            tail_angle_avg = splitpn(tail_angle_avg)
            tail_angle_change_p = [0]
            tail_angle_change_n = [0]
            for i in range(1, l):
                tail_angle_change_p.append((tail_angle_avg.p[i] - tail_angle_avg.p[i - 1]) * fps)
                tail_angle_change_n.append((tail_angle_avg.n[i] - tail_angle_avg.n[i - 1]) * fps)
            tail_angle_change_avg_p = runavg(tail_angle_change_p, 1, l)
            tail_angle_change_avg_n = runavg(tail_angle_change_n, 1, l)
            
            tail_angle_change_avg_p = gen_peaks(tail_angle_change_avg_p, angle_cutoff, angle_criteria)
            tail_angle_change_avg_n = gen_peaks(tail_angle_change_avg_n, angle_cutoff, angle_criteria)
            
            tail_angle_change_avg_p.write_peaks(path + '/' + videoname + '_tail_angle_clockwise_peaks_data.csv', header)
            tail_angle_change_avg_n.write_peaks(path + '/' + videoname + '_tail_angle_anticlockwise_peaks_data.csv', header)
            
            if plot_anglegraph:
            
                total_abs_angle_graph = np.zeros((2000, l, 3), np.uint8)
                for i in range(l):
                    cv.circle(total_abs_angle_graph, (i + 1, 1000 - int(round(total_abs_angle_avg[i] * fps))), 1, (255, 255, 255), -1)
                for i in range(total_abs_angle_change_avg.freq):
                    for j in range(total_abs_angle_change_avg.peaks[i].pos, total_abs_angle_change_avg.peaks[i].pos + total_abs_angle_change_avg.peaks[i].length):
                        cv.circle(total_abs_angle_graph, (j + 1, 1000 - int(round(total_abs_angle_avg[j] * fps))), 1, (0, 127, 255), -1)
                cv.imwrite(path + '/' + videoname + '_total_abs_angle_graph.png', total_abs_angle_graph)
                
                headtotail_angle_graph = np.zeros((2000, l, 3), np.uint8)
                for i in range(l):
                    cv.circle(headtotail_angle_graph, (i + 1, 1000 - int(round(headtotail_angle_avg.b[i] * fps))), 1, (255, 255, 255), -1)
                for i in range(headtotail_angle_change_avg_p.freq):
                    for j in range(headtotail_angle_change_avg_p.peaks[i].pos, headtotail_angle_change_avg_p.peaks[i].pos + headtotail_angle_change_avg_p.peaks[i].length):
                        cv.circle(headtotail_angle_graph, (j + 1, 1000 - int(round(headtotail_angle_avg.b[j] * fps))), 1, (0, 255, 0), -1)
                for i in range(headtotail_angle_change_avg_n.freq):
                    for j in range(headtotail_angle_change_avg_n.peaks[i].pos, headtotail_angle_change_avg_n.peaks[i].pos + headtotail_angle_change_avg_n.peaks[i].length):
                        cv.circle(headtotail_angle_graph, (j + 1, 1000 - int(round(headtotail_angle_avg.b[j] * fps))), 1, (0, 127, 255), -1)
                cv.imwrite(path + '/' + videoname + '_headtotail_angle_graph.png', headtotail_angle_graph)
                
                tail_angle_graph = np.zeros((2000, l, 3), np.uint8)
                for i in range(l):
                    cv.circle(tail_angle_graph, (i + 1, 1000 - int(round(tail_angle_avg.b[i] * fps))), 1, (255, 255, 255), -1)
                for i in range(tail_angle_change_avg_p.freq):
                    for j in range(tail_angle_change_avg_p.peaks[i].pos, tail_angle_change_avg_p.peaks[i].pos + tail_angle_change_avg_p.peaks[i].length):
                        cv.circle(tail_angle_graph, (j + 1, 1000 - int(round(tail_angle_avg.b[j] * fps))), 1, (0, 255, 0), -1)
                for i in range(tail_angle_change_avg_n.freq):
                    for j in range(tail_angle_change_avg_n.peaks[i].pos, tail_angle_change_avg_n.peaks[i].pos + tail_angle_change_avg_n.peaks[i].length):
                        cv.circle(tail_angle_graph, (j + 1, 1000 - int(round(tail_angle_avg.b[j] * fps))), 1, (0, 127, 255), -1)
                cv.imwrite(path + '/' + videoname + '_tail_angle_graph.png', tail_angle_graph)
            
            headtotail_angle_mean = (headtotail_angle_change_avg_p.height_sum + headtotail_angle_change_avg_n.height_sum) / (headtotail_angle_change_avg_p.freq + headtotail_angle_change_avg_n.freq)
            tail_angle_mean = (tail_angle_change_avg_p.height_sum + tail_angle_change_avg_n.height_sum) / (tail_angle_change_avg_p.freq + tail_angle_change_avg_n.freq)
            tail_angle_velocity_mean = (tail_angle_change_avg_p.meanslope_sum + tail_angle_change_avg_n.meanslope_sum) / Decimal(tail_angle_change_avg_p.freq + tail_angle_change_avg_n.freq)
            analysis.update({
                'Body bend frequency (Hz)': Decimal(total_abs_angle_change_avg.freq) / total_time,
                'Mean body bend amplitude (deg)': Decimal(total_abs_angle_change_avg.height_mean) * 180 / pi,
                '[head-to-tail] Body bend frequency (Hz)': Decimal(headtotail_angle_change_avg_p.freq + headtotail_angle_change_avg_n.freq) / total_time,
                '[head-to-tail] Body bend amplitude per sec (deg/s)': (headtotail_angle_change_avg_p.height_sum + headtotail_angle_change_avg_n.height_sum) * 180 / pi / total_time,
                '[head-to-tail] Clockwise body bend amplitude per sec (deg/s)': headtotail_angle_change_avg_p.height_sum * 180 / pi / total_time,
                '[head-to-tail] Anticlockwise body bend amplitude per sec (deg/s)': headtotail_angle_change_avg_n.height_sum * 180 / pi / total_time,
                '[head-to-tail] Body bend amplitude per sec preference (positive for clockwise)': cal_bias(headtotail_angle_change_avg_p.height_sum, headtotail_angle_change_avg_n.height_sum),
                '[head-to-tail] Mean body bend amplitude (deg)': headtotail_angle_mean * 180 / pi,
                '[head-to-tail] Mean clockwise body bend amplitude (deg)': headtotail_angle_change_avg_p.height_mean * 180 / pi,
                '[head-to-tail] Mean anticlockwise body bend amplitude (deg)': headtotail_angle_change_avg_n.height_mean * 180 / pi,
                '[head-to-tail] Body bend amplitude preference (positive for clockwise)': cal_bias(headtotail_angle_change_avg_p.height_mean, headtotail_angle_change_avg_n.height_mean),
                'Tail bend frequency (Hz)': Decimal(tail_angle_change_avg_p.freq + tail_angle_change_avg_n.freq) / total_time,
                'Clockwise tail bend frequency (Hz)': Decimal(tail_angle_change_avg_p.freq) / total_time,
                'Anticlockwise tail bend frequency (Hz)': Decimal(tail_angle_change_avg_n.freq) / total_time,
                'Tail bend frequency preference (positive for clockwise)': cal_bias(tail_angle_change_avg_p.freq, tail_angle_change_avg_n.freq),
                'Tail bend amplitude per sec (deg/s)': (tail_angle_change_avg_p.height_sum + tail_angle_change_avg_n.height_sum) * 180 / pi / total_time,
                'Clockwise tail bend amplitude per sec (deg/s)': tail_angle_change_avg_p.height_sum * 180 / pi / total_time,
                'Anticlockwise tail bend amplitude per sec (deg/s)': tail_angle_change_avg_n.height_sum * 180 / pi / total_time,
                'Tail bend amplitude per sec preference (positive for clockwise)': cal_bias(tail_angle_change_avg_p.height_sum, tail_angle_change_avg_n.height_sum),
                'Mean tail bend amplitude (deg)': tail_angle_mean * 180 / pi,
                'Mean clockwise tail bend amplitude (deg)': tail_angle_change_avg_p.height_mean * 180 / pi,
                'Mean anticlockwise tail bend amplitude (deg)': tail_angle_change_avg_n.height_mean * 180 / pi,
                'Tail bend amplitude preference (positive for clockwise)': cal_bias(tail_angle_change_avg_p.height_mean, tail_angle_change_avg_n.height_mean),
                'Mean tail bend velocity (deg/s)': tail_angle_velocity_mean * 180 / pi,
                'Mean clockwise tail bend velocity (deg/s)': tail_angle_change_avg_p.meanslope_mean * 180 / pi,
                'Mean anticlockwise tail bend velocity (deg/s)': tail_angle_change_avg_n.meanslope_mean * 180 / pi,
                'Tail bend velocity preference (positive for clockwise)': cal_bias(tail_angle_change_avg_p.meanslope_mean, tail_angle_change_avg_n.meanslope_mean)
            })
            if total_freeze_time > 0:
                headtotail_angle_mean_a = (headtotail_angle_change_avg_p.height_sum_a + headtotail_angle_change_avg_n.height_sum_a) / (headtotail_angle_change_avg_p.freq_a + headtotail_angle_change_avg_n.freq_a)
                tail_angle_mean_a = (tail_angle_change_avg_p.height_sum_a + tail_angle_change_avg_n.height_sum_a) / Decimal(tail_angle_change_avg_p.freq_a + tail_angle_change_avg_n.freq_a)
                tail_angle_velocity_mean_a = (tail_angle_change_avg_p.meanslope_sum_a + tail_angle_change_avg_n.meanslope_sum_a) / Decimal(tail_angle_change_avg_p.freq_a + tail_angle_change_avg_n.freq_a)
                analysis.update({
                    'Body bend frequency during active mvt (Hz)': Decimal(total_abs_angle_change_avg.freq_a) / active_time,
                    'Mean body bend amplitude during active mvt (deg)': total_abs_angle_change_avg.height_mean_a * 180 / pi,
                    '[head-to-tail] Body bend frequency during active mvt (Hz)': Decimal(headtotail_angle_change_avg_p.freq_a + headtotail_angle_change_avg_n.freq_a) / active_time,
                    '[head-to-tail] Body bend amplitude per sec during active mvt (deg/s)': (headtotail_angle_change_avg_p.height_sum_a + headtotail_angle_change_avg_n.height_sum_a) * 180 / pi / active_time,
                    '[head-to-tail] Clockwise body bend amplitude per sec during active mvt (deg/s)': headtotail_angle_change_avg_p.height_sum_a * 180 / pi / active_time,
                    '[head-to-tail] Anticlockwise body bend amplitude per sec during active mvt (deg/s)': headtotail_angle_change_avg_n.height_sum_a * 180 / pi / active_time,
                    '[head-to-tail] Body bend amplitude per sec preference during active mvt (positive for clockwise)': cal_bias(headtotail_angle_change_avg_p.height_sum_a, headtotail_angle_change_avg_n.height_sum_a),
                    '[head-to-tail] Mean body bend amplitude during active mvt (deg)': headtotail_angle_mean_a * 180 / pi,
                    '[head-to-tail] Mean clockwise body bend amplitude during active mvt (deg)': headtotail_angle_change_avg_p.height_mean_a * 180 / pi,
                    '[head-to-tail] Mean anticlockwise body bend amplitude during active mvt (deg)': headtotail_angle_change_avg_n.height_mean_a * 180 / pi,
                    '[head-to-tail] Body bend amplitude preference during active mvt (positive for clockwise)': cal_bias(headtotail_angle_change_avg_p.height_mean_a, headtotail_angle_change_avg_n.height_mean_a),
                    'Tail bend frequency during active mvt (Hz)': Decimal(tail_angle_change_avg_p.freq_a + tail_angle_change_avg_n.freq_a) / active_time,
                    'Clockwise tail bend frequency during active mvt (Hz)': Decimal(tail_angle_change_avg_p.freq_a) / active_time,
                    'Anticlockwise tail bend frequency during active mvt (Hz)': Decimal(tail_angle_change_avg_n.freq_a) / active_time,
                    'Tail bend frequency preference during active mvt (positive for clockwise)': cal_bias(tail_angle_change_avg_p.freq_a, tail_angle_change_avg_n.freq_a),
                    'Tail bend amplitude per sec during active mvt (deg/s)': (tail_angle_change_avg_p.height_sum_a + tail_angle_change_avg_n.height_sum_a) * 180 / pi / active_time,
                    'Clockwise tail bend amplitude per sec during active mvt (deg/s)': tail_angle_change_avg_p.height_sum_a * 180 / pi / active_time,
                    'Anticlockwise tail bend amplitude per sec during active mvt (deg/s)': tail_angle_change_avg_n.height_sum_a * 180 / pi / active_time,
                    'Tail bend amplitude per sec preference during active mvt (positive for clockwise)': cal_bias(tail_angle_change_avg_p.height_sum_a, tail_angle_change_avg_n.height_sum_a),
                    'Mean tail bend amplitude during active mvt (deg)': tail_angle_mean_a * 180 / pi,
                    'Mean clockwise tail bend amplitude  during active mvt(deg)': tail_angle_change_avg_p.height_mean_a * 180 / pi,
                    'Mean anticlockwise tail bend amplitude  during active mvt(deg)': tail_angle_change_avg_n.height_mean_a * 180 / pi,
                    'Tail bend amplitude preference during active mvt (positive for clockwise)': cal_bias(tail_angle_change_avg_p.height_mean_a, tail_angle_change_avg_n.height_mean_a),
                    'Mean tail bend velocity during active mvt (deg/s)': tail_angle_velocity_mean_a * 180 / pi,
                    'Mean clockwise tail bend velocity during active mvt (deg/s)': tail_angle_change_avg_p.meanslope_mean_a * 180 / pi,
                    'Mean anticlockwise tail bend velocity during active mvt (deg/s)': tail_angle_change_avg_n.meanslope_mean_a * 180 / pi,
                    'Tail bend velocity preference during active mvt (positive for clockwise)': cal_bias(tail_angle_change_avg_p.meanslope_mean_a, tail_angle_change_avg_n.meanslope_mean_a)
                })
            
        with open(path + '/' + videoname + '_analysis.csv', 'w') as f:
            for key in analysis:
                f.write(key + ', ' + str(analysis[key]) + '\n')
        
