from pathlib import Path
import csv
import cv2 as cv
import pandas as pd
from math import sqrt, cos
from copy import copy
from BullFish_pkg.general import create_path, load_settings
from BullFish_pkg.math import pyth, cal_direction, cal_direction_change
from BullFish_pkg.plot import errors_correct, curve, step, DF, results_df
import matplotlib.pyplot as plt
from BullFish_pkg.plot import plot_data

default_settings = {
    "tank_x": 210,
    "tank_y": 144,
    "plot_figure": 0,
    "sampling": 2,
    "speed_limit": 2000,
    'analysis_extended': 0,
    "accel_avg_window": 5,
    "accel_cutoff": 100,
    "min_accel": 0,
    "min_max_accel": 100,
    "min_speed_change": 0,
    "min_accel_dur": 0.02,
    "spine_analysis": 1,
    'alternate_turn': 0,
    'use_s0': 0,
    'correct_errors': 0,
    'correction_window': 0.05,
    'max_turn': 1,
    "turn_avg_window": 3,
    "turn_cutoff": 2,
    "min_turn_velocity": 0,
    "min_max_turn_velocity": 2,
    "min_turn_angle": 0.2,
    "min_turn_dur": 0.02,
    "bend_avg_window": 5,
    "bend_cutoff": 2,
    "min_bend_velocity": 0,
    "min_max_bend_velocity": 2,
    "min_bend_angle": 0.1,
    "min_bend_dur": 0.02,
    "min_amp": 2}
settings = load_settings('analysis', default_settings)

sampling = settings['sampling']
analyses = []
analyses_df = pd.DataFrame()
error2s_all = []
fdirs_all = pd.DataFrame()
angles_all = pd.DataFrame()
steps_all = pd.DataFrame()
first_video = True

metadata_all = pd.read_csv('metadata.csv')
metadata_track_all = []

for _, metadata in metadata_all.iterrows():
    
    filepath = metadata['filepath']
    p = Path(filepath)
    name = metadata['name']
    print(f'\nProcessing {name}')
    subpath = f'{p.parent}/{name}'
    group = metadata['group']
    l = metadata['video_end'] - metadata['video_start']
    fps = metadata['fps']
    if metadata['swimarea_x'] > metadata['swimarea_y']:
        ratio = settings['tank_x'] / metadata['swimarea_x']
    else:
        ratio = settings['tank_y'] / metadata['swimarea_y']
    
    cens = [[0, 0] for i in range(l)]
    with open(f'{subpath}/{name}_cen.csv', 'r') as f:
        cens_temp = [[cell for cell in row] for row in csv.reader(f)]
        cens_temp.pop(0)
        for i in range(l):
            cens[i][0] = float(cens_temp[i][0])
            cens[i][1] = float(cens_temp[i][1])
    
    cen_dists = [0 for i in range(l)] # obtain a list of speed at each frame
    speeds = [0 for i in range(l)]
    for i in range(sampling, l, sampling):
        cen_dists[i] = pyth((cens[i]), (cens[i - sampling])) / sampling * ratio
        speeds[i] = cen_dists[i] * fps
    for i in range(0, sampling):
        cen_dists[i] = cen_dists[sampling]
    for i in range(sampling * 2, l, sampling):
        for j in range(i - sampling + 1, i):
            cen_dists[j] = (cen_dists[i - sampling] * (i - j) + cen_dists[i] * (j - (i - sampling))) / sampling
            speeds[j] = cen_dists[j] * fps
    
    if settings['correct_errors']:
        error1_frames = []
        for i in range(l):
            if speeds[i] > settings['speed_limit']:
                error1_frames.append(i)
        cen_dists = errors_correct(cen_dists, error1_frames, l)
        speeds = errors_correct(speeds, error1_frames, l)
    
    total_distance = sum(cen_dists)
    total_time = l / fps
    speed_avg = total_distance / total_time
    
    freeze = [0 for i in range(l)] # determine whether the fish is freezing for each frame
    for i in range(round(fps * 3), l):
        prev3 = round(i - fps * 3)
        prev2 = round(i - fps * 2)
        prev1 = round(i - fps)
        cdist1 = pyth((cens[prev2][0], cens[prev2][1]), (cens[prev3][0], cens[prev3][1])) * ratio
        cdist2 = pyth((cens[prev1][0], cens[prev1][1]), (cens[prev2][0], cens[prev2][1])) * ratio
        cdist3 = pyth((cens[i][0], cens[i][1]), (cens[prev1][0], cens[prev1][1])) * ratio
        if cdist1 < 1 and cdist2 < 1 and cdist3 < 1:
            for j in range(prev3 + 1, i + 1):
                freeze[j] = 1
        elif cdist1 > 1 and cdist2 > 1 and cdist3 > 1:
            for j in range(prev2 + 1, i + 1):
                freeze[j] = 0
    
    total_freeze_time = sum(freeze) / fps
    freeze_percent = total_freeze_time / total_time * 100
    active_time = total_time - total_freeze_time
    active_speed = total_distance / active_time
    freeze_count = 0
    if freeze[1] == 1:
        freeze_count += 1
    for i in range(2, l):
        if freeze[i] - freeze[i - 1] == 1:
            freeze_count += 1
    freeze_freq = freeze_count * 60 / total_time
    
    # obtain a list of speeds with running average and obtain max speed
    speeds = curve(speeds, start=1, end=l, window=settings['accel_avg_window'])
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
    xl = metadata['swimarea_tlx']
    length = metadata['swimarea_x']
    xr = xl + length
    yt = metadata['swimarea_tly']
    width = metadata['swimarea_y']
    yb = yt + width
    d_from_wall = min(length, width) / 4
    for i in range(l):
        x = min(cens[i][0] - xl, xr - cens[i][0])
        y = min(cens[i][1] - yt, yb - cens[i][1])
        ds_from_wall[i] = min(x, y)
        if ds_from_wall[i] < d_from_wall:
            thigmotaxis_time += 1
    thigmotaxis_percent = thigmotaxis_time / fps / total_time
    
    analysis = {
        'name': name,
        'total_time': total_time,
        'total_distance': total_distance,
        'speed_avg': speed_avg,
        'max_speed': max_speed,
        'max_distance_1s': max_distance_1s,
        'active_speed': active_speed,
        'freeze_percent': freeze_percent,
        'freeze_count': freeze_count,
        'freeze_freq': freeze_freq,
        'thigmotaxis_percent': thigmotaxis_percent}
    
    if settings['plot_figure']:
        create_path('Tracks')
        fig, ax = plt.subplots()
        ax.scatter(x=[cens[i][0] for i in range(l)], y=[cens[i][1] for i in range(l)])
        fig.savefig(f'Tracks/{name}_track.png')
        plt.close()
    
    speeds.dt(fps)
    speeds.get_p_dflns(settings['accel_cutoff'], settings['min_accel'], settings['min_accel'],
                       settings['min_accel_dur'], settings['min_max_accel'], settings['min_speed_change'])
    
    if speeds.p_dflns_count == 0:
        print('No detectable movement')
        if settings['save_individually']:
            with open(f'{subpath}/{name}_analysis.csv', 'w') as f:
                for key in analysis:
                    f.write(key + ',' + str(analysis[key]) + '\n')
        analyses.append(analysis)
        print('Analysis of ' + name + ' complete.')
        with open(f'{subpath}/{name}_analysis_notes.csv', 'w') as f:
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
        speeds.graph_dflns(10, f'{subpath}/{name}_speed.png')
    
    if not settings['spine_analysis']:
        analyses.append(analysis)
        print('Analysis of ' + name + ' complete.')
        with open(f'{subpath}/{name}_analysis_notes.csv', 'w') as f:
            for key in settings:
                f.write(key + ', ' + str(settings[key]) + '\n')
        continue # step analysis when spine_analysis is disabled is not yet available
    
    # load midline points data
    with open(f'{subpath}/{name}_spine.csv', 'r') as f:
        spines_temp = [[cell for cell in row] for row in csv.reader(f)]
        spines_temp.pop(0)
        spine_len = len(spines_temp[0]) // 2
        spines = [[[0, 0] for j in range(spine_len)] for i in range(l)]
        for i in range(l):
            for j in range(spine_len):
                spines[i][j][0] = float(spines_temp[i][j * 2])
                spines[i][j][1] = float(spines_temp[i][j * 2 + 1])
    
    s0s = [[0, 0] for i in range(l)]
    if settings['use_s0']:
        with open(f'{subpath}/{name}_s0s.csv', 'r') as f:
            s0s_temp = [[cell for cell in row] for row in csv.reader(f)]
            s0s_temp.pop(0)
            for i in range(l):
                s0s[i][0] = float(s0s_temp[i][0])
                s0s[i][1] = float(s0s_temp[i][1])
    
    error2_pts = []
    dirs_nn = [0 for i in range(l)]
    for i in range(l):
        dirs_nn[i] = cal_direction(spines[i][spine_len - 2], spines[i][spine_len - 1])
    turns_nn = [0 for i in range(l)]
    for i in range(1, l):
        turns_nn[i] = cal_direction_change(dirs_nn[i - 1], dirs_nn[i])
    dirs_12 = [0 for i in range(l)]
    for i in range(l):
        dirs_12[i] = cal_direction(spines[i][0], spines[i][1])
    turns_12 = [0 for i in range(l)]
    for i in range(1, l):
        turns_12[i] = cal_direction_change(dirs_12[i - 1], dirs_12[i])
    for i in range(1, l):
        if abs(turns_nn[i]) > settings['max_turn'] or abs(turns_12[i]) > settings['max_turn']:
            error2_pts.append(i)
    
    error2_count = 0
    error2_pt_count = len(error2_pts)
    error2_intervals = []
    window = round(settings['correction_window'] * fps)
    i = 0
    while i < error2_pt_count - 1:
        if error2_pts[i + 1] - error2_pts[i] <= window:
            error2_intervals.append([error2_pts[i], error2_pts[i + 1]])
            error2_count += (error2_pts[i + 1] - error2_pts[i])
            i += 1
        i += 1
    print(f'{error2_count} frames with abnormal midline points detected at intervals {error2_intervals}')
    
    if settings['correct_errors']:
        
        correct_frames_path = f'{subpath}/Corrected_frames'
        create_path(correct_frames_path)
        video_n = cv.VideoCapture(f'{subpath}/{name}_n.avi')
        for interval in error2_intervals:
            i = interval[0] - 1
            j = interval[1]
            w = j - i
            for ii in range(1, w):
                for jj in range(spine_len):
                    spines[i + ii][jj][0] = (spines[i][jj][0] * (w - ii) + spines[j][jj][0] * ii) / w
                    spines[i + ii][jj][1] = (spines[i][jj][1] * (w - ii) + spines[j][jj][1] * ii) / w
                s0s[i + ii][0] = float(spines[i + ii][0][0])
                s0s[i + ii][1] = float(spines[i + ii][0][1])
                video_n.set(cv.CAP_PROP_POS_FRAMES, i + ii)
                ret, frame = video_n.read()
                if ret:
                    for jj in range(spine_len):
                        colorn = int(jj / spine_len * 255)
                        cv.circle(frame, (round(spines[i + ii][jj][0]), round(spines[i + ii][jj][1])), 3, (colorn, 255 - colorn // 2, 255 - colorn), -1)
                    cv.imwrite(f'{correct_frames_path}/{i + ii}.png', frame)
        video_n.release()
        
        error2_pts_new = []
        dirs_nn_new = [0 for i in range(l)]
        for i in range(l):
            dirs_nn_new[i] = cal_direction(spines[i][spine_len - 2], spines[i][spine_len - 1])
        turns_nn_new = [0 for i in range(l)]
        for i in range(1, l):
            turns_nn_new[i] = cal_direction_change(dirs_nn_new[i - 1], dirs_nn_new[i])
        dirs_12_new = [0 for i in range(l)]
        for i in range(l):
            dirs_12_new[i] = cal_direction(spines[i][0], spines[i][1])
        turns_12_new = [0 for i in range(l)]
        for i in range(1, l):
            turns_12_new[i] = cal_direction_change(dirs_12_new[i - 1], dirs_12_new[i])
        for i in range(1, l):
            if abs(turns_nn_new[i]) > settings['max_turn'] or abs(turns_12_new[i]) > settings['max_turn']:
                error2_pts_new.append(i)
        
        error2_count_new = 0
        error2_pt_count_new = len(error2_pts_new)
        error2_intervals_new = []
        i = 0
        while i < error2_pt_count_new - 1:
            if error2_pts_new[i + 1] - error2_pts_new[i] <= window:
                error2_intervals_new.append([error2_pts_new[i], error2_pts_new[i + 1]])
                error2_count_new += (error2_pts_new[i + 1] - error2_pts_new[i])
                i += 1
            i += 1
        print(f'{error2_count_new} frames remain at intervals {error2_intervals_new}')
        error2s_all.append({
            'name': name,
            'error2_count': error2_count,
            'error2_count_new': error2_count_new})
        
        with open(f'{subpath}/{name}_error2s.csv', 'w') as f:
            f.write('Frame,Turn_nn,Turn_nn_corrected,Turn_12,Turn_12_corrected\n')
            for i in range(l):
                f.write(f'{i},{turns_nn[i]},{turns_nn_new[i]},{turns_12[i]},{turns_12_new[i]}\n')
        
        with open(f'{subpath}/{name}_spine_corrected.csv', 'w') as f:
            for i in range(l):
                for j in range(spine_len):
                    f.write(f'{spines[i][j][0]},{spines[i][j][1]},')
                f.write('\n')
        
    else:
        
        error2s_all.append({
            'name': name,
            'error2_count': error2_count})
        with open(f'{subpath}/{name}_error2s.csv', 'w') as f:
            f.write('Frame,Turn_nn,Turn_12\n')
            for i in range(l):
                f.write(f'{i},{turns_nn[i]},{turns_12[i]}\n')
    
    directions = [0 for i in range(l)]
    for i in range(l):
        if settings['alternate_turn']:
            directions[i] = cal_direction(spines[i][round(spine_len * 2 / 3)], spines[i][spine_len - 1])
        else:
            directions[i] = cal_direction(spines[i][spine_len - 2], spines[i][spine_len - 1])
    turns = [0 for i in range(l)]
    for i in range(1, l):
        turns[i] = cal_direction_change(directions[i - 1], directions[i])
    
    spine_lens = [spine_len for i in range(l)]
    if settings['use_s0']:
        for i in range(l):
            if abs(s0s[i][0] - spines[i][0][0]) >= 0.1 and abs(s0s[i][1] - spines[i][0][1]) >= 0.1:
                spines[i].insert(0, [float(s0s[i][0]), float(s0s[i][1])])
                spine_lens[i] += 1
    
    heads = [0 for i in range(l)]
    with open(f'{subpath}/{name}_sn+1s.csv', 'r') as f:
        temp = [[cell for cell in row] for row in csv.reader(f)]
        temp.pop(0)
        for i in range(l):
            heads[i] = (float(temp[i][0]), float(temp[i][1]))
    
    fish_segs = [[0 for j in range(spine_lens[i])] for i in range(l)]
    for i in range(l):
        for j in range(spine_lens[i] - 1):
            fish_segs[i][j] = pyth(spines[i][j], spines[i][j + 1]) * ratio
        fish_segs[i][spine_lens[i] - 1] = pyth(spines[i][spine_lens[i] - 1], heads[i]) * ratio
    fish_lengths = [sum(fish_segs[i]) for i in range(l)]
    
    spine_angles = [[] for i in range(l)] # calculate bend angles
    angles = [0 for i in range(l)]
    for i in range(l):
        spine_dirs = [] # calculate direction from one midline point to another, caudal to cranial
        for j in range(1, spine_lens[i]):
            spine_dirs.append(cal_direction(spines[i][j - 1], spines[i][j]))
        for j in range(2, spine_lens[i]): # calculate bend angles. left is +, right is -
            spine_angles[i].append(cal_direction_change(spine_dirs[j - 1], spine_dirs[j - 2]))
            angles[i] += spine_angles[i][j - 2]
    
    bend_poss = [0 for i in range(l)]
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
                trunk_amps[i][j] = abs(m * spines[i][j][0] - spines[i][j][1] + c) / sqrt(m ** 2 + 1) * ratio
        amps[i] = trunk_amps[i][0]
    for i in range(l):
        for j in range(spine_lens[i] - 2):
            if trunk_amps[i][j] < settings['min_amp']:
                if j >= 1:
                    bend_poss[i] = sum(fish_segs[i][0:j]) / fish_lengths[i]
                break
    bend_poss = curve(bend_poss, 0, l, 0)
    
    fish_length_med = float(pd.DataFrame(fish_lengths).median().iloc[0])
    analysis.update({'fish_length': fish_length_med})
    
    if settings['correct_errors']:
        error_frames = []
        with open(f'{subpath}/{name}_errors.csv', 'r') as f:
            for row in csv.reader(f):
                for cell in row:
                    if cell.isnumeric():
                        error_frames.append(int(cell))
        angles = errors_correct(angles, error_frames, l)
    
    fdirs = [0 for i in range(l)] # fdirs is a list of special running average of direction of locomotion
    fdirs[0] = directions[0]
    for i in range(1, l):
        fdirs[i] = fdirs[i - 1] + turns[i]
    
    fdirs = curve(fdirs, start=0, end=l, window=settings['turn_avg_window'])
    fdirs.dt(fps) # turning left is -, turning right is +
    fdirs.get_p_dflns(settings['turn_cutoff'], settings['min_turn_velocity'], settings['min_turn_velocity'],
                      settings['min_turn_dur'], settings['min_max_turn_velocity'], settings['min_turn_angle'])
    fdirs.get_n_dflns(settings['turn_cutoff'], -settings['min_turn_velocity'], -settings['min_turn_velocity'],
                      settings['min_turn_dur'], settings['min_max_turn_velocity'], settings['min_turn_angle'])
    fdirs.merge_dflns()
    
    if settings['plot_figure']:
        fdirs.graph_dflns(max(fdirs.list) - min(fdirs.list), f'{subpath}/{name}_orient.png')
    
    for i in range(fdirs.dflns_count):
        fdirs.dflns[i].dict.update({
            'turn_angle': abs(fdirs.dflns[i].change),
            'turn_angular_velocity': abs(fdirs.dflns[i].maxslope),
            'turn_duration': fdirs.dflns[i].dur,
            'turn_laterality': 'left' if fdirs.dflns[i].change < 0 else 'right'})
        fdirs.dflns[i].env.update({'speed': cdist1s[round(fdirs.dflns[i].centralpos)]})
    
    angles = curve(angles, start=0, end=l, window=settings['bend_avg_window'])
    angles.dt(fps) # turning left is +, turning right is -
    angles.get_p_dflns(settings['bend_cutoff'], settings['min_bend_velocity'], settings['min_bend_velocity'],
                       settings['min_bend_dur'], settings['min_max_bend_velocity'], settings['min_bend_angle'])
    angles.get_n_dflns(settings['bend_cutoff'], -settings['min_bend_velocity'], -settings['min_bend_velocity'],
                       settings['min_bend_dur'], settings['min_max_bend_velocity'], settings['min_bend_angle'])
    angles.merge_dflns()
    
    if settings['plot_figure']:
        angles.graph_dflns(max(angles.list) - min(angles.list), f'{subpath}/{name}_angle.png')
    
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
    if settings['analysis_extended']:
        agg1 = ['sum', 'mean', 'std', 'median', 'p5', 'p95', 'ipr']
        agg2 = ['mean', 'std', 'median', 'p5', 'p95', 'ipr']
        agg3 = ['median', 'p5', 'p95', 'ipr']
    else:
        agg1 = ['sum', 'mean']
        agg2 = ['mean']
        agg3 = ['median']
    
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
        fdirs_df[i].update({'centralpos': fdirs.dflns[i].centralpos})
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
    analysis_df.add(fdirs_DF.stratify2('turn left', 'turn right', 'speed'))
    
    total_turn_angle = analysis_df.df.loc[(analysis_df.df['Type'] == 'turn') &
                                          analysis_df.df['Classify'].isna() &
                                          (analysis_df.df['Parameter'] == 'turn_angle') &
                                          (analysis_df.df['Method'] == 'sum'), 'Value']
    analysis.update({'meandering': total_turn_angle.iloc[0] / total_distance})
    
    angles_df = [d.dict for d in angles.dflns]
    for i in range(angles.dflns_count):
        angles_df[i].update(angles.dflns[i].env)
        angles_df[i].update({'centralpos': angles.dflns[i].centralpos})
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
    analysis_df.add(angles_DF.stratify2('bend left', 'bend right', 'speed'))
    
    recoils_df = angles_df[angles_df['recoil'] == True]
    recoils_DF = DF(recoils_df, 'recoil', angles_methods.keys())
    recoils_DF.dfs.update({
        'bend left': recoils_df[recoils_df['bend_laterality'] == 'left'],
        'bend right': recoils_df[recoils_df['bend_laterality'] == 'right']})
    analysis_df.add(recoils_DF.agg(angles_methods))
    analysis_df.add(recoils_DF.agg(angles_methods, 'bend left'))
    analysis_df.add(recoils_DF.agg(angles_methods, 'bend right'))
    analysis_df.add(recoils_DF.stratify1('speed'))
    analysis_df.add(recoils_DF.stratify2('bend left', 'bend right', 'speed'))
    
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
        
        if fdirs.dflns[i].belong == -1:
        
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
        elif len(steps[i].turns) >= 1:
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
        fig.set_size_inches(l / fps, 10)
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
        fig.savefig(f'{subpath}/{name}_steps.png')
        plt.close()
        
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
            
            if abs(turn_angle_overall) >= settings['min_turn_angle']:
                if turn_angle_overall > settings['min_turn_angle']:
                    s.properties.update({'turn_laterality': 'right'})
                elif turn_angle_overall < -settings['min_turn_angle']:
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
        s.properties.update({'velocity_change': sqrt(a ** 2 + b ** 2 - 2 * a * b * cos(angle))})
        
        if s.bends_count == 0:
            
            continue
        
        elif s.bends_count == 1:
            
            s.properties.update({
                'bend_angle_reached': max(abs(s.bends[0].dict['angle start']), abs(s.bends[0].dict['angle end'])),
                'bend_angle_traveled': s.bends[0].dict['angle_change'],
                'bend_dur_total': s.bends[0].dict['bend_dur'],
                'bend_angular_velocity': s.bends[0].dict['bend_angular_velocity'],
                'bend_pos': s.bends[0].dict['bend_pos']})
        
        elif s.bends_count == 2:
            
            s.properties.update({
                'bend_angle_reached': abs(s.bends[0].dict['angle end']),
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
                'bend_angle_traveled': sum(angles_traveled),
                'bend_dur_total': sum(durs),
                'bend_angular_velocity': max(angular_velocitys),
                'bend_pos': max(bend_pos),
                'max angle pos': max_angle_pos})
            
        s.properties.update({'bend_wave_freq': s.bends_count / s.properties['bend_dur_total']})
    
    steps_df = []
    for s in steps:
        temp = s.properties
        temp.update({'centralpos': s.centralpos})
        steps_df.append(temp)
    steps_df = pd.DataFrame(steps_df)
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
        'without turn': steps_df[steps_df['turn_laterality'] == 'neutral'],})
    
    analysis_df.add(steps_DF.agg(steps_methods))
    analysis_df.add(steps_DF.agg(steps_methods, 'turn left'))
    analysis_df.add(steps_DF.agg(steps_methods, 'turn right'))
    analysis_df.add(steps_DF.agg(steps_methods, 'with turn'))
    analysis_df.add(steps_DF.agg(steps_methods, 'without turn'))
    
    for i in steps_methods:
        analysis_df.add(steps_DF.stratify1(i))
    for i in steps_methods:
        analysis_df.add(steps_DF.stratify2('turn left', 'turn right', i))
        analysis_df.add(steps_DF.stratify2('with turn', 'without turn', i))
    
    fdirs_df['name'] = name
    fdirs_all = pd.concat([fdirs_all, fdirs_df])
    angles_df['name'] = name
    angles_all = pd.concat([angles_all, angles_df])
    steps_df['name'] = name
    steps_all = pd.concat([steps_all, steps_df])
    
    print(f'Analysis of {name} complete.')
    analyses.append(analysis)
    
    if first_video:
        analyses_df = analysis_df.df[['Type', 'Classify', 'Stratify', 'Parameter', 'Method']]
        first_video = False
    analyses_df[name] = analysis_df.df['Value']
    
    with open(f'{subpath}/{name}_analysis_notes.csv', 'w') as f:
        for key in settings:
            f.write(key + ', ' + str(settings[key]) + '\n')

fdirs_all.to_csv('turns_all.csv', index=False)
angles_all.to_csv('bends_all.csv', index=False)
steps_all.to_csv('steps_all.csv', index=False)

names = metadata_all['name']
analyses = pd.DataFrame(analyses).T
analyses = analyses.drop('name')
analyses.columns = names
analyses.to_csv('analyses.csv')

with open('error2s.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=error2s_all[0].keys())
    writer.writeheader()
    writer.writerows(error2s_all)

analyses_df.to_csv('analyses_df.csv', index=False)
if not settings['spine_analysis']:
    from sys import exit
    exit()

steps_all_adjusted = pd.DataFrame(steps_all)
analyses_df_adjusted = pd.DataFrame(analyses_df)
adjusts = ['current_speed', 'step_length', 'speed_change', 'velocity_change', 'accel']
for name in names:
    fish_length = analyses.at['fish_length', name]
    steps_all_adjusted.loc[steps_all_adjusted['name'] == name, adjusts] = steps_all_adjusted.loc[steps_all_adjusted['name'] == name, adjusts].transform(lambda a: a / fish_length)
    analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), name] = analyses_df_adjusted.loc[analyses_df['Parameter'].isin(adjusts), name].transform(lambda a: a / fish_length)
steps_all_adjusted.to_csv('steps_all_adjusted.csv', index=False)
analyses_df_adjusted.to_csv('analyses_df_adjusted.csv', index=False)
print('All analyses complete.')

with pd.ExcelWriter('turns_properties.xlsx') as writer:
    for i in fdirs_all.columns:
        if i == 'name':
            continue
        p = pd.DataFrame()
        for name in names:
            p[name] = fdirs_all[fdirs_all['name'] == name][i]
        p.to_excel(writer, sheet_name=i, index=False)

with pd.ExcelWriter('bends_properties.xlsx') as writer:
    for i in angles_all.columns:
        if i == 'name':
            continue
        p = pd.DataFrame()
        for name in names:
            p[name] = angles_all[angles_all['name'] == name][i]
        p.to_excel(writer, sheet_name=i, index=False)

with pd.ExcelWriter('steps_properties.xlsx') as writer:
    for i in steps_all_adjusted.columns:
        if i == 'name':
            continue
        p = pd.DataFrame()
        for name in names:
            p[name] = steps_all_adjusted[steps_all_adjusted['name'] == name][i]
        p.to_excel(writer, sheet_name=i, index=False)
