import pandas as pd
import statsmodels.formula.api as smf
from BullFish_pkg.general import create_path

params = [
    'turn_angle',
    'turn_angular_velocity',
    'turn_duration']

turns_all = pd.read_csv('turns_all.csv')
turns_all['Treatment'] = turns_all['video'].apply(lambda a: a[0:3])

with open('Turns_Results.txt', 'w') as f:
    for param in params:
        model = smf.mixedlm(f'{param} ~ Treatment', turns_all, groups=turns_all['video'])
        result = model.fit()
        f.write(str(result.summary()))

videos = turns_all['video'].unique()
p = []
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
for video in videos:
    turns = turns_all[turns_all['video'] == video].copy()
    turns = turns.sort_values(by=['speed'])
    l = len(turns)
    df1 = pd.concat([df1, turns.iloc[:round(l / 4)]])
    df2 = pd.concat([df2, turns.iloc[round(l / 4):round(l * 3 / 4)]])
    df3 = pd.concat([df3, turns.iloc[round(l * 3 / 4):l]])
for param in params:
    model = smf.mixedlm(f'{param} ~ Treatment', df1, groups=df1['video'])
    result = model.fit()
    p.append({
        'Param': param,
        'Stratify': 'low speed',
        'p': result.pvalues.iloc[1],
        'Control mean': result.params.iloc[0],
        'Difference from control': result.params.iloc[1]})
    model = smf.mixedlm(f'{param} ~ Treatment', df2, groups=df2['video'])
    result = model.fit()
    p.append({
        'Param': param,
        'Stratify': 'mid speed',
        'p': result.pvalues.iloc[1],
        'Control mean': result.params.iloc[0],
        'Difference from control': result.params.iloc[1]})
    model = smf.mixedlm(f'{param} ~ Treatment', df3, groups=df3['video'])
    result = model.fit()
    p.append({
        'Param': param,
        'Stratify': 'high speed',
        'p': result.pvalues.iloc[1],
        'Control mean': result.params.iloc[0],
        'Difference from control': result.params.iloc[1]})
p = pd.DataFrame(p)
p.to_csv('Turns_Stratified_mixedlm.csv', index=False)

params = [
    'angle_change',
    'bend_dur',
    'bend_angular_velocity',
    'bend_pos']

bends_all = pd.read_csv('bends_all.csv')
bends_all['Treatment'] = bends_all['video'].apply(lambda a: a[0:3])

with open('bends_Results.txt', 'w') as f:
    for param in params:
        model = smf.mixedlm(f'{param} ~ Treatment', bends_all, groups=bends_all['video'])
        result = model.fit()
        f.write(str(result.summary()))

videos = bends_all['video'].unique()
p = []
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
for video in videos:
    bends = bends_all[bends_all['video'] == video].copy()
    bends = bends.sort_values(by=['speed'])
    l = len(bends)
    df1 = pd.concat([df1, bends.iloc[:round(l / 4)]])
    df2 = pd.concat([df2, bends.iloc[round(l / 4):round(l * 3 / 4)]])
    df3 = pd.concat([df3, bends.iloc[round(l * 3 / 4):l]])
for param in params:
    model = smf.mixedlm(f'{param} ~ Treatment', df1, groups=df1['video'])
    result = model.fit()
    p.append({
        'Param': param,
        'Stratify': 'low speed',
        'p': result.pvalues.iloc[1],
        'Control mean': result.params.iloc[0],
        'Difference from control': result.params.iloc[1]})
    model = smf.mixedlm(f'{param} ~ Treatment', df2, groups=df2['video'])
    result = model.fit()
    p.append({
        'Param': param,
        'Stratify': 'mid speed',
        'p': result.pvalues.iloc[1],
        'Control mean': result.params.iloc[0],
        'Difference from control': result.params.iloc[1]})
    model = smf.mixedlm(f'{param} ~ Treatment', df3, groups=df3['video'])
    result = model.fit()
    p.append({
        'Param': param,
        'Stratify': 'high speed',
        'p': result.pvalues.iloc[1],
        'Control mean': result.params.iloc[0],
        'Difference from control': result.params.iloc[1]})
p = pd.DataFrame(p)
p.to_csv('bends_Stratified_mixedlm.csv', index=False)

params = [
    'current_speed',
    'step_length',
    'speed_change',
    'velocity_change',
    'accel',
    'step_dur',
    'coast_dur',
    'coast_percent',
    'current_step_s',
    'current_bend_s',
    'turn_angle',
    'turn_dur',
    'turn_angular_velocity',
    'bend_count',
    'bend_angle_reached',
    'bend_pos',
    'bend_angle_traveled',
    'bend_angular_velocity',
    'bend_dur_total',
    'bend_wave_freq']

steps_all = pd.read_csv('steps_all_adjusted.csv')
steps_all['Treatment'] = steps_all['video'].apply(lambda a: a[0:3])

with open('Results.txt', 'w') as f:
    for param in params:
        model = smf.mixedlm(f'{param} ~ Treatment', steps_all, groups=steps_all['video'])
        result = model.fit()
        f.write(str(result.summary()))

videos = steps_all['video'].unique()
create_path('Stratified')
p = []
with open('Results_stratified.txt', 'w') as f:
    for i in params:
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df3 = pd.DataFrame()
        for video in videos:
            steps = steps_all[steps_all['video'] == video].copy()
            steps = steps.sort_values(by=[i])
            l = len(steps)
            df1 = pd.concat([df1, steps.iloc[:round(l / 4)]])
            df2 = pd.concat([df2, steps.iloc[round(l / 4):round(l * 3 / 4)]])
            df3 = pd.concat([df3, steps.iloc[round(l * 3 / 4):l]])
        df1.to_csv(f'Stratified/{i} low.csv', index=False)
        df2.to_csv(f'Stratified/{i} mid.csv', index=False)
        df3.to_csv(f'Stratified/{i} high.csv', index=False)
        for param in params:
            if i == param:
                continue
            model = smf.mixedlm(f'{param} ~ Treatment', df1, groups=df1['video'])
            result = model.fit()
            f.write(f'\n{param}, low {i}\n')
            f.write(str(result.summary()))
            p.append({
                'Param': param,
                'Stratify': f'low {i}',
                'p': result.pvalues.iloc[1],
                'Control mean': result.params.iloc[0],
                'Difference from control': result.params.iloc[1]})
            model = smf.mixedlm(f'{param} ~ Treatment', df2, groups=df2['video'])
            result = model.fit()
            f.write(f'\n{param}, mid {i}\n')
            f.write(str(result.summary()))
            p.append({
                'Param': param,
                'Stratify': f'mid {i}',
                'p': result.pvalues.iloc[1],
                'Control mean': result.params.iloc[0],
                'Difference from control': result.params.iloc[1]})
            model = smf.mixedlm(f'{param} ~ Treatment', df3, groups=df3['video'])
            result = model.fit()
            f.write(f'\n{param}, high {i}\n')
            f.write(str(result.summary()))
            p.append({
                'Param': param,
                'Stratify': f'high {i}',
                'p': result.pvalues.iloc[1],
                'Control mean': result.params.iloc[0],
                'Difference from control': result.params.iloc[1]})
p = pd.DataFrame(p)
p.to_csv('mixedlm.csv', index=False)
