import pandas as pd
import statsmodels.formula.api as smf
from BullFish_pkg.general import create_path

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
    'bend_angle_reached',
    'bend_pos',
    'bend_angle_traveled',
    'bend_angular_velocity',
    'bend_dur_total',
    'bend_frequency']

steps_all = pd.read_csv('steps_all_adjusted.csv')
steps_all['Treatment'] = steps_all['video'].apply(lambda a: a[13:16])

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
            steps = steps_all[steps_all['video'] == video]
            intervals = steps.describe()
            df1 = pd.concat([df1, steps[steps[i] <= intervals.loc['25%'][i]]])
            df2 = pd.concat([df2, steps[(steps[i] > intervals.loc['25%'][i]) &
                                        (steps[i] < intervals.loc['75%'][i])]])
            df3 = pd.concat([df3, steps[steps[i] >= intervals.loc['75%'][i]]])
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
                'p': result.pvalues.iloc[1]})
            model = smf.mixedlm(f'{param} ~ Treatment', df2, groups=df2['video'])
            result = model.fit()
            f.write(f'\n{param}, mid {i}\n')
            f.write(str(result.summary()))
            p.append({
                'Param': param,
                'Stratify': f'mid {i}',
                'p': result.pvalues.iloc[1]})
            model = smf.mixedlm(f'{param} ~ Treatment', df3, groups=df3['video'])
            result = model.fit()
            f.write(f'\n{param}, high {i}\n')
            f.write(str(result.summary()))
            p.append({
                'Param': param,
                'Stratify': f'high {i}',
                'p': result.pvalues.iloc[1]})
p = pd.DataFrame(p)
p.to_csv('mixedlm.csv', index=False)
