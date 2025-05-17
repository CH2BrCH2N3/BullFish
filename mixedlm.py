import pandas as pd
import statsmodels.formula.api as smf

params = [
    'step_length',
    'speed_change',
    'velocity_change',
    'accel',
    'step_dur',
    'coast_dur',
    'coast_percent',
    'turn_angle',
    'turn_dur',
    'turn_angular_velocity',
    'bend_angle_reached',
    'bend_pos',
    'bend_angle_traveled',
    'bend_angular_velocity',
    'bend_dur_total',
    'bend_wave_frequency']

steps_all_adjusted = pd.read_csv('steps_all_adjusted.csv')
steps_all_adjusted['Group'] = steps_all_adjusted['video'].apply(lambda a: a[13:16])

with open('Results.txt', 'w') as f:
    for param in params:
        data = steps_all_adjusted[['Group', 'video', param]]
        model = smf.mixedlm(f'{param} ~ Group', data, groups=data['video'])
        result = model.fit()
        f.write(str(result.summary()))