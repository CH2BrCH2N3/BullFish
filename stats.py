# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:52:22 2024

@author: Sunny
"""

import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from numpy import nan

df = pd.read_csv('analyses_df_adjusted.csv')
t = '6OH'
c = 'veh'

def compare(df, Type, a, b):
    dfa = df[(df['Type'] == Type) & (df['Classify'] == a) & (pd.isna(df['Stratify']))]
    dfb = df[(df['Type'] == Type) & (df['Classify'] == b) & (pd.isna(df['Stratify']))]
    results = pd.DataFrame()
    for i in range(len(dfa)):
        sa = dfa.iloc[i]
        parameter = sa['Parameter']
        method = sa['Method']
        if method == 'count':
            sb = dfb.loc[dfb['Method'] == 'count']
        else:
            sb = dfb.loc[(dfb['Parameter'] == parameter) & (dfb['Method'] == method)]
        data = []
        for name in dfa.columns:
            if t in name:
                data.append({'Value': sa[name],
                             'Treatment': t,
                             'Group': a})
                data.append({'Value': sb[name].iloc[0],
                             'Treatment': t,
                             'Group': b})
            if c in name:
                data.append({'Value': sa[name],
                             'Treatment': c,
                             'Group': a})
                data.append({'Value': sb[name].iloc[0],
                             'Treatment': c,
                             'Group': b})
        data = pd.DataFrame(data)
        if data.isnull().values.any():
            continue
        model = ols('Value ~ C(Treatment) * C(Group)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        result = anova_table['PR(>F)'].T
        result.pop('Residual')
        result['Type'] = Type
        result['Parameter'] = parameter
        result['Method'] = method
        result['Group'] = a + ' vs ' + b
        results = pd.concat([results, pd.DataFrame(result).T])
    return results

compare_results = pd.DataFrame()
compare_results = pd.concat([compare_results, compare(df, 'bend', 'bend left', 'bend right')])
compare_results = pd.concat([compare_results, compare(df, 'recoil', 'bend left', 'bend right')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'HT', 'MT')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'turn left', 'turn right')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'with turn', 'without turn')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'bend left', 'bend right')])
compare_results.to_csv('compare_results.csv')

def stratify(df):
    dfs = df[(pd.isna(df['Classify'])) & (pd.notna(df['Stratify']))]
    results = pd.DataFrame()
    for i in range(len(dfs)):
        Sl = dfs.iloc[i]
        parameter = Sl['Parameter']
        if Sl['Stratify'].split(sep='_')[1] != 'low':
            continue
        strat = Sl['Stratify'].split(sep='_')[0]
        Sm = dfs.loc[(dfs['Stratify'] == strat + '_mid') & (dfs['Parameter'] == parameter)]
        Sh = dfs.loc[(dfs['Stratify'] == strat + '_high') & (dfs['Parameter'] == parameter)]
        data = []
        for name in dfs.columns:
            if t in name:
                data.append({'Value': Sl[name],
                             'Treatment': t,
                             'Stratify': Sl['Stratify']})
                data.append({'Value': Sm[name].iloc[0],
                             'Treatment': t,
                             'Stratify': Sm['Stratify'].iloc[0]})
                data.append({'Value': Sh[name].iloc[0],
                             'Treatment': t,
                             'Stratify': Sh['Stratify'].iloc[0]})
            if c in name:
                data.append({'Value': Sl[name],
                             'Treatment': c,
                             'Stratify': Sl['Stratify']})
                data.append({'Value': Sm[name].iloc[0],
                             'Treatment': c,
                             'Stratify': Sm['Stratify'].iloc[0]})
                data.append({'Value': Sh[name].iloc[0],
                             'Treatment': c,
                             'Stratify': Sh['Stratify'].iloc[0]})
        data = pd.DataFrame(data)
        if data.isnull().values.any():
            continue
        model = ols('Value ~ C(Treatment) * C(Stratify)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        result = anova_table['PR(>F)'].T
        result.pop('Residual')
        result['Parameter'] = parameter
        result['Stratify'] = strat
        results = pd.concat([results, pd.DataFrame(result).T])
    return results

stratify_results = pd.DataFrame()
stratify_results = pd.concat([stratify_results, stratify(df)])
stratify_results.to_csv('stratify_results.csv')

def threeway(df, a, b):
    dfa = df[(df['Classify'] == a) & (pd.notna(df['Stratify'])) & (df['Method'] != 'count')]
    dfb = df[(df['Classify'] == b) & (pd.notna(df['Stratify'])) & (df['Method'] != 'count')]
    results = pd.DataFrame()
    for i in range(len(dfa)):
        sal = dfa.iloc[i]
        parameter = sal['Parameter']
        if sal['Stratify'].split(sep='_')[1] != 'low':
            continue
        strat = sal['Stratify'].split(sep='_')[0]
        sam = dfa.loc[(dfa['Stratify'] == strat + '_mid') & (dfa['Parameter'] == parameter)]
        sah = dfa.loc[(dfa['Stratify'] == strat + '_high') & (dfa['Parameter'] == parameter)]
        sbl = dfb.loc[(dfb['Stratify'] == strat + '_low') & (dfb['Parameter'] == parameter)]
        sbm = dfb.loc[(dfb['Stratify'] == strat + '_mid') & (dfb['Parameter'] == parameter)]
        sbh = dfb.loc[(dfb['Stratify'] == strat + '_high') & (dfb['Parameter'] == parameter)]
        data = []
        for name in dfa.columns:
            if t in name:
                data.append({'Value': sal[name],
                             'Treatment': t,
                             'Group': a,
                             'Stratify': sal['Stratify']})
                data.append({'Value': sam[name].iloc[0],
                             'Treatment': t,
                             'Group': a,
                             'Stratify': sam['Stratify'].iloc[0]})
                data.append({'Value': sah[name].iloc[0],
                             'Treatment': t,
                             'Group': a,
                             'Stratify': sah['Stratify'].iloc[0]})
                data.append({'Value': sbl[name].iloc[0],
                             'Treatment': t,
                             'Group': b,
                             'Stratify': sbl['Stratify'].iloc[0]})
                data.append({'Value': sbm[name].iloc[0],
                             'Treatment': t,
                             'Group': b,
                             'Stratify': sbm['Stratify'].iloc[0]})
                data.append({'Value': sbh[name].iloc[0],
                             'Treatment': t,
                             'Group': b,
                             'Stratify': sbh['Stratify'].iloc[0]})
            if c in name:
                data.append({'Value': sal[name],
                             'Treatment': c,
                             'Group': a,
                             'Stratify': sal['Stratify']})
                data.append({'Value': sam[name].iloc[0],
                             'Treatment': c,
                             'Group': a,
                             'Stratify': sam['Stratify'].iloc[0]})
                data.append({'Value': sah[name].iloc[0],
                             'Treatment': c,
                             'Group': a,
                             'Stratify': sah['Stratify'].iloc[0]})
                data.append({'Value': sbl[name].iloc[0],
                             'Treatment': c,
                             'Group': b,
                             'Stratify': sbl['Stratify'].iloc[0]})
                data.append({'Value': sbm[name].iloc[0],
                             'Treatment': c,
                             'Group': b,
                             'Stratify': sbm['Stratify'].iloc[0]})
                data.append({'Value': sbh[name].iloc[0],
                             'Treatment': c,
                             'Group': b,
                             'Stratify': sbh['Stratify'].iloc[0]})
        data = pd.DataFrame(data)
        if data.isnull().values.any():
            continue
        model = ols('Value ~ C(Treatment) * C(Group) * C(Stratify)', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)
        result = anova_table['PR(>F)'].T
        result.pop('Residual')
        result['Parameter'] = parameter
        result['Group'] = a + ' vs ' + b
        result['Stratify'] = strat
        results = pd.concat([results, pd.DataFrame(result).T])
    return results

threeway_results = pd.DataFrame()
threeway_results = pd.concat([threeway_results, threeway(df, 'turn left', 'turn right')])
threeway_results = pd.concat([threeway_results, threeway(df, 'with turn', 'without turn')])
threeway_results = pd.concat([threeway_results, threeway(df, 'bend left', 'bend right')])
threeway_results.to_csv('threeway_results.csv')