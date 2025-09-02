import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('analyses_df_adjusted.csv')
t = '6OH'
c = 'veh'

Dict = df.to_dict()
sw_t = {}
sw_c = {}
mwp = {}
l = len(df)
for i in range(l):
    st = []
    sc = []
    for name in df.columns:
        if t in name:
            st.append(Dict[name][i])
        if c in name:
            sc.append(Dict[name][i])
    st = pd.Series(st)
    sc = pd.Series(sc)
    sw_t.update({i: scipy.stats.shapiro(st).pvalue})
    sw_c.update({i: scipy.stats.shapiro(sc).pvalue})
    mwp.update({i: scipy.stats.mannwhitneyu(st, sc, method='exact').pvalue})
Dict.update({
    f'shapiro_{t}': sw_t,
    f'shapiro_{c}': sw_c,
    'MWp': mwp})
df_new = pd.DataFrame(Dict)
df_new.to_csv('analyses_df_adjusted.csv', index=False)

def t_test(df, Type):
    dfs = df[(df['Type'] == Type) & (pd.isna(df['Classify'])) & (pd.isna(df['Stratify']))]
    results = []
    for i in range(len(dfs)):
        s = dfs.iloc[i]
        st = []
        sc = []
        for name in dfs.columns:
            if t in name:
                st.append(s[name])
            if c in name:
                sc.append(s[name])
        st = pd.Series(st)
        sc = pd.Series(sc)
        test = scipy.stats.ttest_ind(st, sc, equal_var=False)
        ci = test.confidence_interval()
        _, mwp = scipy.stats.mannwhitneyu(st, sc, method='exact')
        result = {
            'Type': Type,
            'Parameter': s['Parameter'],
            'Method': s['Method'],
            t + '_mean': st.mean(),
            c + '_mean': sc.mean(),
            't': test.statistic,
            'low t': ci[0],
            'hi t': ci[1],
            'p': test.pvalue,
            'MWp': mwp}
        results.append(result)
    return pd.DataFrame(results)

t_test_results = pd.DataFrame()
t_test_results = pd.concat([t_test_results, t_test(df, 'turn')])
t_test_results = pd.concat([t_test_results, t_test(df, 'bend')])
t_test_results = pd.concat([t_test_results, t_test(df, 'recoil')])
t_test_results = pd.concat([t_test_results, t_test(df, 'step')])
t_test_results.to_csv('t_test_results.csv', index=False)

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
        result[t + '_a'] = data.loc[(data['Treatment'] == t) & (data['Group'] == a), 'Value'].mean()
        result[t + '_b'] = data.loc[(data['Treatment'] == t) & (data['Group'] == b), 'Value'].mean()
        result[c + '_a'] = data.loc[(data['Treatment'] == c) & (data['Group'] == a), 'Value'].mean()
        result[c + '_b'] = data.loc[(data['Treatment'] == c) & (data['Group'] == b), 'Value'].mean()
        result['Type'] = Type
        result['Parameter'] = parameter
        result['Method'] = method
        result['Group'] = a + ' vs ' + b
        results = pd.concat([results, pd.DataFrame(result).T])
    return results

compare_results = pd.DataFrame()
compare_results = pd.concat([compare_results, compare(df, 'turn', 'turn left', 'turn right')])
compare_results = pd.concat([compare_results, compare(df, 'bend', 'bend left', 'bend right')])
compare_results = pd.concat([compare_results, compare(df, 'recoil', 'bend left', 'bend right')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'turn left', 'turn right')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'with turn', 'without turn')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'bend left', 'bend right')])
compare_results.to_csv('compare_results.csv', index=False)

def stratify(df):
    dfs = df[(pd.isna(df['Classify'])) & (pd.notna(df['Stratify']))]
    results = pd.DataFrame()
    for i in range(len(dfs)):
        Sl = dfs.iloc[i]
        parameter = Sl['Parameter']
        split = Sl['Stratify'].split(sep='_')
        if split[len(split) - 1] != 'low':
            continue
        strat = Sl['Stratify'].split(sep='_low')[0]
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
        result[t + '_l'] = data.loc[(data['Treatment'] == t) & (data['Stratify'] == Sl['Stratify']), 'Value'].mean()
        result[t + '_m'] = data.loc[(data['Treatment'] == t) & (data['Stratify'] == Sm['Stratify'].iloc[0]), 'Value'].mean()
        result[t + '_h'] = data.loc[(data['Treatment'] == t) & (data['Stratify'] == Sh['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_l'] = data.loc[(data['Treatment'] == c) & (data['Stratify'] == Sl['Stratify']), 'Value'].mean()
        result[c + '_m'] = data.loc[(data['Treatment'] == c) & (data['Stratify'] == Sm['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_h'] = data.loc[(data['Treatment'] == c) & (data['Stratify'] == Sh['Stratify'].iloc[0]), 'Value'].mean()
        result['Parameter'] = parameter
        result['Stratify'] = strat
        results = pd.concat([results, pd.DataFrame(result).T])
    return results

stratify_results = pd.DataFrame()
stratify_results = pd.concat([stratify_results, stratify(df)])
stratify_results.to_csv('stratify_results.csv', index=False)

def threeway(df, a, b):
    dfa = df[(df['Classify'] == a) & (pd.notna(df['Stratify']))]
    dfb = df[(df['Classify'] == b) & (pd.notna(df['Stratify']))]
    results = pd.DataFrame()
    for i in range(len(dfa)):
        sal = dfa.iloc[i]
        split = sal['Stratify'].split(sep='_')
        if split[len(split) - 1] != 'low':
            continue
        strat = sal['Stratify'].split(sep='_low')[0]
        if sal['Method'] == 'count':
            parameter = 'count'
            sam = dfa.loc[(dfa['Stratify'] == strat + '_mid') & (dfa['Method'] == 'count')]
            sah = dfa.loc[(dfa['Stratify'] == strat + '_high') & (dfa['Method'] == 'count')]
            sbl = dfb.loc[(dfb['Stratify'] == strat + '_low') & (dfb['Method'] == 'count')]
            sbm = dfb.loc[(dfb['Stratify'] == strat + '_mid') & (dfb['Method'] == 'count')]
            sbh = dfb.loc[(dfb['Stratify'] == strat + '_high') & (dfb['Method'] == 'count')]
        else:
            parameter = sal['Parameter']
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
        result[t + '_a_l'] = data.loc[(data['Treatment'] == t) & (data['Group'] == a) & (data['Stratify'] == sal['Stratify']), 'Value'].mean()
        result[t + '_a_m'] = data.loc[(data['Treatment'] == t) & (data['Group'] == a) & (data['Stratify'] == sam['Stratify'].iloc[0]), 'Value'].mean()
        result[t + '_a_h'] = data.loc[(data['Treatment'] == t) & (data['Group'] == a) & (data['Stratify'] == sah['Stratify'].iloc[0]), 'Value'].mean()
        result[t + '_b_l'] = data.loc[(data['Treatment'] == t) & (data['Group'] == b) & (data['Stratify'] == sbl['Stratify'].iloc[0]), 'Value'].mean()
        result[t + '_b_m'] = data.loc[(data['Treatment'] == t) & (data['Group'] == b) & (data['Stratify'] == sbm['Stratify'].iloc[0]), 'Value'].mean()
        result[t + '_b_h'] = data.loc[(data['Treatment'] == t) & (data['Group'] == b) & (data['Stratify'] == sbh['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_a_l'] = data.loc[(data['Treatment'] == c) & (data['Group'] == a) & (data['Stratify'] == sal['Stratify']), 'Value'].mean()
        result[c + '_a_m'] = data.loc[(data['Treatment'] == c) & (data['Group'] == a) & (data['Stratify'] == sam['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_a_h'] = data.loc[(data['Treatment'] == c) & (data['Group'] == a) & (data['Stratify'] == sah['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_b_l'] = data.loc[(data['Treatment'] == c) & (data['Group'] == b) & (data['Stratify'] == sbl['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_b_m'] = data.loc[(data['Treatment'] == c) & (data['Group'] == b) & (data['Stratify'] == sbm['Stratify'].iloc[0]), 'Value'].mean()
        result[c + '_b_h'] = data.loc[(data['Treatment'] == c) & (data['Group'] == b) & (data['Stratify'] == sbh['Stratify'].iloc[0]), 'Value'].mean()
        result['Parameter'] = parameter
        result['Group'] = a + ' vs ' + b
        result['Stratify'] = strat
        results = pd.concat([results, pd.DataFrame(result).T])
    return results

threeway_results = pd.DataFrame()
threeway_results = pd.concat([threeway_results, threeway(df, 'turn left', 'turn right')])
threeway_results = pd.concat([threeway_results, threeway(df, 'with turn', 'without turn')])
threeway_results = pd.concat([threeway_results, threeway(df, 'bend left', 'bend right')])
threeway_results.to_csv('threeway_results.csv', index=False)
