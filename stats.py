import pandas as pd
import scipy.stats

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
df_new.to_csv('analyses_df_adjusted_stats.csv', index=False)

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

def ai(a, b):
    return (a - b) / (a + b)

def compare(df, Type, a, b):
    
    dfa = df[(df['Type'] == Type) & (df['Classify'] == a) & (pd.isna(df['Stratify']))]
    dfb = df[(df['Type'] == Type) & (df['Classify'] == b) & (pd.isna(df['Stratify']))]
    results = []
    
    for i in range(len(dfa)):
        
        sa = dfa.iloc[i]
        parameter = sa['Parameter']
        method = sa['Method']
        if method == 'count':
            sb = dfb.loc[dfb['Method'] == 'count']
        else:
            sb = dfb.loc[(dfb['Parameter'] == parameter) & (dfb['Method'] == method)]
        result = {
            'Type': Type,
            'Classify': a + '_vs_' + b,
            'Parameter': sa['Parameter'],
            'Method': sa['Method']}
        
        ait = []
        aic = []
        for name in dfa.columns:
            if t in name:
                value = ai(sa[name], sb[name].iloc[0])
                ait.append(value)
                result.update({name: value})
            if c in name:
                value = ai(sa[name], sb[name].iloc[0])
                aic.append(value)
                result.update({name: value})
        
        ait = pd.Series(ait)
        aic = pd.Series(aic)
        test = scipy.stats.ttest_ind(ait, aic, equal_var=False)
        ci = test.confidence_interval()
        _, mwp = scipy.stats.mannwhitneyu(ait, aic, method='exact')
        result.update({
            t + '_mean': ait.mean(),
            c + '_mean': aic.mean(),
            't': test.statistic,
            'low t': ci[0],
            'hi t': ci[1],
            'p': test.pvalue,
            'MWp': mwp})
        
        results.append(result)
    
    return pd.DataFrame(results)

compare_results = pd.DataFrame()
compare_results = pd.concat([compare_results, compare(df, 'turn', 'turn left', 'turn right')])
compare_results = pd.concat([compare_results, compare(df, 'bend', 'bend left', 'bend right')])
compare_results = pd.concat([compare_results, compare(df, 'recoil', 'bend left', 'bend right')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'turn left', 'turn right')])
compare_results = pd.concat([compare_results, compare(df, 'step', 'with turn', 'without turn')])
compare_results.to_csv('compare_results.csv', index=False)

def threeway(df, Type, a, b):
    
    dfa = df[(df['Type'] == Type) & (df['Classify'] == a) & (pd.notna(df['Stratify']))]
    dfb = df[(df['Type'] == Type) & (df['Classify'] == b) & (pd.notna(df['Stratify']))]
    results = []
    
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
        result = {
            'Type': Type,
            'Parameter': parameter,
            'Classify': a + '_vs_' + b}
        
        resultl = result.copy()
        aitl = []
        aicl = []
        for name in dfa.columns:
            if t in name:
                value = ai(sal[name], sbl[name].iloc[0])
                aitl.append(value)
                resultl.update({name: value})
            if c in name:
                value = ai(sal[name], sbl[name].iloc[0])
                aicl.append(value)
                resultl.update({name: value})
        aitl = pd.Series(aitl)
        aicl = pd.Series(aicl)
        test = scipy.stats.ttest_ind(aitl, aicl, equal_var=False)
        ci = test.confidence_interval()
        _, mwp = scipy.stats.mannwhitneyu(aitl, aicl, method='exact')
        resultl.update({
            'Stratify': strat + '_low',
            t + '_mean': aitl.mean(),
            c + '_mean': aicl.mean(),
            't': test.statistic,
            'low t': ci[0],
            'hi t': ci[1],
            'p': test.pvalue,
            'MWp': mwp})
        results.append(resultl)
        
        resultm = result.copy()
        aitm = []
        aicm = []
        for name in dfa.columns:
            if t in name:
                value = ai(sam[name].iloc[0], sbm[name].iloc[0])
                aitm.append(value)
                resultm.update({name: value})
            if c in name:
                value = ai(sam[name].iloc[0], sbm[name].iloc[0])
                aicm.append(value)
                resultm.update({name: value})
        aitm = pd.Series(aitm)
        aicm = pd.Series(aicm)
        test = scipy.stats.ttest_ind(aitm, aicm, equal_var=False)
        ci = test.confidence_interval()
        _, mwp = scipy.stats.mannwhitneyu(aitm, aicm, method='exact')
        resultm.update({
            'Stratify': strat + '_mid',
            t + '_mean': aitm.mean(),
            c + '_mean': aicm.mean(),
            't': test.statistic,
            'low t': ci[0],
            'hi t': ci[1],
            'p': test.pvalue,
            'MWp': mwp})
        results.append(resultm)
        
        resulth = result.copy()
        aith = []
        aich = []
        for name in dfa.columns:
            if t in name:
                value = ai(sah[name].iloc[0], sbh[name].iloc[0])
                aith.append(value)
                resulth.update({name: value})
            if c in name:
                value = ai(sah[name].iloc[0], sbh[name].iloc[0])
                aich.append(value)
                resulth.update({name: value})
        aith = pd.Series(aith)
        aich = pd.Series(aich)
        test = scipy.stats.ttest_ind(aith, aich, equal_var=False)
        ci = test.confidence_interval()
        _, mwp = scipy.stats.mannwhitneyu(aith, aich, method='exact')
        resulth.update({
            'Stratify': strat + '_high',
            t + '_mean': aith.mean(),
            c + '_mean': aich.mean(),
            't': test.statistic,
            'low t': ci[0],
            'hi t': ci[1],
            'p': test.pvalue,
            'MWp': mwp})
        results.append(resulth)
        
    return pd.DataFrame(results)

threeway_results = pd.DataFrame()
threeway_results = pd.concat([threeway_results, threeway(df, 'turn', 'turn left', 'turn right')])
threeway_results = pd.concat([threeway_results, threeway(df, 'bend', 'bend left', 'bend right')])
threeway_results = pd.concat([threeway_results, threeway(df, 'recoil', 'bend left', 'bend right')])
threeway_results = pd.concat([threeway_results, threeway(df, 'step', 'turn left', 'turn right')])
threeway_results = pd.concat([threeway_results, threeway(df, 'step', 'with turn', 'without turn')])
threeway_results.to_csv('threeway_results.csv', index=False)
