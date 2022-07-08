# Abhay Vijayvargiya
# B20176
# 6377967485

import pandas as pd
import numpy as np

# Q1(a)--------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('pima-indians-diabetes.csv')

def outlierdel(dfr, lst):
    for i in range(0, len(lst)-1):
        q1 = np.percentile(dfr[lst[i]], 25, interpolation='midpoint')
        q3 = np.percentile(dfr[lst[i]], 75, interpolation='midpoint')
        iqr = q3-q1
        median = df[lst[i]].median()
        for j in range(len(dfr)):
            if dfr.iloc[j, i] < (q1 - 1.5*iqr) or dfr.iloc[j, i] > (q3 + 1.5*iqr):
                dfr.loc[j, lst[i]] = median
            else:
                continue
    return dfr


col_names = list(df.columns)
outlierdel(df, col_names)
for k in range(len(col_names)-1):
    max = df[col_names[k]].max()
    minn = df[col_names[k]].min()
    print('Maximum value for '+str(col_names[k])+' before normalization is: '+str(max))
    print('Minimum value for ' + str(col_names[k]) + ' before normalization is: ' + str(minn))

def min_max_norm(dfr, lst):
    for m in range(len(lst)-1):
        minn = dfr[lst[m]].min()
        maxx = dfr[lst[m]].max()
        n_minn = 5
        n_maxx = 12
        for n in range(len(dfr)):
            x_new = ((dfr.iloc[n, m] - minn)/(maxx - minn))*(n_maxx - n_minn) + n_minn
            dfr.loc[n, lst[m]] = x_new
        new_min = dfr[lst[m]].min()
        new_max = dfr[lst[m]].max()
        print('\nMaximum value for '+str(lst[m])+' after normalization is: '+str(new_max))
        print('\nMinimum value for '+str(lst[m])+' after normalization is: '+str(new_min))


min_max_norm(df, col_names)
# Q1(b)-----------------------------------------------------------------------------------------------------------------
df1 = pd.read_csv('pima-indians-diabetes.csv')
outlierdel(df1, col_names)
def standardization(dfr, lst):
    for s in range(len(lst)-1):
        avg = dfr[lst[s]].mean()
        stddev = dfr[lst[s]].std()
        print('\nMean of '+str(lst[s])+' before standardization is: '+str(avg.round(3)))
        print('\nStd.Dev of '+str(lst[s])+' before standardization is: '+str(stddev.round(3)))
        for t in range(len(dfr)):
            xnew = ((dfr.iloc[t, s] - avg)/stddev)
            dfr.loc[t, lst[s]] = xnew
        new_avg = dfr[lst[s]].mean()
        new_stddev = dfr[lst[s]].std()
        print('\nMean of ' + str(lst[s]) + ' after standardization is: ' + str(new_avg.round(3)))
        print('\nStd.Dev of ' + str(lst[s]) + ' after standardization is: ' + str(new_stddev.round(3)))
    return dfr


standardization(df1, col_names)

