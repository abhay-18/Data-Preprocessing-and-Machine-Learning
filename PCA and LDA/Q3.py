# Abhay Vijayvargiya
# B20176
# 6377967485

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statistics as stat

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
def standardization(dfr, lst):
    for s in range(len(lst)-1):
        avg = dfr[lst[s]].mean()
        stddev = dfr[lst[s]].std()
        for t in range(len(dfr)):
            xnew = ((dfr.iloc[t, s] - avg)/stddev)
            dfr.loc[t, lst[s]] = xnew
    return dfr


standardization(df, col_names)
# (a)-------------------------------------------------------------------------------------------------------------------
df = df.drop(['class'], axis=1)
dfn = df.to_numpy()
cov = np.cov(dfn.T)
cov = cov.round(3)
print('Covariance matrix for original data----\n' + str(cov))
eig_val, eig_vect = np.linalg.eig(cov)
print(eig_val)

pca = PCA(n_components=2)
pca.fit(df)
df_pca = pca.transform(df)
var1 = stat.variance(df_pca[:, 0])
var2 = stat.variance((df_pca[:, 1]))
print(var1, var2)

plt.scatter(df_pca[:, 0], df_pca[:, 1], marker='x')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# (b)-------------------------------------------------------------------------------------------------------------------
eig_val = list(eig_val)
eig_val.sort(reverse=True)
x = [1, 2, 3, 4, 5, 6, 7, 8]
plt.plot(x, eig_val, 'ro-')
plt.title('Plot of eigen values')
plt.show()

# (c)-------------------------------------------------------------------------------------------------------------------
def rec_error(dfr, lst):
    error = []
    for i in range(8):
        pca = PCA(n_components=lst[i])
        pca.fit(dfr)
        dfr_pca = pca.transform(dfr)
        dfr_org = pca.inverse_transform(dfr_pca)
        summ = 0
        for j in range(len(dfr)):
            err = np.linalg.norm(dfr_org[j, :] - dfr.iloc[j, :])
            summ = summ + err
        avg = summ/len(dfr)
        error.append(avg)
    plt.plot(lst, error, 'ro-')
    plt.yticks(np.arange(0, 3, 0.5))
    plt.title('Reconstruction error for different L')
    plt.show()


rec_error(df, x)

def print_covmat(dfr):  # (d) part of q3 is already included in this function (when n_components = 8)
    for i in range(2, 9, 1):
        pca = PCA(n_components=i)
        pca.fit(dfr)
        dfr_p = pca.transform(dfr)
        cov_mat = np.cov(dfr_p.T)
        cov_mat = cov_mat.round(3)
        print('\n\n' + str(cov_mat))


print_covmat(df)
