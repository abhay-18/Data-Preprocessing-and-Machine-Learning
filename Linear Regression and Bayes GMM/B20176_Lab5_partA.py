import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import metrics

# Part-A-Q1-------------------------------------------------------------------------------------------------------------
dftr = pd.read_csv('SteelPlateFaults-train.csv')
dfte = pd.read_csv('SteelPlateFaults-test.csv')
actual = dfte['Class']
dfte = dfte.drop(['Class'], axis=1)
dftr = dftr.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'X_Minimum', 'Y_Minimum'], axis=1)
dfte = dfte.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'X_Minimum', 'Y_Minimum'], axis=1)
grouped = dftr.groupby('Class')
dftr_0 = grouped.get_group(0)
dftr_1 = grouped.get_group(1)
dftr_0 = dftr_0.drop(['Class'], axis=1)
dftr_1 = dftr_1.drop(['Class'], axis=1)
prior_0 = len(dftr_0)/len(dftr)
prior_1 = len(dftr_1)/len(dftr)
def GMM(q):
    gmm_0 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5)
    gmm_1 = GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5)
    gmm_0.fit(dftr_0)
    gmm_1.fit(dftr_1)
    like_0 = gmm_0.score_samples(dfte)
    like_1 = gmm_1.score_samples(dfte)
    post_0 = like_0 + np.log(prior_0)
    post_1 = like_1 + np.log(prior_1)
    pred = []
    for i in range(len(post_0)):
        if post_0[i] > post_1[i]:
            pred.append(0)
        else:
            pred.append(1)
    print(metrics.confusion_matrix(actual, pred))
    acc = metrics.accuracy_score(actual, pred)
    print(acc)


for i in [2, 4, 8, 16]:
    GMM(i)
#-----------------------------------------------------------------------------------------------------------------------

