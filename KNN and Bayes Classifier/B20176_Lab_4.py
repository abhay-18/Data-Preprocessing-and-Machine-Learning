# Abhay Vijayvargiya
# B20176
# 6377967485

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

# Q1--------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('SteelPlateFaults-2class.csv')
grouped = df.groupby('Class')   # Grouping the data based on classes
df_0 = grouped.get_group(0)
df_1 = grouped.get_group(1)
[X_train_0, X_test_0] = train_test_split(df_0, test_size=0.3, random_state=42)   # Splitting the Data
[X_train_1, X_test_1] = train_test_split(df_1, test_size=0.3, random_state=42)
df_train = pd.concat([X_train_1, X_train_0])   # Combining the train data of both the classes
df_train = df_train.sample(frac=1)
df_train = df_train.reset_index(drop=True)
df_test = pd.concat([X_test_1, X_test_0])   # Combining the test data of both the classes
df_test = df_test.sample(frac=1)
df_train = df_train.reset_index(drop=True)
df_train.to_csv('SteelPlateFaults-train.csv', index=False)
df_test.to_csv('SteelPlateFaults-test.csv', index=False)

Class = df_train['Class']
dftest = df_test.drop(['Class'], axis=1)
dftrain = df_train.drop(['Class'], axis=1)

def knn_classifier(n, dftr, dfte):  # Function for KNN-classifier and confusion matrix
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(dftr, Class)
    pred = model.predict(dfte)
    mat = confusion_matrix(df_test['Class'], pred)
    print('For k = '+str(n)+' confusion matrix is -------> \n'+str(mat))
    print("Accuracy for k = " + str(n)+' is: ' + str(metrics.accuracy_score(df_test['Class'], pred))+'\n')


for i in range(1, 6, 2):
    knn_classifier(i, dftrain, dftest)

# Q2--------------------------------------------------------------------------------------------------------------------
scaler = MinMaxScaler()   # Normalizing the Data for KNN-Classifier
scaled_dftrain = scaler.fit_transform(dftrain)
scaled_dftest = scaler.fit_transform(dftest)

for i in range(1, 6, 2):
    knn_classifier(i, scaled_dftrain, scaled_dftest)

# Q3--------------------------------------------------------------------------------------------------------------------
df_train = df_train.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'X_Minimum', 'Y_Minimum'], axis=1)
dftest = dftest.drop(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'X_Minimum', 'Y_Minimum'], axis=1)
grp = df_train.groupby('Class')
df_train0 = grp.get_group(0)
df_train0 = df_train0.drop(['Class'], axis=1)
df_train1 = grp.get_group(1)
df_train1 = df_train1.drop(['Class'], axis=1)
col = list(df_train0.columns)
mean_0, mean_1 = [], []
for i in range(len(col)):  # Calculating mean vector
    avg0 = df_train0[col[i]].mean()
    avg1 = df_train1[col[i]].mean()
    mean_0.append(avg0)
    mean_1.append(avg1)
mean_0 = np.array(mean_0)
mean_1 = np.array(mean_1)
cov_0 = df_train0.cov()     # Calculating the covariance matrix
cov_1 = df_train1.cov()

prior_0 = len(df_train0)/len(df_train)
prior_1 = len(df_train1)/len(df_train)
def likelihood(x, mean, cov):  # Function for finding likelihood of a class
    power = -0.5*np.dot(np.dot((x-mean).T, np.linalg.inv(cov)), (x-mean))
    like = (np.exp(power))/(((2*np.pi)**13.5)*(np.linalg.det(cov))**0.5)
    return like


pred_bayes = []
for i in range(len(dftest)):   # Assigning class to each test tuple on basis of posterior probability
    tup = dftest.iloc[i, :]
    like_0 = likelihood(tup, mean_0, cov_0)
    like_1 = likelihood(tup, mean_1, cov_1)
    post_0 = (like_0 * prior_0)
    post_1 = (like_1 * prior_1)
    if post_0 > post_1:
        pred_bayes.append(0)
    else:
        pred_bayes.append(1)

np.set_printoptions(precision=3, suppress=True)
print(mean_0)
print(mean_1)
cov_0.to_csv('Covariance_matrix_0.csv', index=True)
cov_1.to_csv('Covariance_matrix_1.csv', index=True)
print('Accuracy for bayes classifier: ', metrics.accuracy_score(df_test['Class'], pred_bayes))
print(confusion_matrix(df_test['Class'], pred_bayes))
# ----------------------------------------------------------------------------------------------------------------------
