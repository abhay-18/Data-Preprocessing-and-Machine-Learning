# Abhay Vijayvargiya
# B20176
# 6377967485
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy import spatial as spatial

df = pd.read_csv('Iris.csv')
df1 = df.drop(['Species'], axis=1)
pca = PCA(n_components=2)
pca.fit(df1)
df_pca = pca.transform(df1)
cov = np.cov(df1.T)
eig_vals, eig_vect = np.linalg.eig(cov)
plt.plot([1, 2, 3, 4], eig_vals, 'ro-')
plt.title('Plot of eigen values')
plt.show()
plt.scatter(df_pca[:, 0], df_pca[:, 1])
plt.title('Scatter plot of 2-D data')
plt.show()

# Q2--------------------------------------------------------------------------------------------------------------------
K = 3
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(df_pca)
kmeans_pred = kmeans.predict(df_pca)
centres = kmeans.cluster_centers_
plt.scatter(df_pca[kmeans_pred == 0, 0], df_pca[kmeans_pred == 0, 1], s=100, c='red', label='Iris-setosa')
plt.scatter(df_pca[kmeans_pred == 2, 0], df_pca[kmeans_pred == 2, 1], s=100, c='orange', label='Iris-versicolor')
plt.scatter(df_pca[kmeans_pred == 1, 0], df_pca[kmeans_pred == 1, 1], s=100, c='green', label='Iris-virginica')
plt.scatter(centres[:, 0], centres[:, 1], s=100, c='black', label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.show()
dist_measure = kmeans.inertia_
print(dist_measure)
true_class = []
for i in range(len(df)):
    if df.iloc[i, 4] == 'Iris-setosa':
        true_class.append(0)
    if df.iloc[i, 4] == 'Iris-versicolor':
        true_class.append(2)
    if df.iloc[i, 4] == 'Iris-virginica':
        true_class.append(1)

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    pur = contingency_matrix[row_ind, col_ind].sum()/np.sum(contingency_matrix)
    return pur


# Q3--------------------------------------------------------------------------------------------------------------------
distortion = []
def k_Means(n):
    k_means = KMeans(n, random_state=0)
    k_means.fit(df_pca)
    y_pred = k_means.predict(df_pca)
    distortion.append(k_means.inertia_)
    print('Purity score for k = ', str(n), ': ', str(purity_score(true_class, y_pred)))


for i in range(2, 8):
    k_Means(i)

plt.plot([x for x in range(2, 8)], distortion, 'ro-')
plt.title('Distortion measure vs k-value')
plt.xlabel('K-value')
plt.ylabel('Distortion measure')
plt.show()

# Q4--------------------------------------------------------------------------------------------------------------------
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(df_pca)
GMM_pred = gmm.predict(df_pca)
log_likeli = gmm.score(df_pca)
means = gmm.means_
plt.scatter(df_pca[GMM_pred == 0, 0], df_pca[GMM_pred == 0, 1], s=100, c='red', label='Iris-setosa')
plt.scatter(df_pca[GMM_pred == 2, 0], df_pca[GMM_pred == 2, 1], s=100, c='orange', label='Iris-versicolor')
plt.scatter(df_pca[GMM_pred == 1, 0], df_pca[GMM_pred == 1, 1], s=100, c='green', label='Iris-virginica')
plt.scatter(means[:, 0], means[:, 1], s=100, c='black', label='Centroids')
plt.title('Clustering using GMM')
plt.legend()
plt.show()
print(purity_score(true_class, GMM_pred))

# Q5--------------------------------------------------------------------------------------------------------------------
list_loglike = []
def gmm(k):
    gmm_5 = GaussianMixture(n_components=k, random_state=0)
    gmm_5.fit(df_pca)
    gmm5_pred = gmm_5.predict(df_pca)
    log_like = gmm_5.score(df_pca)
    list_loglike.append(log_like*150)
    print('Purity score for k = ', str(k), ': ', str(purity_score(true_class, gmm5_pred)))


for i in range(2, 8):
    gmm(i)

plt.plot([x for x in range(2, 8)], list_loglike, 'ro-')
plt.title('Distortion measure vs k-value')
plt.xlabel('K-value')
plt.ylabel('Distortion measure')
plt.show()

# Q6--------------------------------------------------------------------------------------------------------------------
def dbscan_predict(dbscan_model, X_new, metric=spatial.distance.euclidean):
    y_new = np.ones(shape=len(X_new), dtype=int) * -1
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


def dbscan(e, m):
    dbscan_model = DBSCAN(eps=e, min_samples=m).fit(df_pca)
    DBSCAN_predictions = dbscan_model.labels_
    predicted = dbscan_predict(dbscan_model, df_pca, metric=spatial.distance.euclidean)
    for i in range(2):
        plt.scatter(df_pca[DBSCAN_predictions == i, 0], df_pca[DBSCAN_predictions == i, 1], label=i)
    plt.legend()
    plt.title("DBSCAN clustering (eps:{} and min_samples:{})".format(e, m))
    plt.show()
    p = purity_score(true_class, predicted)
    print("Purity score (eps:{} and min_samples:{}) = {}".format(e, m, p))


dbscan(1, 4)
dbscan(1, 10)
dbscan(5, 4)
dbscan(5, 10)
# ---------------------------------------------------- END -------------------------------------------------------------
