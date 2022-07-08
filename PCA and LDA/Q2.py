# Abhay Vijayvargiya
# B20176
# 6377967485

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

origin = [0, 0]
mean = np.array([0, 0])
cov_mat = np.array([[13, -3], [-3, 5]])
w, v = np.linalg.eig(cov_mat)
print(w)
D = np.random.multivariate_normal(mean, cov_mat, 1000)
pca = PCA(n_components=2)
pca.fit(D)
D_pca = pca.transform(D)
plt.scatter(D[:, 0], D[:, 1], marker='x')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()
plt.scatter(D[:, 0], D[:, 1], marker='x')
plt.quiver(*origin, *v[:, 0], color='black', scale=4)
plt.quiver(*origin, *v[:, 1], color='black')
plt.show()
plt.scatter(D[:, 0], D[:, 1], marker='x')
plt.quiver(*origin, *v[:, 0], color='black', scale=4)
plt.quiver(*origin, *v[:, 1], color='black')
plt.scatter(D_pca[:, 0]*v[0, 0], D_pca[:, 0]*v[1, 0], color='red', marker='x')
plt.show()
plt.scatter(D[:, 0], D[:, 1], marker='x')
plt.quiver(*origin, *v[:, 0], color='black', scale=4)
plt.quiver(*origin, *v[:, 1], color='black')
plt.scatter(D_pca[:, 1]*v[0, 1], D_pca[:, 1]*v[1, 1], color='green', marker='x')
plt.show()

D_cap = pca.inverse_transform(D_pca)
summ = 0
for i in range(1000):
    err = np.linalg.norm(D_cap[i, :] - D[i, :])
    summ = summ + err
me = (summ/1000)
print(me.round(3))
