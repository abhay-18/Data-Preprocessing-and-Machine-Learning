# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# reading csv file
df = pd.read_excel('lda_data.xlsx')

df1 = df.iloc[:,:2]

classes = df.iloc[:,2]

clf = LinearDiscriminantAnalysis(n_components=1)

clf.fit(df1, classes)
X = clf.transform(df1)

X = pd.DataFrame(X,columns= ["Attribute 1"])
print(X)
classes = pd.Series(classes)

X.insert(1,"Classes",classes)

class_1 = X[X["Classes"]==1]
class_2 = X[X["Classes"]==2]

plt.scatter(class_1,np.zeros(class_1.size),c="b")
plt.scatter(class_2,np.zeros(class_2.size),c="r")
plt.show()
