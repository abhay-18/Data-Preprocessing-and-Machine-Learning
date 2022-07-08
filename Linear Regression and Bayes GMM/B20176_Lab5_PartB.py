import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('abalone.csv')
[train, test] = train_test_split(df, test_size=0.3, random_state=42)
train.to_csv('abalone-train.csv', index=False)
test.to_csv('abalone-test.csv', index=False)
col_names = train.columns
all_corr = []
for i in range(7):
    corr = pearsonr(train['Rings'], train[col_names[i]])
    print('Correlation between rings and '+str(col_names[i]) + 'is: '+str(corr))

reg = LinearRegression().fit(np.array(train['Shell weight']).reshape(-1, 1), np.array(train['Rings']).reshape(-1, 1))
y_pred_tr = reg.predict(np.array(train['Shell weight']).reshape(-1, 1))
y_pred_te = reg.predict(np.array(test['Shell weight']).reshape(-1, 1))
plt.scatter(np.array(train['Shell weight']).reshape(-1, 1), np.array(train['Rings']).reshape(-1, 1), label='Train data')
plt.plot(np.array(train['Shell weight']).reshape(-1, 1), y_pred_tr, color='red', label='best fit line on train data')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Q1-(a)')
plt.legend()
plt.show()

rmse_train = mean_squared_error(np.array(train['Rings']).reshape(-1, 1), y_pred_tr, squared=False)
print('Prediction accuracy on train data: '+str(rmse_train))
rmse_test = mean_squared_error(np.array(test['Shell weight']).reshape(-1, 1), y_pred_te, squared=False)
print('Prediction accuracy on test data: '+str(rmse_test))

plt.scatter(np.array(test['Rings']).reshape(-1, 1), y_pred_te, marker='x')
plt.title('Scatter plot between actual and predicted rings of test data')
plt.xlabel('Actual rings')
plt.ylabel('Predicted rings')
plt.show()

# Q2--------------------------------------------------------------------------------------------------------------------
tr = train.drop(['Rings'], axis=1)
te = test.drop(['Rings'], axis=1)
multi_reg = LinearRegression().fit(tr, train['Rings'])
y2_pred_tr = multi_reg.predict(tr)
y2_pred_te = multi_reg.predict(te)

rmse2_tr = mean_squared_error(train['Rings'], y2_pred_tr, squared=False)
rmse2_te = mean_squared_error(test['Rings'], y2_pred_te, squared=False)
print('Prediction accuracy on train data for multivariate reg: '+str(rmse2_tr))
print('Prediction accuracy on test data for multivariate reg: '+str(rmse2_te))

plt.scatter(test['Rings'], y2_pred_te, marker='x', color='red')
plt.title('Scatter plot between actual and predicted rings of test data')
plt.xlabel('Actual rings')
plt.ylabel('Predicted rings')
plt.show()

# Q3--------------------------------------------------------------------------------------------------------------------
RMSE_tr = []
RMSE_te = []
p = [2, 3, 4, 5]
xt = np.array(train['Shell weight']).reshape(-1, 1)
yt = np.array(train['Rings']).reshape(-1, 1)
for i in p:
    poly_features = PolynomialFeatures(i)
    x_poly_tr = poly_features.fit_transform(xt)
    x_poly_te = poly_features.fit_transform(np.array(test['Shell weight']).reshape(-1,  1))
    regressor = LinearRegression()
    regressor.fit(x_poly_tr, yt)
    y3_pred_tr = regressor.predict(x_poly_tr)
    y3_pred_te = regressor.predict(x_poly_te)
    rmse3_tr = mean_squared_error(yt, y3_pred_tr, squared=False)
    RMSE_tr.append(rmse3_tr)
    rmse3_te = mean_squared_error(np.array(test['Rings']).reshape(-1, 1), y3_pred_te, squared=False)
    RMSE_te.append(rmse3_te)
    if i == 4:
        X_New = np.linspace(0, 1, 2923).reshape(-1, 1)
        X_NEW_TRANSF = poly_features.fit_transform(X_New)
        Y_New = regressor.predict(X_NEW_TRANSF)
        plt.plot(X_New, Y_New, color='red', linewidth=3)
        plt.scatter(train['Shell weight'], train['Rings'], marker='x')
        plt.xlabel('Shell Weight')
        plt.ylabel('Rings')
        plt.title('Best curve fit on train data for p = 4')
        plt.show()

        plt.scatter(test['Rings'], y3_pred_te, marker='x', color='red')
        plt.title('Scatter plot for actual rings vs predicted rings for p=4')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()
    else:
        continue

print(RMSE_tr)
print(RMSE_te)
plt.plot(p, RMSE_tr, 'ro-')
plt.xticks(np.arange(2, 6, 1))
plt.xlabel('p-value')
plt.ylabel('RMSE Error')
plt.title('RMSE vs p-value for train data')
plt.show()
plt.plot(p, RMSE_te, 'ro-')
plt.xticks(np.arange(2, 6, 1))
plt.xlabel('p-value')
plt.ylabel('RMSE Error')
plt.title('RMSE vs p-value for test data')
plt.show()

# Q4--------------------------------------------------------------------------------------------------------------------
RMSE4_tr = []
RMSE4_te = []
for i in p:
    multi_poly = PolynomialFeatures(i)
    x_mpoly_tr = multi_poly.fit_transform(tr)
    x_mpoly_te = multi_poly.fit_transform(te)
    regr = LinearRegression().fit(x_mpoly_tr, train['Rings'])
    y4_pred_tr = regr.predict(x_mpoly_tr)
    y4_pred_te = regr.predict(x_mpoly_te)
    rmse4_tr = mean_squared_error(train['Rings'], y4_pred_tr, squared=False)
    RMSE4_tr.append(rmse4_tr)
    rmse4_te = mean_squared_error(test['Rings'], y4_pred_te, squared=False)
    RMSE4_te.append(rmse4_te)
    if i == 2:
        plt.scatter(test['Rings'], y4_pred_te, marker='x')
        plt.title('Actual data vs predicted data for multivariate non-linear regression')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()
    else:
        continue

print(RMSE4_tr)
print(RMSE4_te)
plt.plot(p, RMSE4_tr, 'ro-')
plt.xticks(np.arange(2, 6, 1))
plt.xlabel('p-value')
plt.ylabel('RMSE Error')
plt.title('RMSE vs p-value for train data(Multivariate)')
plt.show()
plt.plot(p, RMSE4_te, 'ro-')
plt.xticks(np.arange(2, 6, 1))
plt.xlabel('p-value')
plt.ylabel('RMSE Error')
plt.title('RMSE vs p-value for test data(Multivariate)')
plt.show()
# ----------------------------------------------------------------------------------------------------------------------

