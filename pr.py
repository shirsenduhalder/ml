import numpy as np 
m = 100
X = 6 * np.random.rand(m,1) - 3
y = 0.5*X**2 + 2 + np.random.randn(m,1)
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# poly_features = PolynomialFeatures(degree=2,include_bias=False)
# X_poly = poly_features.fit_transform(X)
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.intercept_, lin_reg.coef_)

# X_new = np.linspace(-3,3,100).reshape(100,1)
# # print(X_new)
# X_new_poly = poly_features.transform(X_new)
# y_new = lin_reg.predict(X_new_poly)

# import matplotlib.pyplot as plt 
# plt.plot(X_new,y_new,'r-')
# plt.plot(X,y,'b.')
# plt.show()

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(X,y)
ridge_prediction = ridge_reg.predict([[1.5]])
print(ridge_prediction)

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000 ,penalty='l2')
sgd_reg.fit(X,y.ravel())
sgd_prediction = sgd_reg.predict([[1.5]])
print(sgd_prediction)

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
lasso_prediction = lasso_reg.predict([[1.5]])
print(lasso_prediction)

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.5,l1_ratio=0.5)
elastic_net.fit(X,y)
elastic_prediction = elastic_net.predict([[1.5]])
print(elastic_prediction)