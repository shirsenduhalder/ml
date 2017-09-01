from sklearn import datasets
import numpy as np 
iris = datasets.load_iris()
iris_list = list(iris.keys())
# print(iris_list)
X = iris["data"][:,3:]
y  = (iris["target"] == 2).astype(np.int)
# print(X)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X,y)
X_new = np.linspace(0,3,1000).reshape(-1,1)
# print(X_new)
y_proba = log_reg.predict_proba(X_new) 
import matplotlib.pyplot as plt 
plt.plot(X_new,y_proba[:,1],'g-',label="Iris_Virginia")
plt.plot(X_new,y_proba[:,0],'b--',label="Not_virginia")
plt.show()