from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# print(mnist)

X,y = mnist['data'],mnist['target']

import matplotlib.pyplot as plt 
import matplotlib

some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)

# plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
# plt.axis('off')
# plt.show()

X_train,X_test,y_train,y_test = X[:60000],X[60000:],y[:60000],y[60000:]

import numpy as np 
shuffled_indexes = np.random.permutation(60000)
X_train = X_train[shuffled_indexes]
y_train = y_train[shuffled_indexes]

#classifier for only 5

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42,max_iter=1000,tol=0.01)
sgd_clf.fit(X_train,y_train)
prediction = sgd_clf.decision_function([some_digit])

# from sklearn.ensemble import RandomForestClassifier
# forest_clf = RandomForestClassifier()
# forest_clf.fit(X_train,y_train)
# forest_prediction = forest_clf.predict([some_digit])
# print(forest_prediction)

#scaling the imputs

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3))

#prediction
y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)

#confusion matrix
conf_mat = confusion_matrix(y_train,y_train_pred)
# plt.matshow(conf_mat,cmap=plt.cm.gray)
# plt.show()

#error rates matrix. Normalising according to error rate
row_sums = conf_mat.sum(axis=1,keepdims=True)
norm_conf_mat = conf_mat/row_sums
np.fill_diagonal(norm_conf_mat,0)
# plt.matshow(norm_conf_mat,cmap=plt.cm.gray)
# plt.show()

#multilabel classification

from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train%2==1)
y_multilabel = np.c_[y_train_large,y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
knn_predict = knn_clf.predict([some_digit])
print(knn_predict)

from sklearn.metrics import f1_score
y_train_pred = cross_val_predict(knn_clf,X_train,y_train,cv=3)
print(f1_score(y_train,y_train_pred,average='macro'))

