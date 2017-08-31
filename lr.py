import numpy as np 
X = 2* np.random.rand(100,1)
y = 4+ 3*X + np.random.randn(100,1)
X_b = np.c_[np.ones((100,1)),X]

eta = 0.1
n_iterations =1000
m = 100

theta = np.random.randn(2,1)

# for iteration in range(n_iterations):
# 	gradients = 2/m * X_b.T.dot(X_b.dot(theta)-y)
# 	theta = theta - eta*gradients

# print(theta)

n_epochs = 50
t0,t1=5,50

def learning_schedule(t):
	return t0/(t+t1)

import matplotlib.pyplot as plt
plt.plot(X,y,'b.')
plt.axis([0,2,0,15])
plt.show()

for epoch in range(n_epochs):
	plot_index = 10
	for i in range(m):
		random_index = np.random.randint(m)
		xi = X_b[random_index:random_index+2]
		yi = y[random_index:random_index+2]
		while plot_index > 0:
			plt.plot(xi,yi,'r-')
		gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
		eta = learning_schedule(epoch*m+i)
		theta = theta - eta * gradients
		plot_index-=plot_index
plt.show()
print(theta)