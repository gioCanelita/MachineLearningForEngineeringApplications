import numpy as np
import matplotlib.pyplot as plt


def rbf_kernel(x1,x2,l=1,sigma=1):
    return sigma**2*np.exp(-(x1-x2)**2/(2*l**2))

X = np.linspace(0,10,20)

K = np.zeros((20,20))

for i in range(20):
    for j in range(20):
        K[i,j] = rbf_kernel(X[i],X[j])
        

plt.imshow(K)
plt.colorbar()
plt.show()

X = np.linspace(-5,5,100)

K = np.zeros((100,100))

for i in range(100):
    for j in range(100):
        K[i,j] = rbf_kernel(X[i],X[j],l=1)

samples = np.random.multivariate_normal(
            np.zeros(100),
            K,
            5)

for s in samples:
    plt.plot(X,s)

plt.show()