import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

X = np.linspace(0,10,20)
y = np.sin(X) + 0.3*np.random.randn(len(X))

# Fit a straight line: y = mx + c
p = np.polyfit(X, y, 10)
y_pfit = np.polyval(p, X)

plt.scatter(X,y)
plt.plot(np.linspace(0,10,100),
         np.sin(np.linspace(0,10,100)),
         'r')
plt.plot(X,y_pfit,'--b')
plt.show()








