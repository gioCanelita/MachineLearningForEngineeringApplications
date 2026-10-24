import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Training data
x_train = np.array([0.0, 0.8, 1.7, 2.5, 3.8, 5.0, 6.2, 7.5, 8.7, 10.0])

y_train = np.array([0.05, 0.74, 1.02, 0.55, -0.58,
                    -0.92, -0.10, 0.90, 0.72, -0.50])

# Test data
x_test = np.linspace(0, 10, 100)

# True values, useful for checking the GP prediction
y_test = np.sin(x_test)

X_train = x_train.reshape(-1, 1)
X_test = x_test.reshape(-1, 1)

kernel = RBF(length_scale=1.0)

gp = GaussianProcessRegressor(kernel=kernel)

gp.fit(X_train, y_train)

mean, std = gp.predict(X_test, return_std=True)

plt.figure(figsize=(8, 5))

# True function
plt.plot(x_test, y_test, 'k--', label='True function')

# Training observations
plt.scatter(x_train, y_train, color='red',
            label='Training data', zorder=5)

# GP prediction
plt.plot(x_test, mean, color='blue',
         label='GP mean')

# Uncertainty
plt.fill_between(
    x_test,
    mean - 2*std,
    mean + 2*std,
    alpha=0.25,
    color='blue',
    label='±2 standard deviations'
)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()