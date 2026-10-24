# ============================================================
# 2D Gaussian Process Regression Example
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


# ============================================================
# 1. Define the true function
# ============================================================

def true_function(X):
    """
    Hidden function that relates the two input variables
    x1 and x2 to the output y.
    """

    x1 = X[:, 0]
    x2 = X[:, 1]

    return np.sin(x1) * np.cos(x2)


# ============================================================
# 2. Generate training data
# ============================================================

# Make the random numbers reproducible
np.random.seed(42)

# Number of training experiments
n_train = 25

# Random training locations between 0 and 6
X_train = np.random.uniform(
    0,
    6,
    size=(n_train, 2)
)

# Standard deviation of experimental noise
noise_std = 0.05

# Generate noisy observations
y_train = (
    true_function(X_train)
    + noise_std * np.random.randn(n_train)
)


# ============================================================
# 3. Generate test data
# ============================================================

# Create a regular grid of test locations
x1_test = np.linspace(0, 6, 50)
x2_test = np.linspace(0, 6, 50)

X1, X2 = np.meshgrid(
    x1_test,
    x2_test
)

# Convert the 2D grid into a list of x1,x2 coordinates
X_test = np.column_stack((
    X1.ravel(),
    X2.ravel()
))

# Calculate the true response at the test locations
y_test = true_function(X_test)


# ============================================================
# 4. Display information about the data
# ============================================================

print("Training data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nTest data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 5. Define the Gaussian Process
# ============================================================

# RBF kernel with a separate length scale for x1 and x2
kernel = RBF(
    length_scale=[1.0, 1.0]
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=noise_std**2,
    normalize_y=True,
    n_restarts_optimizer=5
)


# ============================================================
# 6. Train the Gaussian Process
# ============================================================

gp.fit(X_train, y_train)

print("\nOptimised kernel:")
print(gp.kernel_)


# ============================================================
# 7. Make predictions
# ============================================================

y_pred, y_std = gp.predict(
    X_test,
    return_std=True
)


# ============================================================
# 8. Reshape predictions back onto the 2D grid
# ============================================================

Y_true = y_test.reshape(X1.shape)

Y_pred = y_pred.reshape(X1.shape)

Y_std = y_std.reshape(X1.shape)


# ============================================================
# 9. Calculate prediction error
# ============================================================

rmse = np.sqrt(
    np.mean((y_test - y_pred)**2)
)

print("\nTest RMSE:")
print(rmse)


# ============================================================
# 10. Plot results
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)


# ------------------------------------------------------------
# Plot 1: True response surface
# ------------------------------------------------------------

c1 = axes[0].contourf(
    X1,
    X2,
    Y_true,
    levels=30
)

fig.colorbar(
    c1,
    ax=axes[0],
    label="True response"
)

axes[0].scatter(
    X_train[:, 0],
    X_train[:, 1],
    color="red",
    edgecolor="black"
)

axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

axes[0].set_title(
    "True Function"
)


# ------------------------------------------------------------
# Plot 2: Gaussian Process prediction
# ------------------------------------------------------------

c2 = axes[1].contourf(
    X1,
    X2,
    Y_pred,
    levels=30
)

fig.colorbar(
    c2,
    ax=axes[1],
    label="Predicted response"
)

axes[1].scatter(
    X_train[:, 0],
    X_train[:, 1],
    color="red",
    edgecolor="black"
)

axes[1].set_xlabel("x1")
axes[1].set_ylabel("x2")

axes[1].set_title(
    "Gaussian Process Prediction"
)


# ------------------------------------------------------------
# Plot 3: Gaussian Process uncertainty
# ------------------------------------------------------------

c3 = axes[2].contourf(
    X1,
    X2,
    Y_std,
    levels=30
)

fig.colorbar(
    c3,
    ax=axes[2],
    label="Standard deviation"
)

axes[2].scatter(
    X_train[:, 0],
    X_train[:, 1],
    color="red",
    edgecolor="black"
)

axes[2].set_xlabel("x1")
axes[2].set_ylabel("x2")

axes[2].set_title(
    "Gaussian Process Uncertainty"
)


# Make layout tidy
plt.tight_layout()

plt.show()


# ============================================================
# 11. Identify the point of maximum uncertainty
# ============================================================

max_uncertainty_index = np.argmax(y_std)

next_point = X_test[
    max_uncertainty_index
]

print("\nPoint of maximum uncertainty:")

print("x1 =", next_point[0])
print("x2 =", next_point[1])

print(
    "Standard deviation =",
    y_std[max_uncertainty_index]
)


# ============================================================
# 12. Plot suggested location for the next experiment
# ============================================================

plt.figure(figsize=(7, 6))

plt.contourf(
    X1,
    X2,
    Y_std,
    levels=30
)

plt.colorbar(
    label="GP standard deviation"
)

# Existing experiments
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    color="red",
    edgecolor="black",
    s=60,
    label="Existing experiments"
)

# Suggested next experiment
plt.scatter(
    next_point[0],
    next_point[1],
    color="white",
    edgecolor="black",
    s=200,
    marker="*",
    label="Suggested next experiment"
)

plt.xlabel("x1")
plt.ylabel("x2")

plt.title(
    "Selecting the Next Experiment"
)

plt.legend()

plt.tight_layout()

plt.show()