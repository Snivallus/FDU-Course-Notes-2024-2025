import numpy as np
import pandas as pd

# Data initialization
y = np.array([0.28, 0.278, 0.38, 0.392, 0.266, 0.258, 0.332, 0.334])
X = np.array([
    [1, 10, 20],
    [1, 10, 20],
    [1, 10, 40],
    [1, 10, 40],
    [1, 55, 20],
    [1, 55, 20],
    [1, 55, 40],
    [1, 55, 40]
])

n, p = X.shape  # n: number of observations, p: number of predictors

# Estimate beta_hat
XtX_inv = np.linalg.inv(X.T @ X)
beta_hat = XtX_inv @ X.T @ y

# Calculate residuals and SSE
y_hat = X @ beta_hat
residuals = y - y_hat
SSE = residuals.T @ residuals

# Estimate sigma^2
sigma2_hat = SSE / (n - p)

# Total sum of squares (SST)
y_mean = np.mean(y)
SST = np.sum((y - y_mean) ** 2)

# Regression sum of squares (SSR)
SSR = SST - SSE

# Mean squares
MSR = SSR / (p - 1)
MSE = SSE / (n - p)
MST = SST / (n - 1)

# F-statistic
F_stat = MSR / MSE

# Degrees of freedom
df_total = n - 1
df_regression = p - 1
df_error = n - p

# ANOVA Table
anova_table = pd.DataFrame({
    "Source": ["Regression", "Error", "Total"],
    "Sum of Squares": [SSR, SSE, SST],
    "Degrees of Freedom": [df_regression, df_error, df_total],
    "Mean Squares": [MSR, MSE, MST],
    "F-Value": [F_stat, None, None]
})

print(beta_hat)
print(sigma2_hat)
print(anova_table)
