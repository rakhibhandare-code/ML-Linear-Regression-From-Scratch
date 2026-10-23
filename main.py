
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from linear_regression import LinearRegressionScratch


# 1. Load dataset
print("Loading California Housing dataset...")

housing = fetch_california_housing(as_frame=True)
df = housing.frame.dropna()

feature_names = housing.feature_names
X = df[feature_names].to_numpy(dtype=float)
y = df["MedHouseVal"].to_numpy(dtype=float)

print("Dataset shape:", X.shape)


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. Standardize features using training data only
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# Standardize target for stable gradient descent
y_mean = np.mean(y_train)
y_std = np.std(y_train)

y_train_scaled = (y_train - y_mean) / y_std


# 4. Add intercept column
X_train_b = np.column_stack(
    (np.ones(len(X_train_scaled)), X_train_scaled)
)

X_test_b = np.column_stack(
    (np.ones(len(X_test_scaled)), X_test_scaled)
)


# 5. Train from-scratch multiple-feature model
print("\nTraining NumPy model...")

model = LinearRegressionScratch(
    learning_rate=0.01,
    iterations=5000
)

model.fit(X_train_b, y_train_scaled)

pred_scratch = model.predict(X_test_b) * y_std + y_mean


# 6. Calculate evaluation metrics manually
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return float("nan")

    return 1 - ss_res / ss_tot


mse_scratch = calculate_mse(y_test, pred_scratch)
r2_scratch = calculate_r2(y_test, pred_scratch)

print("\n===== FROM-SCRATCH MODEL =====")
print(f"Test MSE: {mse_scratch:.6f}")
print(f"Test R2:  {r2_scratch:.6f}")


# 7. Train scikit-learn reference model
print("\nTraining scikit-learn reference model...")

reference_model = LinearRegression()
reference_model.fit(X_train_scaled, y_train)

pred_sklearn = reference_model.predict(X_test_scaled)

mse_sklearn = calculate_mse(y_test, pred_sklearn)
r2_sklearn = calculate_r2(y_test, pred_sklearn)

print("\n===== SCIKIT-LEARN MODEL =====")
print(f"Test MSE: {mse_sklearn:.6f}")
print(f"Test R2:  {r2_sklearn:.6f}")


# 8. Compare coefficients
scratch_intercept = model.theta[0] * y_std + y_mean
scratch_coefficients = model.theta[1:] * y_std

print("\n===== COEFFICIENT COMPARISON =====")
print(f"{'Parameter':<18} {'Scratch':>14} {'Scikit-learn':>14}")

print(
    f"{'Intercept':<18} "
    f"{scratch_intercept:>14.6f} "
    f"{reference_model.intercept_:>14.6f}"
)

for name, scratch_coef, sklearn_coef in zip(
    feature_names, scratch_coefficients, reference_model.coef_
):
    print(
        f"{name:<18} {scratch_coef:>14.6f} {sklearn_coef:>14.6f}"
    )


# 9. Plot cost convergence
plt.figure(figsize=(9, 5))
plt.plot(model.cost_iterations, model.cost_history)

plt.xlabel("Iteration")
plt.ylabel("Cost (MSE / 2)")
plt.title("Gradient Descent Cost Convergence")
plt.grid(True)
plt.tight_layout()

plt.savefig("cost_convergence.png", dpi=150)
print("\nSaved graph: cost_convergence.png")


# 10. Compare first five predictions
print("\n===== FIRST 5 TEST PREDICTIONS =====")
print(f"{'Actual':>12} {'Scratch':>12} {'Scikit-learn':>14}")

for actual, scratch, sklearn_pred in zip(
    y_test[:5], pred_scratch[:5], pred_sklearn[:5]
):
    print(
        f"{actual:>12.3f} {scratch:>12.3f} {sklearn_pred:>14.3f}"
    )


# 11. Demonstrate single-feature regression
print("\n===== SINGLE-FEATURE REGRESSION =====")

income_index = feature_names.index("MedInc")

X_single_train = X_train[:, [income_index]]
X_single_test = X_test[:, [income_index]]

single_scaler = StandardScaler()
X_single_train_scaled = single_scaler.fit_transform(X_single_train)
X_single_test_scaled = single_scaler.transform(X_single_test)

single_model = LinearRegressionScratch(
    learning_rate=0.01,
    iterations=2000
)

X_single_train_b = np.column_stack(
    (np.ones(len(X_single_train_scaled)), X_single_train_scaled)
)

X_single_test_b = np.column_stack(
    (np.ones(len(X_single_test_scaled)), X_single_test_scaled)
)

single_model.fit(X_single_train_b, y_train_scaled)

single_predictions = (
    single_model.predict(X_single_test_b) * y_std + y_mean
)

print(f"Feature used: {feature_names[income_index]}")
print(f"Test MSE: {calculate_mse(y_test, single_predictions):.6f}")
print(f"Test R2:  {calculate_r2(y_test, single_predictions):.6f}")


# 12. Finish
print("\nProject execution completed successfully.")

plt.show()
