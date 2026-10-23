
# Linear Regression From Scratch Using NumPy

## 1. Project Overview

This project implements Linear Regression from scratch using NumPy without using `sklearn.linear_model` in the actual model implementation.

The objective is to understand the mathematical foundations of machine learning, including the hypothesis function, Mean Squared Error (MSE), gradient descent, and model evaluation.

The model is trained on the California Housing dataset and compared with Scikit-learn's LinearRegression implementation.

## 2. Objectives

- Implement linear regression using NumPy.
- Implement the hypothesis and cost functions manually.
- Implement gradient descent and parameter updates.
- Support single-feature and multiple-feature regression.
- Evaluate predictions using manually calculated MSE and R-squared.
- Plot cost convergence during training.
- Compare coefficients and predictions with Scikit-learn.

## 3. Technologies Used

- Python 3.12
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (dataset loading, splitting, scaling, and reference comparison)

## 4. Dataset

**Dataset:** California Housing

The dataset contains 20,640 observations and 8 numerical input features.

Features include:
- MedInc: Median income
- HouseAge: Median house age
- AveRooms: Average rooms
- AveBedrms: Average bedrooms
- Population: Population
- AveOccup: Average occupancy
- Latitude: Geographic latitude
- Longitude: Geographic longitude

**Target:** `MedHouseVal`, the median house value in units of $100,000.

The data is split into 80% training data and 20% testing data, using a fixed random state of 42.

Features are standardized using statistics calculated from the training data. The target is also standardized for stable gradient descent.

## 5. Mathematical Background

### 5.1 Hypothesis Function

The model predicts the target using:

h(X) = Xθ

Where:
- X is the input feature matrix, including an intercept column.
- θ is the vector of model parameters.
- h(X) contains the predicted values.

### 5.2 Cost Function

The model minimizes the squared prediction error using:

J(θ) = (1 / 2m) Σ(hθ(x⁽ⁱ⁾) − y⁽ⁱ⁾)²

Where:
- m is the number of training examples.
- hθ(x⁽ⁱ⁾) is the prediction for example i.
- y⁽ⁱ⁾ is the actual target.

### 5.3 Gradient Descent

The gradient for all parameters is calculated as:

∇J(θ) = Xᵀ(Xθ − y) / m

The parameters are updated using:

θ := θ − α∇J(θ)

Where α is the learning rate.

The implementation uses batch gradient descent and records the cost every 10 iterations.

## 6. Implementation

The project is organized into two main Python files.

### `linear_regression.py`

Contains the `LinearRegressionScratch` class, including:
- `hypothesis()` — calculates predictions.
- `compute_cost()` — calculates the cost function.
- `fit()` — trains the model using gradient descent.
- `predict()` — generates predictions.

### `main.py`

Performs the following operations:
1. Loads the dataset.
2. Splits the data into training and testing sets.
3. Standardizes the features and target.
4. Trains the NumPy model.
5. Calculates test MSE and R² manually.
6. Trains a Scikit-learn reference model.
7. Compares model coefficients and predictions.
8. Runs a single-feature regression experiment.
9. Saves the cost convergence graph.

## 7. Experimental Results

Results from the completed run:

| Metric | NumPy Model | Scikit-learn |
|---|---:|---:|
| Test MSE | 0.556184 | 0.555892 |
| Test R² | 0.575565 | 0.575788 |

### Single-Feature Regression

The single-feature experiment uses `MedInc` (median income).

- Test MSE: 0.709116
- Test R²: 0.458859

### Interpretation

The multiple-feature NumPy model produces results close to the Scikit-learn reference model. The cost convergence plot shows a decreasing cost during training, indicating that gradient descent is converging.

Small differences in metrics and coefficients arise because gradient descent uses a finite number of iterations.

## 8. Cost Convergence Graph

The program saves the training cost plot as:

`cost_convergence.png`

The graph shows the cost function against the number of gradient descent iterations.

## 9. How to Run the Project

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ML-Linear-Regression-From-Scratch
```

Replace the placeholder with your actual repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the program

```bash
python main.py
```

The first run may require an internet connection to download the California Housing dataset.

## 10. Project Structure

```text
ML-Linear-Regression-From-Scratch/
├── data/
├── linear_regression.py
├── main.py
├── requirements.txt
├── README.md
└── cost_convergence.png
```

## 11. Conclusion

This project demonstrates the implementation of linear regression from scratch using NumPy. It covers the mathematical foundations of model training, gradient descent optimization, cost convergence, and manual evaluation metrics.

Comparing the implementation with Scikit-learn provides a useful reference for validating the results and understanding how standard machine learning libraries perform linear regression.

## 12. Future Improvements

- Experiment with different learning rates.
- Implement early stopping based on convergence.
- Test additional datasets.
- Investigate feature selection and regularization.
- Improve the convergence rate while maintaining stable training.
