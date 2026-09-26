# Linear Regression from Scratch

A from-scratch implementation of multivariate linear regression using NumPy and the California Housing dataset.

This project focuses on understanding the mathematical foundations of linear regression and implementing the training process manually, rather than relying on a built-in regression model.

## Project Overview

The model predicts the median house value using eight numerical features from the California Housing dataset.

The project implements:

- Train/test split
- Feature standardization
- Multivariate linear regression
- Mean Squared Error (MSE)
- Gradient descent
- Root Mean Squared Error (RMSE)
- $R^2$ evaluation

The model parameters are optimized manually using NumPy.

## Dataset

The project uses the California Housing dataset provided by `scikit-learn`.

The dataset contains 20,640 observations and 8 input features:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

The target variable is the median house value.

## Mathematical Formulation

The linear regression model is

$$
\hat{y} = Xw + b
$$

where:

- $X$ is the feature matrix
- $w$ is the parameter vector
- $b$ is the bias
- $\hat{y}$ is the predicted target

### Mean Squared Error

The loss function is

```math
L(w,b)
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
```

or, in matrix form,

```math
L(w,b)
=
\frac{1}{n}
\|Xw+b-y\|^2.
```

### Gradient

The gradient with respect to the parameters is

```math
\nabla_w L
=
\frac{2}{n}
X^T(Xw+b-y)
```

and

```math
\nabla_b L
=
\frac{2}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i).
```

### Gradient Descent

The parameters are updated according to

```math
w
\leftarrow
w-\eta\nabla_w L
```

and

```math
b
\leftarrow
b-\eta\nabla_b L,
```

where $\eta$ is the learning rate.

## Data Preprocessing

The dataset is divided into training and test sets using an 80/20 split.

Feature standardization is performed using statistics calculated only from the training set:

```math
x'_{ij}
=
\frac{x_{ij}-\mu_j}{\sigma_j}.
```

The same training-set mean and standard deviation are then applied to the test set.

This prevents information from the test set from leaking into the training process.

## Results

After training with gradient descent, the model achieved approximately:

| Metric | Result |
|---|---:|
| Train MSE | 0.5246 |
| Test MSE | 0.5546 |
| Test RMSE | 0.7447 |
| Test $R^2$ | 0.5767 |

The relatively small difference between training and test MSE indicates that the model does not show a large train/test performance gap on this split.

## What I Learned

Through this project, I explored the connection between:

- Linear algebra
- Probability and statistical modeling
- Convex optimization
- Gradient descent
- Machine learning model evaluation

In particular, the project helped me understand linear regression as a quadratic optimization problem:

$$
\min_{w,b}
\frac{1}{n}
\|Xw+b-y\|^2.
$$

## How to Run

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

## Future Improvements

Possible extensions include:

- Visualization of the training loss
- Residual analysis
- Comparison with `sklearn.linear_model.LinearRegression`
- Ridge regression
- Lasso regression
- Experiments with different learning rates
- Comparison of different optimization methods


## Project Status
Completed the basic implementation of linear regression from scratch,including data preprocessing,gradient descent,and model evaluation. 
