# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as NumPy, Pandas, Matplotlib, and the SGDRegressor model from Scikit-learn.

2.Load and preprocess the dataset by separating the input features (independent variables) and the target variables (house price and number of occupants). Split the dataset into training and testing sets.

3.Train the model by creating separate SGDRegressor models for each target variable (house price and number of occupants) and fit them using the training data.

4.Predict and evaluate the results using the testing data, compare the predicted values with the actual values, and calculate performance metrics such as Mean Squared Error (MSE) and R² Score.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Kruthika R
RegisterNumber:  212225240075

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_erro
X = np.array([
    [1200, 3, 1],
    [1500, 4, 2],
    [800, 2, 1],
    [2000, 5, 3],
    [1700, 4, 2],
    [1000, 2, 1],
    [2200, 5, 3],
    [1300, 3, 2]
])
# Targets: [Price (in lakhs), Number of Occupants]
y = np.array([
    [50, 4],
    [65, 5],
    [35, 3],
    [90, 7],
    [70, 6],
    [40, 3],
    [100, 8],
    [55, 4]
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant')
multi_regressor = MultiOutputRegressor(sgd)
multi_regressor.fit(X_train, y_train)
y_pred = multi_regressor.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted.round(2)}")
```

## Output:
<img width="1606" height="231" alt="image" src="https://github.com/user-attachments/assets/42a2e84b-2eab-45c8-a022-fbb5579ed3e7" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
