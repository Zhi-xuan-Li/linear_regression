import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
housing=fetch_california_housing(as_frame=True)
X_train,X_test,y_train,y_test=train_test_split(
    housing.data.values,
    housing.target.values,
    test_size=0.2,
    random_state=42
)

X=housing.data.values
y=housing.target.values
X_mean=X_train.mean(axis=0)
X_std=X_train.std(axis=0)
X_train=(X_train-X_mean)/X_std
X_test=(X_test-X_mean)/X_std

learning_rate=0.01
epochs=1000
w=np.zeros(X_train.shape[1])
b=0.0
n=len(X_train)

for epoch in range(epochs):
    y_train_pred = X_train@w+b
    error=y_train_pred-y_train
    dw=(2/n)*X_train.T@error
    db=(2/n)*np.sum(error)
    w-=learning_rate*dw
    b-=learning_rate*db
    if epoch%100==0:
        train_mse=np.mean((y_train_pred-y_train)**2)
        print(f"Epoch{epoch},Train MSE:{train_mse:.4f}")

y_train_pred = X_train@w+b
y_test_pred=X_test@w+b

train_mse=np.mean((y_train_pred-y_train)**2)
test_mse=np.mean((y_test_pred-y_test)**2)

print("Final Train MSE:",train_mse)
print("Final Test MSE:",test_mse)

y_mean=np.mean(y_test)
ss_res=np.sum((y_test_pred-y_test)**2)
ss_tot=np.sum((y_test-y_mean)**2)
r2=1-ss_res/ss_tot
print("Test R^2",r2)