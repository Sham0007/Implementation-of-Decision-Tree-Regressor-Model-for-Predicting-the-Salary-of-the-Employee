# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: saravanan sham prakash
RegisterNumber:  212224230254

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np

data = pd.read_csv("Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print("Predicted:", y_pred)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("New prediction:", dt.predict(np.array([[5, 6]])))


*/
```

## Output:
![image](https://github.com/user-attachments/assets/1008730d-9ed6-4086-ba6e-e83060119de7)

![image](https://github.com/user-attachments/assets/902aef33-6e86-4579-9f3b-009ab3ac7fbd)

![image](https://github.com/user-attachments/assets/6fe77711-a574-481e-ad99-5e9394de2ffa)

![image](https://github.com/user-attachments/assets/478d91dd-fffc-4f61-9c69-101002cbbfa8)

![image](https://github.com/user-attachments/assets/9d86c768-9a77-4153-8e02-3b0b03e40b6c)

![image](https://github.com/user-attachments/assets/33bd6fea-a162-4e5f-862c-4ee5aed3dda8)

![image](https://github.com/user-attachments/assets/c51907a5-b751-44de-9258-6ecfa274c2c1)

![image](https://github.com/user-attachments/assets/1ba6680a-0ae8-452e-a9fe-6148c7f0f99d)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
