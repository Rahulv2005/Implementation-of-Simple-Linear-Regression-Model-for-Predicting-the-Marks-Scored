# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rahul V
RegisterNumber: 212223240132
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```



## Output:
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/2d769152-e230-4166-8e6a-c83b61a5e831)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/7d685185-51f3-42ea-8d82-c024c3fa2ca8)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/63ae8089-8db2-4f28-8f73-5fb9e23e2280)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/1edb18f9-afb4-4ab6-8a02-f70404366008)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/c49d608d-039f-491a-928b-ffeed261be62)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/8f3adafd-10f5-48f4-b8c1-95a1d4da1a8d)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/cd4e848d-eabf-47c2-8056-5c5b4866f381)
![image](https://github.com/Rahulv2005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152600335/aa1a3ac9-1b16-443f-bec2-0ad5e1a6d6fb)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
