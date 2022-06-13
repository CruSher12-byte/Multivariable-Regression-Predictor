import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings

from sqlalchemy import column

warnings.filterwarnings("ignore")

def featurescaling(arr):
    return ((arr-np.min(arr))/(np.max(arr)-np.min(arr)))

def encoding(arr):
    for i in range(0,len(arr)):
        if arr[i]=="Petrol":
            arr[i]=4
        elif arr[i]=="Diesel":
            arr[i]=3
        elif arr[i]=="CNG":
            arr[i]=2
        else:
            arr[i]=1
    return featurescaling(arr)

#Data Collection and Cleaning
df=pd.read_csv('Toyota.csv',index_col=0,na_values=['??',"????"])

#Filling missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['KM'].fillna(df['KM'].mean(),inplace=True)
df['FuelType'].fillna(df['FuelType'].mode()[0],inplace=True)
df['HP'].fillna(df['HP'].mean(),inplace=True)

#Dropping unnecessary columns
df.drop('MetColor',axis=1,inplace=True)
df.drop('Doors',axis=1,inplace=True)

#Scaling and Encoding the required features 
df['Age']=featurescaling(df['Age'])
df['KM']=featurescaling(df['KM'])
df['FuelType']=encoding(df['FuelType'])
df['HP']=featurescaling(df['HP'])
df['CC']=featurescaling(df['CC'])
df['Weight']=featurescaling(df['Weight'])

print(df.sample(10))

#Creating target and predictor dataframes
X=df.drop('Price',axis=1)
y=df['Price']

#Applying Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 64)

lr = LinearRegression()
lr.fit(X_train, y_train)

print('Linear Regression Train Score is : ' , lr.score(X_train, y_train))
print('Linear Regression Test Score is : ' , lr.score(X_test, y_test))
print('Linear Regression Coefficients are : ' , lr.coef_)
print('Linear Regression intercept is : ' , lr.intercept_)
