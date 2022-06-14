import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

#Data Collection and Cleaning
df=pd.read_csv('Toyota.csv',index_col=0,na_values=['??',"????"])

#Filling missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['KM'].fillna(df['KM'].mean(),inplace=True)
df['FuelType'].fillna(df['FuelType'].mode()[0],inplace=True)
df['HP'].fillna(df['HP'].mean(),inplace=True)
df['MetColor'].fillna(df['MetColor'].mode()[0],inplace=True)
df['Doors']=df['Doors'].map({'three':3,'five':5,'four':4,'3':3,'5':5,'4':4,'2':2})

#Encoding the FuelType feature
le = preprocessing.LabelEncoder()
le.fit(["Petrol","Diesel","CNG"])
df["FuelType"]=le.transform(df["FuelType"]) 

#Scaling the features
scaler=preprocessing.MinMaxScaler()
val=df.values
val_scaled=scaler.fit_transform(val)
df=pd.DataFrame(val_scaled,columns=df.columns)

#Creating target and predictor dataframes
X=df.drop('Price',axis=1)
y=df['Price']

#Splitting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

#Applying Regression Models

lr = LinearRegression()
lr.fit(X_train, y_train)

print('Linear Regression Train Score is : ' , lr.score(X_train, y_train))
print('Linear Regression Test Score is : ' , lr.score(X_test, y_test))

clf=Ridge(alpha=0.2)
clf.fit(X_train,y_train)

print('\nRidge Regression Train Score is : ' , clf.score(X_train, y_train))
print('Ridge Regression Test Score is : ' , clf.score(X_test, y_test))

ls=Lasso(alpha=0.0005)
ls.fit(X_train,y_train)

print('\nLasso Regression Train Score is : ' , ls.score(X_train, y_train))
print('Lasso Regression Test Score is : ' , ls.score(X_test, y_test))

dt=DecisionTreeRegressor(max_depth=6)
dt.fit(X_train,y_train)

print("\nDecision Tree Regression Train Score is : " , dt.score(X_train, y_train))
print("Decision Tree Regression Test Score is : " , dt.score(X_test, y_test))

knn=KNeighborsRegressor(n_neighbors=8)
knn.fit(X_train,y_train)

print("\nKNN Regression Train Score is : " , knn.score(X_train, y_train))
print("KNN Regression Test Score is : " , knn.score(X_test, y_test))

