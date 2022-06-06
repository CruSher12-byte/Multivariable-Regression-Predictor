import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Collection and EDA
cars=pd.read_csv('Toyota.csv',index_col=0,na_values=['??',"????"])
cars.dropna(axis=0,inplace=True)
price=np.array(cars['Price'])

#Data Preprocessing
def featurescaling(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

n=price.shape[0]

age=np.array(cars['Age'])
age=featurescaling(age)

distance=np.array(cars['KM'])
distance=featurescaling(distance)

fueltype=np.array(cars['FuelType'])
#Encoding Fuel Type using Label Encoder
j=0
for i in fueltype:
    if i=="Petrol":
        fueltype[j]=4
    elif i=="Diesel":
        fueltype[j]=3
    elif i=="CNG":
        fueltype[j]=2
    else:
        fueltype[j]=1
    j+=1
fueltype=featurescaling(fueltype)

hp=np.array(cars["HP"])
hp=featurescaling(hp)

mc=np.array(cars["MetColor"])
mc=featurescaling(mc)

engtype=np.array(cars["Automatic"])
engtype=featurescaling(engtype)

cc=np.array(cars["CC"])
cc=featurescaling(cc)


weight=np.array(cars["Weight"])
weight=featurescaling(weight)

lbd=0.170000011

X=np.array([[1,age[k],distance[k],fueltype[k],hp[k],engtype[k],cc[k],weight[k]] for k in range(n)])
theta=np.dot(np.linalg.inv(np.dot(X.T,X)+lbd*np.identity(8)),np.dot(X.T,price))

theta_reshaped=np.reshape(theta,(8,1))
print(f"Weights:{theta_reshaped}")
predictions,actual,mse,error,incorrect,correct=[],[],[],[],0,0

for i in range(n):
    Y=np.dot(theta.T,np.array([[1],[age[i]],[distance[i]],[fueltype[i]],[hp[i]],[engtype[i]],[cc[i]],[weight[i]]]))
    
    mse.append((Y-price[i])**2)
    error.append(abs(Y-price[i]))
    if abs(Y-price[i])<=(price[i]*0.2):
        correct+=1
    else:
        incorrect+=1
    predictions.append(Y)
    actual.append(price[i])
    
rmse=np.sqrt(np.mean(mse))
mae=np.mean(error)

print(f"Correct predictions: {correct} Incorrect predictions: {incorrect}")
print(f"Accuracy: {correct/(correct+incorrect)*100}%")
print(f"Root Mean Squared Error is {rmse}")
print(f"Mean Absolute Error is {mae}")

plt.scatter(actual,predictions,c='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predictions vs Actual")

x1,y1=[0,25000],[0,25000]
plt.plot(x1,y1,marker='o',c='green')
plt.show()