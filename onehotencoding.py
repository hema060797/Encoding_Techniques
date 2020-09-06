# importing libraries 
import numpy as np 
import pandas as pd 
data=pd.read_csv("data1.csv")
print(data.shape)
print(data.head(5))
print(data.tail(2))
print(data.columns)

# After importing the required data 
print(data) 
# label encoding the data 
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

data['Gender']= le.fit_transform(data['Gender']) 
data['Geography']= le.fit_transform(data['Geography']) 
print(data)
# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 

# creating one hot encoder object by default 
# entire data passed is one hot encoded 
onehotencoder= OneHotEncoder() 

data=onehotencoder.fit_transform(data).toarray() 

print(data)