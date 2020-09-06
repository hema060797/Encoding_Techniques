import pandas as pd

df=pd.read_csv("data.csv")
print(df.shape)
print(df.head(5))
print(df.tail(2))
print(df.columns)

print(df['Country'].unique)

from sklearn import preprocessing 
# Import label encoder 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

# Encode labels in column 'species'. 
df1= label_encoder.fit_transform(df['Country']) 

print(df1)


