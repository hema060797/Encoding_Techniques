import pandas as pd
#import numpy as np

# Define the headers since the data does not have any
headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/autos/imports-85.data",
                  header=None, names=headers, na_values="?" )
print(df.head())
print(df.dtypes)
#Since this article will only focus on encoding the categorical variables,
# we are going to include only the object columns in our dataframe. 
#Pandas has a helpful select_dtypes function which we can use to build a new dataframe 
#containing only the object columns.

print('object columns')   
obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df.head())
print('missing values')
print(obj_df[obj_df.isnull().any(axis=1)])
print(obj_df["num_doors"].value_counts())
obj_df = obj_df.fillna({"num_doors": "four"})
print(obj_df.head())
print('FIND AND REPLACE')
print(obj_df["num_cylinders"].value_counts())
#If you review the replace documentation, you can see that it is a powerful command that has many options.
# For our uses, we are going to create a mapping dictionary that contains each column to process as well as 
#a dictionary of the values to translate.

#Here is the complete dictionary for cleaning up the num_doors and num_cylinders columns
cleanup_nums = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
print('To convert the columns to numbers using replace')
print(obj_df.replace(cleanup_nums, inplace=True))
print(obj_df.head())
print(obj_df.dtypes)
print('Object type to int type')


#obj_df1=obj_df.copy()
#print(obj_df1.head())
print('LABEL ENCODING')
obj_df["body_style"] = obj_df["body_style"].astype('category')
print(obj_df.dtypes)

#Then you can assign the encoded variable to a new column using the cat.codes accessor:
print('USING CAT CODES ACCESSOR')
obj_df["body_style_cat"] = obj_df["body_style"].cat.codes
print(obj_df.head())
print(obj_df['drive_wheels'])



print('ONE HOT ENCODING')
df2=pd.get_dummies(obj_df, columns=["drive_wheels"])
print(df2)
print('The new data set contains three new columns,drive_wheels_4wd,drive_wheels_rwd,drive_wheels_fwd')
print(pd.get_dummies(obj_df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head())


#The other concept to keep in mind is that get_dummies returns the full dataframe so you will need
#to filter out the objects using select_dtypes when you are ready to do the final analysis.

#One hot encoding, is very useful but it can cause the number of columns
#to expand greatly if you have very many unique values in a column. For the number of values 
#in this example, it is not a problem. However you can see how this gets really challenging
# to manage when you have many more options.


