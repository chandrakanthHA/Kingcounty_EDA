import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Reading the cleaned data
df = pd.read_csv('model_dataset.csv')

# Dummy variables creation
bed_room_dummies = pd.get_dummies(df['number_of_bedrooms'], prefix='bed_rm', drop_first=True)
bath_room_dummies = pd.get_dummies(df['numberofbathrooms_per_house'], prefix='bath_rm', drop_first=True)
floor_dummies = pd.get_dummies(df['floors'], prefix='flr', drop_first=True)
house_condition_dummies = pd.get_dummies(df['house_condition'], prefix='cond', drop_first=True)
housing_grade_dummies = pd.get_dummies(df['housing_grade'], prefix='grd', drop_first=True)
yr_built_dummies = pd.get_dummies(df['yr_built'], prefix='yr_b', drop_first=True)

# Joining the dataframe and dummyvariable
df = pd.concat([df, bath_room_dummies, housing_grade_dummies], axis=1)
df = df.drop(['numberofbathrooms_per_house','housing_grade','number_of_bedrooms','floors',
             'house_condition','yr_built'], axis=1)

# feature engineering
Y = df.price
X = df.drop(['price'], axis=1)

#splitting data
print("-----  Splitting the data in train and test ----")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#adding the constant

X_train = sm.add_constant(X_train) # adding a constant
X_test = sm.add_constant(X_test) # adding a constant

print("-----  Training the model ----")
model = sm.OLS(y_train, X_train).fit()
print_model = model.summary()

print("-----  Evaluating the model ----")
predictions = model.predict(X_train)
err_train = np.sqrt(mean_squared_error(y_train, predictions))
predictions_test = model.predict(X_test)
err_test = np.sqrt(mean_squared_error(y_test, predictions_test))

print(print_model)
print ("-------------")
print (f"RMSE on train data: {err_train}")
print (f"RMSE on test data: {err_test}")