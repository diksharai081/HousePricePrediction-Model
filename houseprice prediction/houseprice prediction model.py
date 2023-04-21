import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
house_price_dataset = sklearn.datasets.fetch_california_housing()
print(house_price_dataset)
# Loading the dataset to a Pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
# Print First 5 rows of our DataFrame
house_price_dataframe.head()

# add the target (price) column to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target

house_price_dataframe.head()
# checking the number of rows and Columns in the data frame
house_price_dataframe.shape

# check for missing values
house_price_dataframe.isnull().sum()

# statistical measures of the dataset
house_price_dataframe.describe()

correlation = house_price_dataframe.corr()

# constructing a heatmap to nderstand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

# loading the model
rmodel = XGBRegressor()

# training the model with X_train
rmodel.fit(X_train, Y_train)

# accuracy for prediction on training data
training_data_prediction = rmodel.predict(X_train)
print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()

# accuracy for prediction on test data
test_data_prediction = rmodel.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

from lightgbm import LGBMRegressor
# loading the model
Lmodel = LGBMRegressor()

# training the model with X_train
Lmodel.fit(X_train, Y_train)

# accuracy for prediction on training data
training_data_prediction = Lmodel.predict(X_train)
print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()

# accuracy for prediction on test data
test_data_prediction = Lmodel.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)

import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor


# model 1 predictions
y_pred_1 = rmodel.predict(X_test)
# model 2 predictions
y_pred_2 = Lmodel.predict(X_test)

# plot actual vs predicted values for both models
plt.scatter(Y_test, y_pred_1, color='blue', label='Model 1')
plt.scatter(Y_test, y_pred_2, color='red', label='Model 2')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Model Predictions')

plt.legend()
plt.show()