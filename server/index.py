# Importing the Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# Data Collection and Processing
car_dataset = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

# # inspecting the first 5 rows of the dataframe
# print(car_dataset.head())

# # checking the number of rows and columns
# print(car_dataset.shape)

# # getting some information about the dataset
# print(car_dataset.info())

# # checking the number of missing values
# print(car_dataset.isnull().sum())

# # checking teh distribution of categorical data
# print(car_dataset.fuel.value_counts())
# print(car_dataset.seller_type.value_counts())
# print(car_dataset.transmission.value_counts())
# print(car_dataset.owner.value_counts())

# -------------- Encoding the Categorical Data ----------------

# # encoding 'fuel", "seller_type", "transmission", and "owner" column
car_dataset = car_dataset.replace({
    'fuel': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4},
    'seller_type': {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2},
    'transmission': {'Manual': 0, 'Automatic': 1},
    'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}
}).infer_objects(copy=False)

# print(car_dataset.head())

# # Splitting the data into Training data and Test data
X = car_dataset.drop(['name', 'selling_price'], axis = 1)
Y = car_dataset['selling_price']

# print(X.head())
# print(Y.head())

# # Spliting training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2) 

# # Model Training

# 1. Linear Regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Prediction on Training Data
training_data_prediction = lin_reg_model.predict(X_train)

# R Squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
# print("R Squaraed Error: ", error_score)

# Visualize the actual prices and Predicted prices of training data
# plt.scatter(Y_train, training_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("predicted Price")
# plt.title("Actual Prices vs Prediceted Prices")
# plt.show()

# Prediction on Test Data
test_data_prediction = lin_reg_model.predict(X_test)

# R Squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
# print("R Squaraed Error: ", error_score)


# Visualize the actual prices and Predicted prices of test data
# plt.scatter(Y_test, test_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("predicted Price")
# plt.title("Actual Prices vs Prediceted Prices")
# plt.show()


# 2. Lasso Regression
lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

training_data_prediction = lass_reg_model.predict(X_train)

# R Squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
# print("R Squaraed Error: ", error_score)

# Visualize the actual prices and Predicted prices of training data
# plt.scatter(Y_train, training_data_prediction)
# plt.xlabel("Actual Price")
# plt.ylabel("predicted Price")
# plt.title("Actual Prices vs Prediceted Prices")
# plt.show()

# Prediction on Test Data
test_data_prediction = lass_reg_model.predict(X_test)

# R Squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
# print("R Squaraed Error: ", error_score)