# Importing the Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

def encode_data(df):
    return df.replace({
        'fuel': {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4},
        'seller_type': {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2},
        'transmission': {'Manual': 0, 'Automatic': 1},
        'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2,
                  'Fourth & Above Owner': 3, 'Test Drive Car': 4}
    }).infer_objects(copy=False)


def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"R² Score (Train): {metrics.r2_score(Y_train, train_pred):.4f}")
    print(f"R² Score (Test): {metrics.r2_score(Y_test, test_pred):.4f}")

    plt.scatter(Y_test, test_pred, c='green')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model.__class__.__name__} - Actual vs Predicted")
    plt.show()


if __name__ == "__main__":
    car_dataset = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
    car_dataset = encode_data(car_dataset)

    X = car_dataset.drop(['name', 'selling_price'], axis=1)
    Y = car_dataset['selling_price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

    print("Linear Regression:")
    evaluate_model(LinearRegression(), X_train, Y_train, X_test, Y_test)

    print("Lasso Regression:")
    evaluate_model(Lasso(), X_train, Y_train, X_test, Y_test)