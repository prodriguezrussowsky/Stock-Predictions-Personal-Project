import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

apple = yf.Ticker("AAPL")

def eda():
    # get all stock info
    apple.info

    # get historical market data
    df = apple.history(interval='30m', start='2024-07-28')

    # show meta information about the history (requires history() to be called first)
    apple.history_metadata

    # show actions (dividends, splits, capital gains)
    apple.actions
    apple.dividends
    apple.splits
    apple.capital_gains  # only for mutual funds & etfs

    # show share count
    apple.get_shares_full(start="2022-01-01", end=None)

    # show financials:
    # - income statement
    apple.income_stmt
    apple.quarterly_income_stmt
    # - balance sheet
    apple.balance_sheet
    apple.quarterly_balance_sheet
    # - cash flow statement
    apple.cashflow
    apple.quarterly_cashflow
    # see `Ticker.get_income_stmt()` for more options

    # show holders
    apple.major_holders
    apple.institutional_holders
    apple.mutualfund_holders
    apple.insider_transactions
    apple.insider_purchases
    apple.insider_roster_holders

    apple.sustainability

    # show recommendations
    apple.recommendations
    apple.recommendations_summary
    apple.upgrades_downgrades
    print("Exploratory Data Analysis")
    print("Shape of the data: ", df.shape)
    print("First 5 rows of the data: ", df.head())
    print("Statistical summary of the data: ", df.describe())
    print("Checking for null values: ", df.isnull().sum())
    print("Checking data types: ", df.dtypes)

# Load the stock data
def load_stock_data():
    data = apple.history(interval='30m', start='2024-07-01')
    return data

# Preprocess the data
def preprocess_data(data):
    data['Target'] = data['Close'].shift(-1)  # Create target column for next 30-minute closing price
    data = data.dropna()  # Drop rows with NaN values created by the shift

    data = data.reset_index() 
    
    data['Day'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['Year'] = data['Datetime'].dt.year
    data['Weekday'] = data['Datetime'].dt.weekday
    data['Hour'] = data['Datetime'].dt.hour
    data['Minute'] = data['Datetime'].dt.minute
    
    data = data.set_index('Datetime')  
    return data

# Split the data into training and testing sets
def split_data(data):
    X = data.drop('Target', axis=1)
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

# Train ML model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# model's performance
def evaluate_model(predictions, y_test):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)

def predict_next_5_hours(model, data):
    predictions = []
    last_row = data.drop('Target', axis=1).iloc[-1].copy()

    for _ in range(10):  # Predict 10 intervals of 30 minutes each
        next_prediction = model.predict(last_row.values.reshape(1, -1))
        predictions.append(next_prediction[0])
        
        last_row['Close'] = next_prediction[0]
        last_row['Minute'] += 30
        if last_row['Minute'] >= 60:
            last_row['Minute'] -= 60
            last_row['Hour'] += 1
            if last_row['Hour'] >= 24:
                last_row['Hour'] -= 24
                last_row['Day'] += 1  

    return predictions

def plot_predictions(data, predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index[-20:], data['Close'].iloc[-20:], label='Actual Prices')
    last_index = data.index[-1]
    prediction_index = pd.date_range(start=last_index, periods=11, freq='30min')[1:]
    plt.plot(prediction_index, predictions, label='Predicted Prices', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.show()

def main():
    data = load_stock_data()
    data = preprocess_data(data)
    print(data.head())
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    predictions = make_predictions(model, X_test)
    evaluate_model(predictions, y_test)
    next_predictions = predict_next_5_hours(model, data)
    print("Predicted stock prices for the next 5 hours (30-minute intervals):", next_predictions)
    plot_predictions(data, next_predictions)

if __name__ == '__main__':
    main()