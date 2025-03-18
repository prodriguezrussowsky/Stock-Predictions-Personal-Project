import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(data: pd.DataFrame):
    """
    Trains a simple Linear Regression model using 'Lag1' as feature and 'Close' as target.
    """
    if 'Lag1' not in data.columns:
        raise ValueError("Data must contain a 'Lag1' column for training.")

    X = data[['Lag1']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_stock(model, input_data):
    """
    Predicts the stock price using the trained model.
    """
    prediction = model.predict(input_data)
    return float(prediction[0])
