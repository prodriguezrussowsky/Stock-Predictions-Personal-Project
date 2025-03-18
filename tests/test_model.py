import pandas as pd
import numpy as np
import pytest
from src.model import train_model, predict_stock

def test_train_and_predict():
    # Create dummy data with a simple linear trend
    data = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=10),
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    # Create lag feature (previous day's close)
    data['Lag1'] = data['Close'].shift(1)
    data = data.dropna().reset_index(drop=True)
    
    model = train_model(data)
    # Use the last 'Lag1' value for prediction
    sample_input = np.array([[data.iloc[-1]['Lag1']]])
    prediction = predict_stock(model, sample_input)
    assert isinstance(prediction, float)
