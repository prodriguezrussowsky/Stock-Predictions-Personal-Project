import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data by sorting, filling missing values, and creating a lag feature.
    
    Returns:
        pd.DataFrame: Preprocessed data with an additional 'Lag1' column.
    """
    # Ensure data is sorted by date
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Fill missing values in 'Close'
    data['Close'] = data['Close'].ffill()
    
    # Create a lag feature: previous day's closing price
    data['Lag1'] = data['Close'].shift(1)
    
    # Drop the first row (which has a NaN lag)
    data = data.dropna().reset_index(drop=True)
    return data
