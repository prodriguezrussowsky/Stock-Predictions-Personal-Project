import yfinance as yf
import pandas as pd

def load_data(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """
    Fetches historical stock data for the given ticker symbol using yfinance.
    
    Parameters:
        ticker (str): Stock symbol (e.g., 'AAPL').
        period (str): Data period to download (e.g., '1mo', '3mo', '1y').
        
    Returns:
        pd.DataFrame: Historical stock data with a 'Date' column.
    """
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            return pd.DataFrame()
        # Reset index to turn the Date index into a column
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
