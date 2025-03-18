import sys
from src.data_loader import load_data
from src.utils import preprocess_data
from src.model import train_model, predict_stock

def main():
    # Default ticker symbol; change this variable to use another stock.
    ticker = "AAPL"
    
    # Optionally allow the user to specify a ticker symbol from command-line arguments.
    if len(sys.argv) > 1:
        ticker = sys.argv[1]

    print(f"Fetching historical data for: {ticker}")

    # Fetch historical data for the specified ticker using yfinance
    data = load_data(ticker, period="3mo")
    if data.empty:
        print("No data found. Please check your ticker symbol and internet connection.")
        return

    # Preprocess data (sort by date, fill missing values, and create a lag feature)
    data = preprocess_data(data)

    # Train a simple Linear Regression model on the lag feature
    model = train_model(data)

    # Use the last available lag feature as input for prediction
    sample_input = data.iloc[-1][['Lag1']].values.reshape(1, -1)
    prediction = predict_stock(model, sample_input)
    print(f"Predicted Stock Price for {ticker}: {prediction:.2f}")

if __name__ == "__main__":
    main()
