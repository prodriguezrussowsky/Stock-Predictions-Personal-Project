import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from ..common.config import settings
from ..common.logging import setup_logging
from ..common.utils import ensure_dir, validate_ticker

logger = setup_logging()


class YahooFinanceIngestion:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
        self.raw_dir = ensure_dir(self.data_dir / "raw")
        self.processed_dir = ensure_dir(self.data_dir / "processed")
    
    def fetch_stock_data(
        self, 
        ticker: str, 
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        ticker = validate_ticker(ticker)
        
        try:
            stock = yf.Ticker(ticker)
            
            if start and end:
                data = stock.history(start=start, end=end, interval=interval)
            else:
                data = stock.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return pd.DataFrame()
                
            # Reset index to make Date a column
            data = data.reset_index()
            data['Ticker'] = ticker
            
            logger.info(f"Fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def save_raw_data(self, data: pd.DataFrame, ticker: str) -> Path:
        """Save raw data to CSV."""
        if data.empty:
            raise ValueError("Cannot save empty dataframe")
            
        filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = self.raw_dir / filename
        
        data.to_csv(filepath, index=False)
        logger.info(f"Saved raw data to {filepath}")
        return filepath
    
    def load_raw_data(self, ticker: str, date_str: Optional[str] = None) -> pd.DataFrame:
        """Load raw data from CSV."""
        if date_str:
            pattern = f"{ticker}_{date_str}*.csv"
        else:
            pattern = f"{ticker}_*.csv"
            
        files = list(self.raw_dir.glob(pattern))
        if not files:
            logger.warning(f"No raw data files found for pattern {pattern}")
            return pd.DataFrame()
            
        # Load most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading raw data from {latest_file}")
        return pd.read_csv(latest_file)
    
    def get_stock_info(self, ticker: str) -> dict:
        """Get stock information from Yahoo Finance."""
        ticker = validate_ticker(ticker)
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            logger.info(f"Retrieved info for {ticker}")
            return info
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return {}
    
    def ingest_multiple_tickers(
        self, 
        tickers: list, 
        period: str = "1y",
        save_raw: bool = True
    ) -> dict:
        """Ingest data for multiple tickers."""
        results = {}
        
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            data = self.fetch_stock_data(ticker, period=period)
            
            if not data.empty:
                if save_raw:
                    self.save_raw_data(data, ticker)
                results[ticker] = data
            else:
                results[ticker] = pd.DataFrame()
                
        return results


def main():
    """CLI entry point for data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest stock data from Yahoo Finance")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="1y", help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)")
    parser.add_argument("--no-save", action="store_true", help="Don't save raw data to file")
    
    args = parser.parse_args()
    
    ingestor = YahooFinanceIngestion()
    data = ingestor.fetch_stock_data(args.ticker, period=args.period, interval=args.interval)
    
    if not data.empty and not args.no_save:
        ingestor.save_raw_data(data, args.ticker)
        
    print(f"Ingested {len(data)} records for {args.ticker}")


if __name__ == "__main__":
    main()