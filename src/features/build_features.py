import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import ta

from ..common.config import settings
from ..common.logging import setup_logging
from ..common.utils import ensure_dir

logger = setup_logging()


class FeatureBuilder:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
        self.processed_dir = ensure_dir(self.data_dir / "processed")
    
    def build_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build technical analysis features."""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'price_to_sma_{window}'] = df['Close'] / df[f'sma_{window}']
        
        # Volatility features
        df['volatility_5d'] = df['returns'].rolling(window=5).std()
        df['volatility_20d'] = df['returns'].rolling(window=20).std()
        
        # Volume features
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        
        # Technical indicators using ta library
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['bb_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum indicators
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Trend indicators
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df['cci'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        
        return df
    
    def build_time_features(self, data: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
        """Build time-based features."""
        df = data.copy()
        
        if date_col not in df.columns:
            logger.warning(f"Date column '{date_col}' not found")
            return df
            
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract time components
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['weekday'] = df[date_col].dt.weekday
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Market timing features
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        df['is_year_end'] = df[date_col].dt.is_year_end.astype(int)
        
        return df
    
    def build_lag_features(self, data: pd.DataFrame, target_col: str = 'Close', lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Build lagged features."""
        df = data.copy()
        
        for lag in lags:
            df[f'{target_col.lower()}_lag_{lag}'] = df[target_col].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag) if 'returns' in df.columns else np.nan
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'{target_col.lower()}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col.lower()}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col.lower()}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col.lower()}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def build_target_variables(self, data: pd.DataFrame, horizons: List[str] = ['1d', '5d', '20d']) -> pd.DataFrame:
        """Build target variables for different prediction horizons."""
        df = data.copy()
        
        horizon_map = {'1d': 1, '5d': 5, '20d': 20}
        
        for horizon in horizons:
            if horizon not in horizon_map:
                logger.warning(f"Unknown horizon: {horizon}")
                continue
                
            periods = horizon_map[horizon]
            
            # Future price target
            df[f'target_price_{horizon}'] = df['Close'].shift(-periods)
            
            # Future return target
            df[f'target_return_{horizon}'] = (df[f'target_price_{horizon}'] - df['Close']) / df['Close']
            
            # Binary classification targets
            df[f'target_up_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
            df[f'target_up_2pct_{horizon}'] = (df[f'target_return_{horizon}'] > 0.02).astype(int)
            df[f'target_down_2pct_{horizon}'] = (df[f'target_return_{horizon}'] < -0.02).astype(int)
        
        return df
    
    def build_all_features(
        self, 
        data: pd.DataFrame, 
        include_technical: bool = True,
        include_time: bool = True,
        include_lags: bool = True,
        target_horizons: List[str] = ['1d', '5d', '20d']
    ) -> pd.DataFrame:
        """Build all features."""
        df = data.copy()
        
        logger.info(f"Building features for {len(df)} records")
        
        if include_technical:
            df = self.build_technical_features(df)
            logger.info("Built technical features")
        
        if include_time:
            df = self.build_time_features(df)
            logger.info("Built time features")
        
        if include_lags:
            df = self.build_lag_features(df)
            logger.info("Built lag features")
        
        if target_horizons:
            df = self.build_target_variables(df, target_horizons)
            logger.info(f"Built target variables for horizons: {target_horizons}")
        
        # Drop rows with NaN values (from rolling windows and lags)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        return df
    
    def get_feature_columns(self, exclude_targets: bool = True) -> List[str]:
        """Get list of feature columns (excluding targets and metadata)."""
        # This would be populated based on the features built
        technical_features = [
            'returns', 'log_returns', 'price_range', 'gap',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
            'volatility_5d', 'volatility_20d',
            'volume_sma_10', 'volume_ratio',
            'rsi', 'macd', 'macd_signal', 'bb_position',
            'stoch_k', 'williams_r', 'adx', 'cci'
        ]
        
        time_features = [
            'year', 'month', 'day', 'weekday', 'quarter', 'day_of_year', 'week_of_year',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
            'is_month_end', 'is_month_start', 'is_quarter_end', 'is_year_end'
        ]
        
        lag_features = [
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
            'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5', 'returns_lag_10',
            'close_rolling_mean_5', 'close_rolling_mean_10', 'close_rolling_mean_20',
            'close_rolling_std_5', 'close_rolling_std_10', 'close_rolling_std_20'
        ]
        
        return technical_features + time_features + lag_features
    
    def save_features(self, data: pd.DataFrame, ticker: str, suffix: str = "features") -> Path:
        """Save processed features to CSV."""
        filename = f"{ticker}_{suffix}.csv"
        filepath = self.processed_dir / filename
        
        data.to_csv(filepath, index=False)
        logger.info(f"Saved features to {filepath}")
        return filepath


def main():
    """CLI entry point for feature building."""
    import argparse
    from ..ingestion.ingest_yf import YahooFinanceIngestion
    
    parser = argparse.ArgumentParser(description="Build features from stock data")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--period", type=str, default="1y", help="Data period")
    parser.add_argument("--horizons", nargs="+", default=["1d", "5d", "20d"], help="Target horizons")
    
    args = parser.parse_args()
    
    # Fetch data
    ingestor = YahooFinanceIngestion()
    data = ingestor.fetch_stock_data(args.ticker, period=args.period)
    
    if data.empty:
        print(f"No data found for {args.ticker}")
        return
    
    # Build features
    builder = FeatureBuilder()
    features = builder.build_all_features(data, target_horizons=args.horizons)
    
    # Save features
    builder.save_features(features, args.ticker)
    
    print(f"Built {len(features.columns)} features for {len(features)} records")


if __name__ == "__main__":
    main()