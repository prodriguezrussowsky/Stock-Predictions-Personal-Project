import hashlib
import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from pathlib import Path


def generate_model_hash(ticker: str, horizon: str, model_type: str, features_config: Dict) -> str:
    """Generate a unique hash for model identification."""
    content = f"{ticker}_{horizon}_{model_type}_{str(features_config)}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_model(model: Any, model_path: Path) -> None:
    """Save model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> Any:
    """Load model from disk."""
    return joblib.load(model_path)


def parse_horizon(horizon: str) -> timedelta:
    """Parse horizon string to timedelta."""
    horizon_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5), 
        "20d": timedelta(days=20)
    }
    if horizon not in horizon_map:
        raise ValueError(f"Unsupported horizon: {horizon}. Supported: {list(horizon_map.keys())}")
    return horizon_map[horizon]


def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol."""
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 10:
        raise ValueError("Invalid ticker symbol")
    return ticker


def generate_signal(current_price: float, predicted_price: float, confidence: float) -> str:
    """Generate trading signal based on prediction."""
    if confidence < 0.6:
        return "Hold"
    
    price_change_pct = (predicted_price - current_price) / current_price * 100
    
    if price_change_pct > 2.0:
        return "Long"
    elif price_change_pct < -2.0:
        return "Short"
    else:
        return "Hold"


def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """Calculate number of business days between two dates."""
    return pd.bdate_range(start=start_date, end=end_date).size - 1


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path