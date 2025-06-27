from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Database Configuration
    database_url: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    
    # Model Configuration
    default_model_type: str = "xgb"
    model_cache_ttl: int = 3600
    max_concurrent_predictions: int = 10
    
    # Data Sources
    yahoo_finance_timeout: int = 30
    data_update_interval: str = "1h"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "stock-predictor"
    
    # Batch Processing
    batch_schedule: str = "0 2 * * *"
    batch_tickers: str = "AAPL,GOOGL,MSFT,TSLA,AMZN"
    
    # Security
    secret_key: str = "dev-secret-key"
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def batch_tickers_list(self) -> List[str]:
        return self.batch_tickers.split(",")


settings = Settings()