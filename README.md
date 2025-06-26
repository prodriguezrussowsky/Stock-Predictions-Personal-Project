# Stock Predictor (Work in Progress)

A local development stock prediction system that provides point forecasts and actionable trading signals. This project focuses on experimentation and model development in a local environment.

## Quick Start (Local Development)

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Data Ingestion** (Basic implementation)
   ```bash
   python -m src.ingestion.ingest_yf --ticker AAPL --days 365
   ```

3. **Feature Engineering** (Basic implementation)
   ```bash
   python -m src.features.build_features --input data/raw/AAPL.parquet
   ```

4. **Train a Model** (Basic implementation)
   ```bash
   python -m src.training.train --horizon 5d --model baseline --ticker AAPL
   ```

## Architecture (Local Focus)

- **Data Ingestion**: Yahoo Finance via yfinance (local storage)
- **Feature Engineering**: Technical indicators using ta-lib and pandas
- **Models**: 
  - **PyTorch**: LSTM, Temporal Fusion Transformer (TFT)
  - **Scikit-learn/XGBoost**: Traditional ML models
  - **Prophet**: Time series forecasting
  - **Baseline**: Simple moving averages
- **Storage**: Local parquet files and pickle models
- **Evaluation**: Backtesting with local results

## Development Status

This is a **work in progress** focused on local development and experimentation:

- ✅ Basic project structure
- ✅ Requirements and dependencies (including PyTorch)
- 🚧 Data ingestion (minimal implementation)
- 🚧 Feature engineering (basic framework)
- 🚧 Model training (baseline only)
- ❌ Model evaluation/backtesting
- ❌ API serving layer
- ❌ Advanced PyTorch models (LSTM, TFT)

## Local Development

```bash
# Run tests
pytest tests/

# Lint code  
flake8 src/

# Install in development mode
pip install -e .
```

**Note**: This project is currently focused on local development only. No deployment, containerization, or production infrastructure is implemented.