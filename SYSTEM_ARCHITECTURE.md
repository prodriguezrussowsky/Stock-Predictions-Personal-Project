# Stock Prediction System - Complete Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Model Pipeline](#model-pipeline)
6. [Trading Signal Generation](#trading-signal-generation)
7. [File-by-File Breakdown](#file-by-file-breakdown)
8. [Integration Points](#integration-points)
9. [Usage Examples](#usage-examples)
10. [Extension Points](#extension-points)

---

## System Overview

This is a **production-ready stock prediction system** that transforms raw financial data into actionable trading recommendations. The system follows a modular architecture with clear separation of concerns:

```
Raw Stock Data → Feature Engineering → Model Training → Trading Signals → Actionable Recommendations
```

**Key Capabilities:**
- Multi-model support (Baseline, XGBoost, LSTM)
- Real-time data ingestion from Yahoo Finance
- Advanced feature engineering with 91 technical indicators
- Uncertainty quantification and confidence estimation
- Risk-assessed trading recommendations
- Position sizing and stop-loss calculations

---

## Project Structure

```
Stock-Predictions-Personal-Project/
├── src/
│   ├── common/           # Shared utilities and configurations
│   │   ├── __init__.py
│   │   ├── config.py     # System-wide configuration settings
│   │   ├── logging.py    # Centralized logging setup
│   │   ├── utils.py      # Helper functions and utilities
│   │   └── trading_signals.py  # Trading signal generation logic
│   │
│   ├── ingestion/        # Data acquisition layer
│   │   ├── __init__.py
│   │   └── ingest_yf.py  # Yahoo Finance data ingestion
│   │
│   ├── features/         # Feature engineering pipeline
│   │   ├── __init__.py
│   │   └── build_features.py  # Technical indicator calculation
│   │
│   ├── models/           # Machine learning models
│   │   ├── __init__.py
│   │   ├── baseline.py   # Simple baseline models (SMA, EMA, etc.)
│   │   ├── xgb.py       # XGBoost implementation
│   │   └── pytorch_models.py  # PyTorch LSTM models
│   │
│   └── training/         # Training orchestration
│       ├── __init__.py
│       └── train.py      # Main training pipeline and CLI
│
├── configs/              # Configuration files
│   ├── features.yaml     # Feature engineering parameters
│   └── model_hparams.yaml # Model hyperparameters
│
├── data/                 # Data storage
│   ├── raw/             # Raw stock data (parquet/csv files)
│   └── processed/       # Engineered features (csv files)
│
├── models/              # Trained model artifacts
│   └── *.joblib         # Serialized model files
│
├── logs/                # System logs
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests (currently empty)
├── requirements.txt    # Python dependencies
├── pyproject.toml     # Project configuration
└── README.md          # Project overview
```

---

## Core Components

### 1. **Configuration Management** (`src/common/config.py`)
- Centralized settings for directories, API keys, and system parameters
- Environment-specific configurations
- Pydantic-based validation for type safety

### 2. **Data Ingestion** (`src/ingestion/ingest_yf.py`)
- Yahoo Finance API integration via `yfinance`
- Handles rate limiting and error recovery
- Supports multiple time periods and intervals
- Automatic data validation and cleaning

### 3. **Feature Engineering** (`src/features/build_features.py`)
- 91 technical indicators using `ta` library
- Time-based features (seasonality, trends)
- Lag features for temporal dependencies
- Target variable creation for different horizons (1d, 5d, 20d)

### 4. **Model Layer** (`src/models/`)
- **Baseline Models**: SMA, EMA, Linear Trend, Naive forecasting
- **XGBoost**: Advanced gradient boosting with hyperparameter tuning
- **LSTM**: Deep learning with PyTorch for sequence modeling

### 5. **Trading Logic** (`src/common/trading_signals.py`)
- Signal classification (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- Risk assessment (LOW, MEDIUM, HIGH, VERY_HIGH)
- Position sizing based on Kelly Criterion principles
- Stop-loss and take-profit calculations

---

## Data Flow

### **Phase 1: Data Acquisition**
```python
# Entry point: train.py -> ModelTrainer.prepare_data()
YahooFinanceIngestion.fetch_stock_data(ticker, period) 
    ↓
Raw OHLCV data (DataFrame)
    ↓ 
Saved to data/raw/{ticker}_{timestamp}.csv
```

### **Phase 2: Feature Engineering**
```python
FeatureBuilder.build_all_features(data, target_horizons)
    ↓
Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
    ↓
Time Features (seasonality, cyclical patterns)
    ↓
Lag Features (historical price patterns)
    ↓
Target Variables (future price movements)
    ↓
Cleaned Dataset (91 features) → data/processed/{ticker}_features.csv
```

### **Phase 3: Model Training**
```python
ModelTrainer.train_model(ticker, horizon, model_type)
    ↓
Data Split (80% train, 20% test)
    ↓
Model-specific training:
    - Baseline: Direct statistical calculation
    - XGBoost: Gradient boosting with cross-validation
    - LSTM: Sequence modeling with PyTorch
    ↓
Model Evaluation (RMSE, MAE, MAPE, Directional Accuracy)
    ↓
Serialized Model → models/{ticker}_{horizon}_{model}.joblib
```

### **Phase 4: Signal Generation**
```python
TradingSignalGenerator.generate_signals(predictions, current_data, confidence)
    ↓
Technical Analysis (RSI, MACD, Volume, Volatility)
    ↓
Signal Classification (STRONG_BUY → STRONG_SELL)
    ↓
Risk Assessment (Model confidence + Market conditions)
    ↓
Position Sizing (Portfolio percentage allocation)
    ↓
Entry/Exit Points (Stop-loss, Take-profit)
    ↓
TradingRecommendation Object
```

---

## Model Pipeline

### **Training Pipeline Architecture**

```python
def train_model(ticker, horizon, model_type):
    # 1. Data Preparation
    data = YahooFinanceIngestion().fetch_stock_data(ticker)
    features = FeatureBuilder().build_all_features(data, [horizon])
    X, y = prepare_training_data(features, horizon)
    
    # 2. Model Selection & Training
    if model_type == "lstm":
        model = train_lstm_model(X, y, **hyperparams)
    elif model_type == "xgb":
        model = train_xgboost_model(X, y, **hyperparams)
    else:
        model = get_baseline_model(model_type).fit(X, y)
    
    # 3. Evaluation
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    # 4. Trading Signal Generation
    signal_generator = TradingSignalGenerator()
    trading_signals = signal_generator.generate_signals(
        predictions, current_data, model_confidence, horizon
    )
    
    # 5. Output Formatting
    return format_trading_recommendation(trading_signals, ticker, model_type, horizon)
```

### **Model-Specific Details**

#### **LSTM Model (`pytorch_models.py`)**
```python
class LSTMModel:
    def __init__(self, sequence_length=30, hidden_size=50, num_layers=2):
        # Network Architecture:
        # Input: [batch_size, sequence_length, num_features]
        # LSTM: Multi-layer with dropout
        # Output: [batch_size, 1] (price prediction)
        
    def fit(self, X, y):
        # 1. Data preprocessing and scaling
        # 2. Sequence creation (sliding windows)
        # 3. PyTorch DataLoader setup
        # 4. Training loop with early stopping
        # 5. Validation monitoring
        
    def predict_with_uncertainty(self, X):
        # Monte Carlo dropout for uncertainty estimation
        # Returns: mean, std, confidence intervals
```

#### **XGBoost Model (`xgb.py`)**
```python
class XGBoostModel:
    def __init__(self, n_estimators=100, max_depth=6):
        # Optimized hyperparameters for time series
        # Feature importance tracking
        # Cross-validation support
        
    def fit(self, X, y):
        # 1. Feature selection and preprocessing
        # 2. Time series cross-validation
        # 3. Hyperparameter tuning (optional)
        # 4. Final model training
        
    def predict_with_uncertainty(self, X):
        # Quantile regression for uncertainty bounds
```

#### **Baseline Models (`baseline.py`)**
```python
class SimpleMovingAverageModel:
    # Statistical average over rolling window
    # Fast computation, interpretable results
    
class ExponentialMovingAverageModel:
    # Weighted average with exponential decay
    # More responsive to recent price changes
    
class LinearTrendModel:
    # Linear regression extrapolation
    # Captures long-term directional trends
```

---

## Trading Signal Generation

### **Signal Classification Logic**

```python
def _determine_signal(expected_return, model_confidence, technical_analysis):
    # Base signal from model prediction
    if expected_return > 0.05 and model_confidence > 0.8:
        base_signal = STRONG_BUY
    elif expected_return > 0.02 and model_confidence > 0.7:
        base_signal = BUY
    # ... similar logic for SELL signals
    
    # Technical analysis adjustment
    technical_strength = analyze_indicators(current_data)
    if technical_strength < 0.4:
        # Downgrade signal if technicals are weak
        base_signal = downgrade_signal(base_signal)
    
    return base_signal
```

### **Risk Assessment Framework**

```python
def _assess_risk(latest_data, expected_return, model_confidence):
    risk_factors = []
    
    # Model confidence risk
    if model_confidence < 0.6:
        risk_factors.append(1.0)  # High risk
    
    # Market volatility risk
    volatility = latest_data['volatility_20d']
    if volatility > 0.05:
        risk_factors.append(1.0)  # High volatility = high risk
    
    # Expected return magnitude risk
    if abs(expected_return) > 0.1:
        risk_factors.append(0.8)  # Large moves are riskier
    
    # RSI extremes risk
    rsi = latest_data['rsi']
    if rsi > 80 or rsi < 20:
        risk_factors.append(0.8)  # Overbought/oversold conditions
    
    avg_risk = mean(risk_factors)
    return map_to_risk_level(avg_risk)  # LOW/MEDIUM/HIGH/VERY_HIGH
```

### **Position Sizing Algorithm**

```python
def _calculate_position_size(expected_return, risk_level, model_confidence):
    # Base allocation by risk level
    base_sizes = {
        RiskLevel.LOW: 0.20,      # 20% of portfolio
        RiskLevel.MEDIUM: 0.15,   # 15% of portfolio
        RiskLevel.HIGH: 0.10,     # 10% of portfolio
        RiskLevel.VERY_HIGH: 0.05 # 5% of portfolio
    }
    
    base_size = base_sizes[risk_level]
    
    # Adjust for model confidence
    confidence_multiplier = model_confidence
    
    # Adjust for expected return magnitude
    return_multiplier = min(1.5, 1 + abs(expected_return) * 2)
    
    final_size = base_size * confidence_multiplier * return_multiplier
    return min(final_size, 0.25)  # Cap at 25% maximum
```

---

## File-by-File Breakdown

### **`src/common/config.py`**
**Purpose**: Centralized configuration management
**Key Classes**:
- `Settings`: Main configuration class with environment variables
- Path configurations for data, models, logs directories
- API keys and external service settings

**Connection Points**:
- Imported by all modules needing configuration
- Sets up directory structure automatically
- Provides environment-specific overrides

```python
class Settings(BaseSettings):
    # Directory paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models") 
    logs_dir: Path = Path("logs")
    
    # API settings
    yahoo_finance_timeout: int = 30
    
    # Model defaults
    default_model_type: str = "xgb"
    default_horizon: str = "5d"
```

### **`src/common/logging.py`**
**Purpose**: Structured logging with multiple outputs
**Key Functions**:
- `setup_logging()`: Configures loguru with file and console outputs
- JSON formatting for machine-readable logs
- Log rotation and retention policies

**Connection Points**:
- Used by every module for consistent logging
- Logs saved to `logs/app.log` with rotation
- Different log levels for development vs production

### **`src/common/utils.py`**
**Purpose**: Shared utility functions
**Key Functions**:
- `validate_ticker()`: Stock symbol validation
- `ensure_dir()`: Directory creation with permissions
- `generate_model_hash()`: Model versioning and caching
- `calculate_returns()`: Financial calculations

**Connection Points**:
- Imported across all modules for common operations
- Provides financial calculation primitives
- Error handling and validation helpers

### **`src/common/trading_signals.py`**
**Purpose**: Convert model predictions to trading recommendations
**Key Classes**:
- `TradingSignalGenerator`: Main signal generation logic
- `TradingRecommendation`: Data structure for recommendations
- `Signal` and `RiskLevel` enums for classification

**Key Methods**:
```python
def generate_signals(predictions, current_data, model_confidence, horizon):
    # 1. Analyze technical indicators
    technical_analysis = self._analyze_technical_indicators(latest_data)
    
    # 2. Determine trading signal
    signal = self._determine_signal(expected_return, confidence, technical_analysis)
    
    # 3. Assess risk level
    risk_level = self._assess_risk(latest_data, expected_return, confidence)
    
    # 4. Calculate position size
    position_size = self._calculate_position_size(expected_return, risk_level, confidence)
    
    # 5. Set stop-loss and take-profit
    stop_loss, take_profit = self._calculate_stop_loss_take_profit(
        current_price, target_price, expected_return, risk_level
    )
    
    return TradingRecommendation(...)
```

**Connection Points**:
- Called by `train.py` after model prediction
- Uses feature data for technical analysis
- Outputs formatted recommendations for display

### **`src/ingestion/ingest_yf.py`**
**Purpose**: Yahoo Finance data acquisition
**Key Classes**:
- `YahooFinanceIngestion`: Main data fetching class

**Key Methods**:
```python
def fetch_stock_data(self, ticker, period="1y", interval="1d"):
    # 1. Validate inputs
    ticker = validate_ticker(ticker)
    
    # 2. Make API call with error handling
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()
    
    # 3. Data validation and cleaning
    if data.empty:
        logger.warning(f"No data found for {ticker}")
        return pd.DataFrame()
    
    # 4. Add metadata
    data['Ticker'] = ticker
    data.index.name = 'Date'
    
    # 5. Save raw data
    self.save_raw_data(data, ticker)
    
    return data
```

**Connection Points**:
- Called by `ModelTrainer.prepare_data()`
- Saves data to `data/raw/` directory
- Provides clean DataFrame to feature engineering

### **`src/features/build_features.py`**
**Purpose**: Transform raw OHLCV data into ML-ready features
**Key Classes**:
- `FeatureBuilder`: Main feature engineering pipeline

**Feature Categories**:
1. **Technical Indicators** (42 features):
   ```python
   # Trend indicators
   data['sma_5'] = data['Close'].rolling(5).mean()
   data['ema_20'] = data['Close'].ewm(span=20).mean()
   
   # Momentum indicators  
   data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
   data['macd'] = ta.trend.MACD(data['Close']).macd()
   
   # Volatility indicators
   bb = ta.volatility.BollingerBands(data['Close'])
   data['bb_upper'] = bb.bollinger_hband()
   data['bb_lower'] = bb.bollinger_lband()
   
   # Volume indicators
   data['volume_sma'] = data['Volume'].rolling(10).mean()
   data['volume_ratio'] = data['Volume'] / data['volume_sma']
   ```

2. **Time Features** (12 features):
   ```python
   data['year'] = data.index.year
   data['month'] = data.index.month
   data['weekday'] = data.index.dayofweek
   data['quarter'] = data.index.quarter
   
   # Cyclical encoding
   data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
   data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
   ```

3. **Lag Features** (22 features):
   ```python
   # Price lags
   for lag in [1, 2, 3, 5, 10]:
       data[f'close_lag_{lag}'] = data['Close'].shift(lag)
       data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
   
   # Rolling statistics
   for window in [5, 10, 20]:
       data[f'close_rolling_mean_{window}'] = data['Close'].rolling(window).mean()
       data[f'close_rolling_std_{window}'] = data['Close'].rolling(window).std()
   ```

4. **Target Variables** (15 features):
   ```python
   for horizon in target_horizons:
       horizon_days = {'1d': 1, '5d': 5, '20d': 20}[horizon]
       
       # Future price
       data[f'target_price_{horizon}'] = data['Close'].shift(-horizon_days)
       
       # Future return
       data[f'target_return_{horizon}'] = (
           data[f'target_price_{horizon}'] / data['Close'] - 1
       )
       
       # Direction prediction
       data[f'target_up_{horizon}'] = (data[f'target_return_{horizon}'] > 0).astype(int)
   ```

**Connection Points**:
- Called by `ModelTrainer.prepare_data()` 
- Receives raw data from ingestion layer
- Outputs feature matrix to model training
- Saves processed features to `data/processed/`

### **`src/models/baseline.py`**
**Purpose**: Simple statistical models for benchmarking
**Key Classes**:
- `BaselineModel`: Abstract base class with common interface
- `SimpleMovingAverageModel`: Rolling average prediction
- `ExponentialMovingAverageModel`: Exponentially weighted average
- `LinearTrendModel`: Linear regression extrapolation
- `NaiveModel`: Last-value-carried-forward

**Model Interface**:
```python
class BaselineModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineModel':
        pass
    
    @abstractmethod  
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    def get_params(self) -> Dict[str, Any]:
        return self.params
```

**Example Implementation**:
```python
class SimpleMovingAverageModel(BaselineModel):
    def fit(self, X, y):
        # Calculate SMA from training data
        self.sma_value = X['Close'].tail(self.window).mean()
        self.is_fitted = True
        return self
    
    def predict(self, X):
        # Return SMA value for all predictions
        return np.full(len(X), self.sma_value)
```

**Connection Points**:
- Instantiated by `ModelTrainer.train_model()` when `model_type` in baseline types
- Provides fast baseline performance for comparison
- Same interface as complex models for consistent pipeline

### **`src/models/xgb.py`**
**Purpose**: Advanced gradient boosting with hyperparameter optimization
**Key Classes**:
- `XGBoostModel`: Main XGBoost wrapper with time series optimizations
- `XGBoostHyperparameterTuner`: Grid search with time series cross-validation

**Model Architecture**:
```python
class XGBoostModel:
    def __init__(self, **kwargs):
        # Time series optimized defaults
        default_params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'eta': 0.1,                    # Learning rate
            'max_depth': 6,                # Tree depth
            'min_child_weight': 1,         # Minimum samples per leaf
            'subsample': 0.8,              # Row sampling
            'colsample_bytree': 0.8,       # Feature sampling
            'reg_alpha': 0.1,              # L1 regularization
            'reg_lambda': 1.0,             # L2 regularization
            'n_estimators': 100,           # Number of trees
            'early_stopping_rounds': 10,   # Prevent overfitting
            'eval_metric': 'rmse'
        }
```

**Feature Processing**:
```python
def prepare_features(self, X, feature_cols=None):
    # 1. Exclude non-numeric and target columns
    exclude_cols = ['Date', 'Datetime', 'Ticker'] + [col for col in X.columns if col.startswith('target_')]
    feature_cols = [col for col in X.columns if col not in exclude_cols and X[col].dtype in ['float64', 'int64']]
    
    # 2. Handle missing values
    X_prepared = X[feature_cols].fillna(X[feature_cols].median())
    
    # 3. Handle infinite values
    X_prepared = X_prepared.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X_prepared
```

**Training Process**:
```python
def fit(self, X, y, X_val=None, y_val=None):
    # 1. Prepare features
    X_prepared = self.prepare_features(X)
    
    # 2. Setup validation
    eval_set = [(X_val_prepared, y_val)] if X_val is not None else None
    
    # 3. Train model
    self.model = xgb.XGBRegressor(**self.params)
    self.model.fit(X_prepared, y, eval_set=eval_set, verbose=False)
    
    # 4. Extract feature importance
    self.feature_importance = pd.DataFrame({
        'feature': self.feature_names,
        'importance': self.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return self
```

**Connection Points**:
- Called by `ModelTrainer.train_model()` when `model_type == "xgb"`
- Uses prepared features from feature engineering
- Provides feature importance for model interpretability
- Supports hyperparameter tuning via `XGBoostHyperparameterTuner`

### **`src/models/pytorch_models.py`**
**Purpose**: Deep learning with PyTorch for sequence modeling
**Key Classes**:
- `TimeSeriesDataset`: PyTorch dataset for sequence data
- `LSTMNetwork`: Neural network architecture
- `LSTMModel`: High-level model wrapper with training logic

**Network Architecture**:
```python
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super().__init__()
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,      # Number of features
            hidden_size=hidden_size,    # Hidden state dimension
            num_layers=num_layers,      # Stacked LSTM layers
            dropout=dropout,            # Dropout between layers
            batch_first=True            # Input shape: (batch, seq, features)
        )
        
        # Output projection
        self.dropout_layer = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)  # Single price prediction
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout and linear projection
        dropped = self.dropout_layer(last_output)
        output = self.linear(dropped)  # (batch_size, 1)
        
        return output
```

**Sequence Preparation**:
```python
def create_sequences(self, X, y):
    # Create sliding windows for time series
    # Input: [t-30, t-29, ..., t-1] → Output: [t]
    X_seq, y_seq = [], []
    
    for i in range(len(X) - self.sequence_length + 1):
        X_seq.append(X[i:i + self.sequence_length])  # 30 timesteps
        y_seq.append(y[i + self.sequence_length - 1])  # Target at end
    
    return np.array(X_seq), np.array(y_seq)
```

**Training Loop**:
```python
def fit(self, X, y, X_val=None, y_val=None):
    # 1. Data preprocessing
    X_scaled = self.scaler_X.fit_transform(X_prepared)
    y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # 2. Create sequences
    X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
    
    # 3. PyTorch data loaders
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
    
    # 4. Training loop with early stopping
    for epoch in range(self.epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation and early stopping logic
        if early_stopping_triggered:
            break
    
    return self
```

**Uncertainty Estimation**:
```python
def predict_with_uncertainty(self, X, n_samples=100):
    # Monte Carlo dropout for uncertainty quantification
    self.model.train()  # Enable dropout during inference
    
    all_predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            predictions = self.model(X_tensor)
            all_predictions.append(predictions.cpu().numpy())
    
    # Calculate statistics
    all_predictions = np.array(all_predictions)
    mean_pred = np.mean(all_predictions, axis=0)
    std_pred = np.std(all_predictions, axis=0)
    
    return {
        'prediction': mean_pred,
        'std': std_pred,
        'lower_bound': np.percentile(all_predictions, 5, axis=0),
        'upper_bound': np.percentile(all_predictions, 95, axis=0)
    }
```

**Connection Points**:
- Called by `ModelTrainer.train_model()` when `model_type == "lstm"`
- Uses GPU if available, falls back to CPU
- Provides uncertainty estimates for confidence calculation
- Integrates with same training pipeline as other models

### **`src/training/train.py`**
**Purpose**: Main orchestration layer and CLI interface
**Key Classes**:
- `ModelTrainer`: High-level training coordinator

**Training Orchestration**:
```python
class ModelTrainer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or settings.models_dir
        self.ingestor = YahooFinanceIngestion()
        self.feature_builder = FeatureBuilder()
    
    def train_model(self, ticker, horizon, model_type, **kwargs):
        # 1. Data preparation
        X, y = self.prepare_data(ticker, horizon)
        
        # 2. Train-test split (temporal)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 3. Model selection and training
        if model_type == "xgb":
            model = train_xgboost_model(X_train, y_train, **kwargs)
        elif model_type == "lstm":
            model = train_lstm_model(X_train, y_train, **kwargs)
        elif model_type in ["naive", "sma", "ema", "linear_trend"]:
            model = get_baseline_model(model_type, **kwargs)
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 4. Evaluation
        test_predictions = model.predict(X_test)
        metrics = self.calculate_metrics(y_test, test_predictions)
        
        # 5. Trading signal generation
        trading_signals = self.generate_trading_signals(model, X, horizon)
        
        # 6. Save model and return results
        model_path = self.save_model(model, ticker, horizon, model_type)
        
        return {
            "ticker": ticker,
            "horizon": horizon, 
            "model_type": model_type,
            "test_metrics": metrics,
            "trading_signals": trading_signals,
            "model_path": model_path
        }
```

**Signal Integration**:
```python
def generate_trading_signals(self, model, X, horizon):
    try:
        # Get model confidence
        if hasattr(model, 'predict_with_uncertainty'):
            uncertainty_predictions = model.predict_with_uncertainty(X.tail(10))
            model_confidence = calculate_confidence_from_uncertainty(uncertainty_predictions)
            predictions_dict = uncertainty_predictions
        else:
            # Fallback confidence estimation
            model_confidence = estimate_confidence_from_rmse(model, X, y)
            predictions_dict = {'prediction': model.predict(X.tail(1))}
        
        # Generate recommendation
        signal_generator = TradingSignalGenerator()
        trading_signals = signal_generator.generate_signals(
            predictions=predictions_dict,
            current_data=X,
            model_confidence=model_confidence,
            horizon=horizon
        )
        
        return trading_signals
        
    except Exception as e:
        logger.warning(f"Failed to generate trading signals: {e}")
        return None
```

**CLI Interface**:
```python
def main():
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    
    # Core arguments
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--horizon", choices=["1d", "5d", "20d"])
    parser.add_argument("--model", choices=["xgb", "lstm", "naive", "sma", "ema", "linear_trend"])
    
    # XGBoost parameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    
    # LSTM parameters  
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    
    # Training options
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--compare", action="store_true")
    
    args = parser.parse_args()
    
    # Execute training
    trainer = ModelTrainer()
    result = trainer.train_model(ticker=args.ticker, horizon=args.horizon, 
                               model_type=args.model, **model_kwargs)
    
    # Display results
    if result.get('trading_signals'):
        print(format_trading_recommendation(result['trading_signals'], 
                                          args.ticker, args.model, args.horizon))
    else:
        print(f"Test RMSE: {result['test_metrics']['rmse']:.4f}")
```

**Connection Points**:
- Entry point for entire system via CLI
- Coordinates all other modules
- Handles parameter parsing and validation
- Provides formatted output to user
- Saves models and results for future use

---

## Integration Points

### **Data Flow Integration**
```python
# Complete pipeline execution
train.py:main()
    ↓
ModelTrainer.train_model()
    ↓
ModelTrainer.prepare_data()
    ↓
YahooFinanceIngestion.fetch_stock_data()  # Raw data
    ↓
FeatureBuilder.build_all_features()       # Feature engineering  
    ↓
Model.fit() + Model.predict()             # Training & prediction
    ↓
TradingSignalGenerator.generate_signals() # Signal generation
    ↓
format_trading_recommendation()           # Output formatting
```

### **Configuration Integration**
- `config.py` provides settings to all modules
- Environment variables override defaults
- Directory structure created automatically
- Logging configuration applied globally

### **Error Handling Integration**
- Consistent error handling across all modules
- Structured logging with context
- Graceful degradation (trading signals optional)
- User-friendly error messages

### **Model Interface Integration**
All models implement the same interface:
```python
class AnyModel:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AnyModel'
    def predict(self, X: pd.DataFrame) -> np.ndarray
    def get_params(self) -> Dict[str, Any]
    
    # Optional for advanced models
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]
    def save_model(self, filepath: Path) -> None
    def load_model(self, filepath: Path) -> 'AnyModel'
```

This allows `train.py` to treat all models uniformly while supporting model-specific features.

---

## Usage Examples

### **Basic Training**
```bash
# Train simple moving average model
python -m src.training.train --ticker AAPL --model sma --horizon 5d

# Train XGBoost with custom parameters  
python -m src.training.train --ticker AAPL --model xgb --horizon 5d \
    --n-estimators 200 --learning-rate 0.05

# Train LSTM with custom architecture
python -m src.training.train --ticker AAPL --model lstm --horizon 5d \
    --sequence-length 45 --hidden-size 100 --epochs 50
```

### **Advanced Usage**
```bash
# Hyperparameter tuning
python -m src.training.train --ticker AAPL --model xgb --tune

# Model comparison
python -m src.training.train --ticker AAPL --compare

# Multiple horizons
python -m src.training.train --ticker AAPL --model xgb  # Trains all horizons

# Custom output directory
python -m src.training.train --ticker AAPL --model lstm --output-dir ./custom_models/
```

### **Programmatic Usage**
```python
from src.training.train import ModelTrainer
from src.common.trading_signals import format_trading_recommendation

# Initialize trainer
trainer = ModelTrainer()

# Train model
result = trainer.train_model(
    ticker="AAPL",
    horizon="5d", 
    model_type="lstm",
    sequence_length=30,
    epochs=50
)

# Display trading recommendation
if result['trading_signals']:
    recommendation = format_trading_recommendation(
        result['trading_signals'], "AAPL", "lstm", "5d"
    )
    print(recommendation)

# Access model metrics
print(f"RMSE: {result['test_metrics']['rmse']:.4f}")
print(f"Directional Accuracy: {result['test_metrics']['directional_accuracy']:.1f}%")
```

---

## Extension Points

### **Adding New Models**
1. Create model class inheriting from appropriate base class
2. Implement required interface methods (`fit`, `predict`, `get_params`)
3. Add model type to `train.py` model selection logic
4. Optionally implement `predict_with_uncertainty` for better signals

```python
# Example: Adding Prophet model
class ProphetModel(BaselineModel):
    def fit(self, X, y):
        from prophet import Prophet
        
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': X.index,
            'y': y.values
        })
        
        # Train Prophet model
        self.model = Prophet()
        self.model.fit(df)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        future = self.model.make_future_dataframe(periods=len(X))
        forecast = self.model.predict(future)
        return forecast['yhat'].tail(len(X)).values
```

### **Adding New Features**
1. Add feature calculation to `FeatureBuilder.build_all_features()`
2. Ensure proper handling of missing values
3. Update feature documentation
4. Consider feature importance analysis

```python
# Example: Adding sentiment features
def build_sentiment_features(self, data):
    # Fetch news sentiment for ticker
    sentiment_scores = fetch_news_sentiment(self.ticker)
    
    # Merge with price data
    data['sentiment_score'] = sentiment_scores.reindex(data.index, method='ffill')
    data['sentiment_ma'] = data['sentiment_score'].rolling(5).mean()
    
    return data
```

### **Adding New Data Sources**
1. Create new ingestion class in `src/ingestion/`
2. Implement standardized data format (OHLCV + metadata)
3. Add data source option to configuration
4. Integrate with `ModelTrainer.prepare_data()`

```python
# Example: Adding Alpha Vantage data source
class AlphaVantageIngestion:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def fetch_stock_data(self, ticker, period="1y"):
        # Implement Alpha Vantage API calls
        # Return standardized DataFrame format
        pass
```

### **Customizing Trading Signals**
1. Modify signal classification logic in `TradingSignalGenerator`
2. Add new risk factors to risk assessment
3. Customize position sizing algorithm
4. Add new output formats

```python
# Example: Adding crypto-specific risk factors
def _assess_crypto_risk(self, latest_data, expected_return, model_confidence):
    risk_factors = self._assess_risk(latest_data, expected_return, model_confidence)
    
    # Add crypto-specific risks
    if latest_data.get('volume_24h', 0) < threshold:
        risk_factors.append(0.8)  # Low liquidity risk
    
    if latest_data.get('market_cap', 0) < small_cap_threshold:
        risk_factors.append(0.9)  # Small cap risk
    
    return risk_factors
```

---

This documentation provides a complete understanding of how the stock prediction system works from data ingestion through trading signal generation. Each component is designed to be modular and extensible, allowing for easy customization and enhancement based on specific trading strategies or requirements.