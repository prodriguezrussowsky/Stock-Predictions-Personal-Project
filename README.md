# Stock Prediction Trading System

turns raw stock data into actionable trading recommendations. basically takes yahoo finance data, runs it through some ml models, and tells you whether to buy, sell, or hold with actual entry/exit points and position sizing.

## quick start

1. **install stuff**
   ```bash
   pip install -r requirements.txt
   ```

2. **train a model and get trading signals**
   ```bash
   # simple moving average model
   python -m src.training.train --ticker AAPL --model sma --horizon 5d
   
   # xgboost model  
   python -m src.training.train --ticker AAPL --model xgb --horizon 5d
   
   # lstm neural network
   python -m src.training.train --ticker AAPL --model lstm --horizon 5d --epochs 50
   ```

3. **compare multiple models**
   ```bash
   python -m src.training.train --ticker AAPL --compare
   ```

## example output

here's what you get when you run a model:

```
=== TRADING SIGNALS: AAPL (5d horizon) ===
MODEL: SMA | CONFIDENCE: 88%

PREDICTION ANALYSIS:
Current Price: $201.00
Target Price:  $230.43 (+14.6%)
Confidence Band: $196.98 - $205.02

TRADING RECOMMENDATION:
Signal: STRONG_BUY
Entry: $201.20
Target: $230.43 (+14.6% gain)
Stop Loss: $194.97 (-3.0%)
Position Size: 22.7% of portfolio (low risk)
Risk/Reward: 1:4.9

- RSI: 50 (Neutral)
- MACD: Bearish crossover
- Volatility: 1.4% (low)
- Volume: High (1.7x avg)

RISK ASSESSMENT:
- Market Risk: LOW
- Expected Holding Period: 5 days
- Key Risks: Large expected move - higher probability of model error

RATIONALE:
Model predicts strong upward movement (14.6%) with high confidence (88%). 
Technical indicators are neutral. MACD shows bearish trend.
```

## how it works

- **data**: pulls from yahoo finance, handles missing data and errors
- **features**: 91 technical indicators (rsi, macd, bollinger bands, etc.) + time features + lag features
- **models**: 
  - baseline models (sma, ema, linear trend)
  - xgboost with hyperparameter tuning
  - pytorch lstm with uncertainty quantification
- **signals**: converts predictions into actual trading advice with risk assessment and position sizing

## current status

what works:
- data ingestion from yahoo finance
- feature engineering (91 indicators)
-  multiple ml models (baseline, xgboost, lstm)
- trading signal generation with risk assessment
- uncertainty quantification
- position sizing and stop-loss calculations

haven't done yet:
- backtesting framework 
- live trading integration
- web interface
- model deployment/serving


```bash
# run linting
flake8 src/

# check types  
mypy src/

# install in dev mode
pip install -e .
```

this is built for local experimentation and research, not production trading (yet). use at your own risk