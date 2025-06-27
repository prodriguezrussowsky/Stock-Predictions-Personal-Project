import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

from ..common.config import settings
from ..common.logging import setup_logging
from ..common.utils import validate_ticker, ensure_dir
from ..ingestion.ingest_yf import YahooFinanceIngestion
from ..features.build_features import FeatureBuilder
from ..models.xgb import XGBoostModel, train_xgboost_model
from ..models.baseline import get_baseline_model, evaluate_baseline_models
from ..models.pytorch_models import LSTMModel, train_lstm_model
from ..common.trading_signals import TradingSignalGenerator, format_trading_recommendation

logger = setup_logging()


class ModelTrainer:
    # handles the whole training process from data to trained model
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.models_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ingestor = YahooFinanceIngestion()
        self.feature_builder = FeatureBuilder()
        
        self.training_results = {}
    
    def prepare_data(self, ticker: str, horizon: str, period: str = "2y") -> tuple[pd.DataFrame, pd.Series]:
        # get data from yahoo, engineer features, split into X and y
        logger.info(f"Preparing data for {ticker}, horizon: {horizon}")
        
        # make sure the inputs make sense
        ticker = validate_ticker(ticker)
        if horizon not in ["1d", "5d", "20d"]:
            raise ValueError(f"Invalid horizon: {horizon}")
        
        # grab the stock data
        data = self.ingestor.fetch_stock_data(ticker, period=period)
        if data.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # turn raw price data into features
        features = self.feature_builder.build_all_features(data, target_horizons=[horizon])
        
        # split into features (X) and target (y)
        target_col = f"target_price_{horizon}"
        if target_col not in features.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Drop rows with missing targets
        features = features.dropna(subset=[target_col])
        
        if len(features) < 100:
            raise ValueError(f"Insufficient data for training: {len(features)} rows")
        
        # Separate features and target
        X = features.drop(columns=[col for col in features.columns if col.startswith('target_')])
        y = features[target_col]
        
        logger.info(f"Prepared {len(X)} training samples with {len(X.columns)} features")
        return X, y
    
    def train_model(
        self, 
        ticker: str, 
        horizon: str, 
        model_type: str = "xgb",
        hyperparameter_tuning: bool = False,
        save_model: bool = True,
        **model_kwargs
    ) -> Dict[str, Any]:
        """Train a model for a specific ticker and horizon."""
        
        logger.info(f"Training {model_type} model for {ticker} ({horizon})")
        
        # Prepare data
        X, y = self.prepare_data(ticker, horizon)
        
        # Split data for evaluation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        training_start = datetime.now()
        
        if model_type == "xgb":
            model = train_xgboost_model(
                X_train, y_train,
                validation_split=0.2,
                hyperparameter_tuning=hyperparameter_tuning,
                **model_kwargs
            )
        elif model_type == "lstm":
            model = train_lstm_model(
                X_train, y_train,
                validation_split=0.2,
                **model_kwargs
            )
        elif model_type in ["naive", "sma", "ema", "linear_trend"]:
            model = get_baseline_model(model_type, **model_kwargs)
            model.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        training_time = (datetime.now() - training_start).total_seconds()
        
        # Evaluate model
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, train_predictions)
        test_metrics = self.calculate_metrics(y_test, test_predictions)
        
        # Generate trading signals for the most recent data
        trading_signals = None
        try:
            # Get model confidence (simplified - could be enhanced based on model type)
            if hasattr(model, 'predict_with_uncertainty'):
                recent_X = X.tail(10)  # Use last 10 data points for prediction
                uncertainty_predictions = model.predict_with_uncertainty(recent_X)
                model_confidence = 1.0 - (uncertainty_predictions.get('std', [0.1])[-1] / uncertainty_predictions.get('prediction', [1.0])[-1])
                model_confidence = max(0.0, min(1.0, model_confidence))  # Clamp between 0 and 1
                predictions_dict = uncertainty_predictions
            else:
                # Use RMSE to estimate confidence
                model_confidence = max(0.0, min(1.0, 1.0 - (test_metrics['rmse'] / y_test.mean())))
                recent_predictions = model.predict(X.tail(1))
                predictions_dict = {'prediction': recent_predictions}
            
            # Generate trading recommendation
            signal_generator = TradingSignalGenerator()
            trading_signals = signal_generator.generate_signals(
                predictions=predictions_dict,
                current_data=X,
                model_confidence=model_confidence,
                horizon=horizon
            )
            
        except Exception as e:
            logger.warning(f"Failed to generate trading signals: {e}")
            trading_signals = None
        
        # Prepare results
        results = {
            "ticker": ticker,
            "horizon": horizon,
            "model_type": model_type,
            "training_time": training_time,
            "data_size": len(X),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_params": model.get_params() if hasattr(model, 'get_params') else {},
            "trading_signals": trading_signals,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add feature importance for XGBoost
        if hasattr(model, 'get_feature_importance'):
            results["feature_importance"] = model.get_feature_importance(top_n=20).to_dict('records')
        
        # Save model
        if save_model:
            model_filename = f"{ticker}_{horizon}_{model_type}.joblib"
            model_path = self.output_dir / model_filename
            
            if hasattr(model, 'save_model'):
                model.save_model(model_path)
            else:
                import joblib
                joblib.dump(model, model_path)
            
            results["model_path"] = str(model_path)
            logger.info(f"Model saved to {model_path}")
        
        logger.info(f"Training completed - Test RMSE: {test_metrics['rmse']:.4f}")
        return results
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        y_true_direction = (y_true.shift(-1) > y_true).astype(int)[:-1]
        y_pred_direction = (y_pred[1:] > y_pred[:-1]).astype(int)
        directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy)
        }
    
    def train_multiple_models(
        self, 
        ticker: str, 
        horizons: list = ["1d", "5d", "20d"],
        model_types: list = ["xgb", "sma"],
        hyperparameter_tuning: bool = False
    ) -> Dict[str, Any]:
        """Train multiple models for a ticker across horizons."""
        
        all_results = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "models": []
        }
        
        for horizon in horizons:
            for model_type in model_types:
                try:
                    result = self.train_model(
                        ticker=ticker,
                        horizon=horizon,
                        model_type=model_type,
                        hyperparameter_tuning=hyperparameter_tuning
                    )
                    all_results["models"].append(result)
                    
                except Exception as e:
                    error_msg = f"Failed to train {model_type} for {horizon}: {str(e)}"
                    logger.error(error_msg)
                    all_results["models"].append({
                        "ticker": ticker,
                        "horizon": horizon,
                        "model_type": model_type,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return all_results
    
    def compare_models(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Compare model performance across horizons."""
        comparison_data = []
        
        for model_result in results["models"]:
            if "test_metrics" in model_result:
                comparison_data.append({
                    "model_type": model_result["model_type"],
                    "horizon": model_result["horizon"],
                    "rmse": model_result["test_metrics"]["rmse"],
                    "mae": model_result["test_metrics"]["mae"],
                    "mape": model_result["test_metrics"]["mape"],
                    "directional_accuracy": model_result["test_metrics"]["directional_accuracy"],
                    "training_time": model_result["training_time"]
                })
        
        return pd.DataFrame(comparison_data)
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Save training results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_results_{results['ticker']}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {filepath}")
        return filepath


def main():
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    parser.add_argument("--horizon", type=str, choices=["1d", "5d", "20d"], 
                       help="Prediction horizon (if not specified, trains all)")
    parser.add_argument("--model", type=str, choices=["xgb", "lstm", "naive", "sma", "ema", "linear_trend"],
                       default="xgb", help="Model type to train")
    parser.add_argument("--period", type=str, default="2y", help="Data period for training")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    parser.add_argument("--compare", action="store_true", help="Train and compare multiple models")
    parser.add_argument("--output-dir", type=str, help="Output directory for models and results")
    
    # XGBoost specific parameters
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    
    # LSTM specific parameters
    parser.add_argument("--sequence-length", type=int, default=30, help="LSTM sequence length")
    parser.add_argument("--hidden-size", type=int, default=50, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Setup trainer
    output_dir = Path(args.output_dir) if args.output_dir else None
    trainer = ModelTrainer(output_dir=output_dir)
    
    # Prepare model parameters
    model_kwargs = {}
    if args.model == "xgb":
        model_kwargs.update({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "eta": args.learning_rate
        })
    elif args.model == "lstm":
        model_kwargs.update({
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate
        })
    
    try:
        if args.compare:
            # Train and compare multiple models
            horizons = [args.horizon] if args.horizon else ["1d", "5d", "20d"]
            model_types = ["xgb", "sma", "ema"]
            
            results = trainer.train_multiple_models(
                ticker=args.ticker,
                horizons=horizons,
                model_types=model_types,
                hyperparameter_tuning=args.tune
            )
            
            # Save results
            trainer.save_results(results)
            
            # Print comparison
            comparison_df = trainer.compare_models(results)
            print("\n=== Model Comparison ===")
            print(comparison_df.to_string(index=False))
            
        else:
            # Train single model
            horizons = [args.horizon] if args.horizon else ["1d", "5d", "20d"]
            
            for horizon in horizons:
                result = trainer.train_model(
                    ticker=args.ticker,
                    horizon=horizon,
                    model_type=args.model,
                    hyperparameter_tuning=args.tune,
                    **model_kwargs
                )
                
                # Display trading signals if available
                if result.get('trading_signals'):
                    trading_output = format_trading_recommendation(
                        result['trading_signals'], 
                        args.ticker, 
                        result['model_type'], 
                        horizon
                    )
                    print(trading_output)
                else:
                    # Fallback to technical metrics
                    print(f"\n=== Training Results: {args.ticker} ({horizon}) ===")
                    print(f"Model Type: {result['model_type']}")
                    print(f"Training Time: {result['training_time']:.2f}s")
                    print(f"Test RMSE: {result['test_metrics']['rmse']:.4f}")
                    print(f"Test MAE: {result['test_metrics']['mae']:.4f}")
                    print(f"Test MAPE: {result['test_metrics']['mape']:.2f}%")
                    print(f"Directional Accuracy: {result['test_metrics']['directional_accuracy']:.2f}%")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()