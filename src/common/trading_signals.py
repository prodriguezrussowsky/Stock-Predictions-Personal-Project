import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .logging import setup_logging

logger = setup_logging()


class Signal(Enum):
    # basic buy/sell signals we can give
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(Enum):
    # how risky this trade looks
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class TradingRecommendation:
    # everything we need to know about a trade recommendation
    signal: Signal
    confidence: float
    current_price: float
    target_price: float
    expected_return: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    risk_level: RiskLevel
    risk_reward_ratio: float
    holding_period_days: int
    supporting_indicators: Dict[str, Any]
    risks: List[str]
    rationale: str


class TradingSignalGenerator:
    # takes model predictions and turns them into actual trading advice
    
    def __init__(self, 
                 risk_free_rate: float = 0.05,
                 max_position_size: float = 0.25,
                 default_stop_loss_pct: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.default_stop_loss_pct = default_stop_loss_pct
    
    def generate_signals(self, 
                        predictions: Dict[str, np.ndarray],
                        current_data: pd.DataFrame,
                        model_confidence: float,
                        horizon: str) -> TradingRecommendation:
        # main function - takes predictions and spits out trading advice
        
        # grab the most recent data point
        latest = current_data.iloc[-1]
        current_price = latest['Close']
        
        # figure out what the model is predicting
        if 'prediction' in predictions:
            target_price = predictions['prediction']
            if isinstance(target_price, np.ndarray):
                target_price = target_price[0] if len(target_price) > 0 else current_price
        else:
            target_price = current_price
        
        # how much return are we expecting
        expected_return = (target_price - current_price) / current_price
        
        # see if we have confidence bands
        lower_bound = predictions.get('lower_bound', [target_price])[0] if 'lower_bound' in predictions else target_price
        upper_bound = predictions.get('upper_bound', [target_price])[0] if 'upper_bound' in predictions else target_price
        
        # check what the technicals are saying
        technical_analysis = self._analyze_technical_indicators(latest)
        
        # Generate signal based on prediction and technical analysis
        signal = self._determine_signal(expected_return, model_confidence, technical_analysis)
        
        # Calculate risk metrics
        risk_level = self._assess_risk(latest, expected_return, model_confidence)
        
        # Calculate position sizing
        position_size = self._calculate_position_size(expected_return, risk_level, model_confidence)
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stop_loss_take_profit(
            current_price, target_price, expected_return, risk_level
        )
        
        # Calculate risk/reward ratio
        risk_reward_ratio = self._calculate_risk_reward_ratio(
            current_price, target_price, stop_loss
        )
        
        # Get holding period
        holding_period = self._get_holding_period(horizon)
        
        # Generate supporting indicators summary
        supporting_indicators = self._get_supporting_indicators(latest, technical_analysis)
        
        # Identify risks
        risks = self._identify_risks(latest, expected_return, model_confidence)
        
        # Generate rationale
        rationale = self._generate_rationale(signal, expected_return, technical_analysis, model_confidence)
        
        return TradingRecommendation(
            signal=signal,
            confidence=model_confidence,
            current_price=current_price,
            target_price=target_price,
            expected_return=expected_return,
            entry_price=current_price * 1.001,  # Slight premium for market orders
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size_pct=position_size,
            risk_level=risk_level,
            risk_reward_ratio=risk_reward_ratio,
            holding_period_days=holding_period,
            supporting_indicators=supporting_indicators,
            risks=risks,
            rationale=rationale
        )
    
    def _analyze_technical_indicators(self, latest_data: pd.Series) -> Dict[str, Any]:
        # look at rsi, macd, volume stuff to see what's happening
        analysis = {
            'trend': 'neutral',
            'momentum': 'neutral',
            'volume': 'normal',
            'volatility': 'normal',
            'strength': 0.5
        }
        
        try:
            # rsi tells us if it's overbought/oversold
            rsi = latest_data.get('rsi', 50)
            if rsi > 70:
                analysis['momentum'] = 'overbought'
            elif rsi < 30:
                analysis['momentum'] = 'oversold'
            elif rsi > 55:
                analysis['momentum'] = 'bullish'
            elif rsi < 45:
                analysis['momentum'] = 'bearish'
            
            # MACD analysis
            macd = latest_data.get('macd', 0)
            macd_signal = latest_data.get('macd_signal', 0)
            if macd > macd_signal:
                analysis['trend'] = 'bullish'
            elif macd < macd_signal:
                analysis['trend'] = 'bearish'
            
            # Volume analysis
            volume_ratio = latest_data.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                analysis['volume'] = 'high'
            elif volume_ratio > 1.2:
                analysis['volume'] = 'above_average'
            elif volume_ratio < 0.8:
                analysis['volume'] = 'low'
            
            # Volatility analysis
            volatility = latest_data.get('volatility_20d', 0.02)
            if volatility > 0.04:
                analysis['volatility'] = 'high'
            elif volatility > 0.025:
                analysis['volatility'] = 'elevated'
            elif volatility < 0.015:
                analysis['volatility'] = 'low'
            
            # Calculate overall strength score
            strength_factors = []
            
            # RSI contribution
            if 30 <= rsi <= 70:
                strength_factors.append(0.7)  # Neutral RSI is good
            elif rsi > 70:
                strength_factors.append(0.3)  # Overbought is risky
            else:
                strength_factors.append(0.8)  # Oversold can be opportunity
            
            # MACD contribution
            if analysis['trend'] == 'bullish':
                strength_factors.append(0.8)
            elif analysis['trend'] == 'bearish':
                strength_factors.append(0.2)
            else:
                strength_factors.append(0.5)
            
            # Volume contribution
            if analysis['volume'] in ['above_average', 'high']:
                strength_factors.append(0.8)
            else:
                strength_factors.append(0.5)
            
            analysis['strength'] = np.mean(strength_factors)
            
        except Exception as e:
            logger.warning(f"Error analyzing technical indicators: {e}")
        
        return analysis
    
    def _determine_signal(self, expected_return: float, model_confidence: float, 
                         technical_analysis: Dict[str, Any]) -> Signal:
        """Determine the trading signal based on multiple factors."""
        
        # Base signal from expected return
        if expected_return > 0.05 and model_confidence > 0.8:
            base_signal = Signal.STRONG_BUY
        elif expected_return > 0.02 and model_confidence > 0.7:
            base_signal = Signal.BUY
        elif expected_return < -0.05 and model_confidence > 0.8:
            base_signal = Signal.STRONG_SELL
        elif expected_return < -0.02 and model_confidence > 0.7:
            base_signal = Signal.SELL
        else:
            base_signal = Signal.HOLD
        
        # Adjust based on technical analysis
        technical_strength = technical_analysis.get('strength', 0.5)
        
        # If technical analysis is weak, downgrade signal
        if technical_strength < 0.4:
            if base_signal == Signal.STRONG_BUY:
                return Signal.BUY
            elif base_signal == Signal.BUY:
                return Signal.HOLD
            elif base_signal == Signal.STRONG_SELL:
                return Signal.SELL
            elif base_signal == Signal.SELL:
                return Signal.HOLD
        
        # If technical analysis is strong, potentially upgrade
        elif technical_strength > 0.7:
            if base_signal == Signal.BUY and expected_return > 0.03:
                return Signal.STRONG_BUY
            elif base_signal == Signal.SELL and expected_return < -0.03:
                return Signal.STRONG_SELL
        
        return base_signal
    
    def _assess_risk(self, latest_data: pd.Series, expected_return: float, 
                    model_confidence: float) -> RiskLevel:
        """Assess the risk level of the trade."""
        
        risk_factors = []
        
        # Model confidence factor
        if model_confidence < 0.6:
            risk_factors.append(1.0)  # High risk
        elif model_confidence < 0.8:
            risk_factors.append(0.6)  # Medium risk
        else:
            risk_factors.append(0.2)  # Low risk
        
        # Volatility factor
        volatility = latest_data.get('volatility_20d', 0.02)
        if volatility > 0.05:
            risk_factors.append(1.0)
        elif volatility > 0.03:
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.3)
        
        # Expected return magnitude factor (higher returns = higher risk)
        abs_return = abs(expected_return)
        if abs_return > 0.1:
            risk_factors.append(0.8)
        elif abs_return > 0.05:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        # RSI factor (extreme values = higher risk)
        rsi = latest_data.get('rsi', 50)
        if rsi > 80 or rsi < 20:
            risk_factors.append(0.8)
        elif rsi > 70 or rsi < 30:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        avg_risk = np.mean(risk_factors)
        
        if avg_risk > 0.8:
            return RiskLevel.VERY_HIGH
        elif avg_risk > 0.6:
            return RiskLevel.HIGH
        elif avg_risk > 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_position_size(self, expected_return: float, risk_level: RiskLevel, 
                               model_confidence: float) -> float:
        """Calculate recommended position size as percentage of portfolio."""
        
        base_size = {
            RiskLevel.LOW: 0.20,
            RiskLevel.MEDIUM: 0.15,
            RiskLevel.HIGH: 0.10,
            RiskLevel.VERY_HIGH: 0.05
        }[risk_level]
        
        # Adjust for model confidence
        confidence_multiplier = model_confidence
        
        # Adjust for expected return magnitude
        return_multiplier = min(1.5, 1 + abs(expected_return) * 2)
        
        position_size = base_size * confidence_multiplier * return_multiplier
        
        return min(position_size, self.max_position_size)
    
    def _calculate_stop_loss_take_profit(self, current_price: float, target_price: float,
                                       expected_return: float, risk_level: RiskLevel) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        
        # Stop loss based on risk level
        stop_loss_pct = {
            RiskLevel.LOW: 0.03,
            RiskLevel.MEDIUM: 0.05,
            RiskLevel.HIGH: 0.07,
            RiskLevel.VERY_HIGH: 0.10
        }[risk_level]
        
        if expected_return > 0:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = target_price
        else:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = target_price
        
        return stop_loss, take_profit
    
    def _calculate_risk_reward_ratio(self, current_price: float, target_price: float, 
                                   stop_loss: float) -> float:
        """Calculate risk/reward ratio."""
        
        potential_gain = abs(target_price - current_price)
        potential_loss = abs(stop_loss - current_price)
        
        if potential_loss == 0:
            return float('inf')
        
        return potential_gain / potential_loss
    
    def _get_holding_period(self, horizon: str) -> int:
        """Get expected holding period in days."""
        horizon_map = {
            '1d': 1,
            '5d': 5,
            '20d': 20
        }
        return horizon_map.get(horizon, 5)
    
    def _get_supporting_indicators(self, latest_data: pd.Series, 
                                 technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of supporting technical indicators."""
        
        return {
            'rsi': {
                'value': latest_data.get('rsi', 50),
                'interpretation': 'Oversold' if latest_data.get('rsi', 50) < 30 else 
                               'Overbought' if latest_data.get('rsi', 50) > 70 else 'Neutral'
            },
            'macd': {
                'trend': technical_analysis.get('trend', 'neutral'),
                'signal': 'Bullish crossover' if technical_analysis.get('trend') == 'bullish' else
                         'Bearish crossover' if technical_analysis.get('trend') == 'bearish' else 'Neutral'
            },
            'volume': {
                'status': technical_analysis.get('volume', 'normal'),
                'ratio': latest_data.get('volume_ratio', 1.0)
            },
            'volatility': {
                'level': technical_analysis.get('volatility', 'normal'),
                'value': f"{latest_data.get('volatility_20d', 0.02) * 100:.1f}%"
            }
        }
    
    def _identify_risks(self, latest_data: pd.Series, expected_return: float, 
                       model_confidence: float) -> List[str]:
        """Identify potential risks for the trade."""
        
        risks = []
        
        if model_confidence < 0.7:
            risks.append("Low model confidence - prediction may be unreliable")
        
        volatility = latest_data.get('volatility_20d', 0.02)
        if volatility > 0.04:
            risks.append("High volatility - large price swings possible")
        
        rsi = latest_data.get('rsi', 50)
        if rsi > 80:
            risks.append("Severely overbought conditions - pullback likely")
        elif rsi < 20:
            risks.append("Severely oversold conditions - bounce possible but risky")
        
        volume_ratio = latest_data.get('volume_ratio', 1.0)
        if volume_ratio < 0.5:
            risks.append("Low volume - limited liquidity and conviction")
        
        if abs(expected_return) > 0.1:
            risks.append("Large expected move - higher probability of model error")
        
        if len(risks) == 0:
            risks.append("Normal market conditions")
        
        return risks
    
    def _generate_rationale(self, signal: Signal, expected_return: float,
                          technical_analysis: Dict[str, Any], model_confidence: float) -> str:
        """Generate human-readable rationale for the trading signal."""
        
        direction = "upward" if expected_return > 0 else "downward"
        magnitude = "strong" if abs(expected_return) > 0.05 else "moderate"
        confidence_desc = "high" if model_confidence > 0.8 else "moderate" if model_confidence > 0.6 else "low"
        
        rationale = f"Model predicts {magnitude} {direction} movement ({expected_return:.1%}) with {confidence_desc} confidence ({model_confidence:.0%}). "
        
        technical_strength = technical_analysis.get('strength', 0.5)
        if technical_strength > 0.7:
            rationale += "Technical indicators strongly support this direction. "
        elif technical_strength < 0.4:
            rationale += "Technical indicators show mixed signals. "
        else:
            rationale += "Technical indicators are neutral. "
        
        trend = technical_analysis.get('trend', 'neutral')
        if trend != 'neutral':
            rationale += f"MACD shows {trend} trend. "
        
        momentum = technical_analysis.get('momentum', 'neutral')
        if momentum in ['overbought', 'oversold']:
            rationale += f"RSI indicates {momentum} conditions. "
        
        return rationale.strip()


def format_trading_recommendation(recommendation: TradingRecommendation, 
                                ticker: str, model_type: str, horizon: str) -> str:
    # just format the trading rec for display, nothing fancy
    
    
    output = f"""
=== TRADING SIGNALS: {ticker.upper()} ({horizon} horizon) ===
MODEL: {model_type.upper()} | CONFIDENCE: {recommendation.confidence:.0%}

PREDICTION ANALYSIS:
Current Price: ${recommendation.current_price:.2f}
Target Price:  ${recommendation.target_price:.2f} ({recommendation.expected_return:+.1%})
Confidence Band: ${recommendation.current_price * 0.98:.2f} - ${recommendation.current_price * 1.02:.2f}

TRADING RECOMMENDATION:
Signal: {recommendation.signal.value}
Entry: ${recommendation.entry_price:.2f}
Target: ${recommendation.target_price:.2f} ({recommendation.expected_return:+.1%} gain)
Stop Loss: ${recommendation.stop_loss:.2f} ({((recommendation.stop_loss - recommendation.current_price) / recommendation.current_price):+.1%})
Position Size: {recommendation.position_size_pct:.1%} of portfolio ({recommendation.risk_level.value.lower()} risk)
Risk/Reward: 1:{recommendation.risk_reward_ratio:.1f}

- RSI: {recommendation.supporting_indicators['rsi']['value']:.0f} ({recommendation.supporting_indicators['rsi']['interpretation']})
- MACD: {recommendation.supporting_indicators['macd']['signal']}
- Volatility: {recommendation.supporting_indicators['volatility']['value']} ({recommendation.supporting_indicators['volatility']['level']})
- Volume: {recommendation.supporting_indicators['volume']['status'].replace('_', ' ').title()} ({recommendation.supporting_indicators['volume']['ratio']:.1f}x avg)

RISK ASSESSMENT:
- Market Risk: {recommendation.risk_level.value}
- Expected Holding Period: {recommendation.holding_period_days} days
- Key Risks: {'; '.join(recommendation.risks)}

RATIONALE:
{recommendation.rationale}
"""
    
    return output.strip()