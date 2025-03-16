import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model
from sklearn.base import BaseEstimator
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

class AdaptiveARIMAX(BaseEstimator):
    """ARIMA avanzato con funzionalità complete"""
    
    def __init__(self, config_path='configs/parameters_config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self._init_hyperparameters()
        self._validate_config()
        
    def _init_hyperparameters(self):
        """Carica e processa i parametri di configurazione"""
        arima_cfg = self.config['arima']
        self.order = arima_cfg['default_order']
        self.seasonal_order = arima_cfg['seasonal_order']
        self.exog_features = self.config['features']['exogenous']
        self.volatility_cfg = self.config['volatility']
        self.auto_tuning = arima_cfg['auto_tuning']
        self.regularization = self.config.get('regularization', {})
        
    def _validate_config(self):
        """Validazione parametri di configurazione"""
        required_sections = ['arima', 'features', 'volatility', 'data_processing']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Sezione mancante nel config: {section}")

    def _determine_differencing(self, series):
        """Calcola automaticamente l'ordine di differenziazione (d)"""
        max_d = self.auto_tuning.get('max_d', 2)
        for d in range(max_d + 1):
            if d == 0:
                result = adfuller(series.dropna())
            else:
                result = adfuller(series.diff(d).dropna())
            if result[1] < 0.05:
                return d
        raise ValueError("Serie non stazionaria dopo differenziazione massima")

    def _preprocess_data(self, data):
        """Pipeline completa di preprocessing"""
        self.d = self._determine_differencing(data[self.config['features']['target']])
        data = data.copy().diff(self.d).dropna()
        
        data = self._create_features(data)
        
        data = self._handle_missing_values(data)
        
        if self.config['data_processing']['scaling']['enabled']:
            data = self._scale_features(data)
            
        return data

    def _create_features(self, data):
        """Generazione features esogene e indicatori tecnici"""
        target = self.config['features']['target']
        tech_cfg = self.config['features']['technical_indicators']
        
        data['returns'] = data[target].pct_change()
        data['volatility'] = data['returns'].rolling(
            self.volatility_cfg['window']).std()
            
        for window in tech_cfg['ma_windows']:
            data[f'MA_{window}'] = data[target].rolling(window).mean()
            
        data['RSI'] = self._calculate_rsi(data[target], tech_cfg['rsi_period'])
        data = self._add_macd(data, tech_cfg)
        
        return data.dropna()

    def _calculate_rsi(self, series, period):
        """Calcola Relative Strength Index"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _add_macd(self, data, cfg):
        """Aggiunge MACD alla feature set"""
        exp1 = data[self.config['features']['target']].ewm(
            span=cfg['macd_fast'], adjust=False).mean()
        exp2 = data[self.config['features']['target']].ewm(
            span=cfg['macd_slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=cfg['macd_signal'], adjust=False).mean()
        data['MACD'] = macd
        data['MACD_Signal'] = signal
        return data

    def fit(self, X, y=None):
        """Addestramento completo del modello con ottimizzazione automatica"""
        data = self._preprocess_data(X)
        exog = data[self.exog_features]
        
        self.model_ = auto_arima(
            data[self.config['features']['target']],
            exogenous=exog,
            seasonal=self.config['arima']['seasonal'],
            m=self.config['arima']['seasonality'],
            max_p=self.auto_tuning['max_p'],
            max_q=self.auto_tuning['max_q'],
            max_d=self.d,
            stepwise=self.auto_tuning['stepwise'],
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            **self._get_regularization_params()
        )
        
        self._fit_volatility_model(data)
        
        self._residual_diagnostics()
        
        return self

    def _get_regularization_params(self):
        """Parametri di regolarizzazione per SARIMAX"""
        return {
            'with_intercept': self.regularization.get('intercept', True),
            'trend': self.regularization.get('trend', 'c'),
            'enforce_stationarity': self.regularization.get('enforce_stationarity', False),
            'enforce_invertibility': self.regularization.get('enforce_invertibility', False)
        }

    def _fit_volatility_model(self, data):
        """Addestramento modello GARCH per intervalli dinamici"""
        self.vol_model_ = arch_model(
            data['returns'],
            vol='GARCH',
            p=self.volatility_cfg['garch_order'][0],
            q=self.volatility_cfg['garch_order'][1]
        ).fit(disp='off')

    def _residual_diagnostics(self):
        """Esegui diagnostica completa dei residui"""
        residuals = self.model_.resid()
        
        lb_test = acorr_ljungbox(residuals, lags=[20], return_df=True)
        self.lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        
        self.arch_pvalue = het_arch(residuals)[1]
        
        self.residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skew': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }

    def predict(self, X, horizon=30):
        """Genera previsioni con intervalli dinamici"""
        data = self._preprocess_data(X)
        exog_future = data[self.exog_features].iloc[-horizon:]
        
        forecast, conf_int = self.model_.predict(
            n_periods=horizon,
            exogenous=exog_future,
            return_conf_int=True
        )
        
        vol_forecast = self.vol_model_.forecast(horizon=horizon)
        risk_multiplier = self.volatility_cfg['risk_multiplier']
        
        adjusted_ci = (
            forecast - risk_multiplier * vol_forecast.variance.values**0.5,
            forecast + risk_multiplier * vol_forecast.variance.values**0.5
        )
        
        return {
            'forecast': forecast,
            'confidence_interval': adjusted_ci,
            'volatility': vol_forecast.variance.values**0.5,
            'model_metadata': self._get_model_metadata()
        }

    def _get_model_metadata(self):
        """Metadati diagnostici del modello"""
        return {
            'order': self.model_.order,
            'seasonal_order': self.model_.seasonal_order,
            'residual_pvalues': {
                'ljung_box': self.lb_pvalue,
                'arch_test': self.arch_pvalue
            },
            'residual_stats': self.residual_stats
        }

    def generate_signals(self, predictions):
        """Genera segnali trading con threshold dinamico"""
        signals = []
        thresholds = self.volatility_cfg['trading_signals']
        
        for fcst, vol in zip(predictions['forecast'], predictions['volatility']):
            z_score = (fcst - np.mean(predictions['forecast'])) / vol
            
            if z_score > thresholds['sell_threshold']:
                signals.append('SELL')
            elif z_score < thresholds['buy_threshold']:
                signals.append('BUY')
            else:
                signals.append('HOLD')
                
        return signals