import argparse
import threading
import time
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model
from sklearn.base import BaseEstimator
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import yaml
import sys
import logging
import debugpy
import os
import socket 
from ipaddress import ip_address
from typing import List, Tuple, Dict, Any
from datetime import datetime 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveARIMAX(BaseEstimator):
    """ARIMA avanzato con funzionalità complete"""
    
    def __init__(self, config_path: str = 'configs/parameters_config.yaml'):
        logger.info('Inizializzazione modello ARIMAX')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        logger.info('Configurazione caricata con successo')
            
        self._init_hyperparameters()
        self._validate_config()
        logger.info('Configurazione validata con successo')

    def _parse_cli_args(self):
        ''' Configuara parser argomenti di riga di comandi'''
        parser = argparse.ArgumentParser(description='Adaptive ARIMAX model')
        parser.add_argument('--debug-host',  default='127.0.0.1', 
                            help='Indirizzo di ascolto debugger')
        parser.add_argument('--debug-port',  default=5678, 
                            help='Porta debugger remoto')
        parser.add_argument('--strict-debug', action='store_true',
                            help='Blocca esecuzioni su errire debbuger')
        parser.add_argument('--no-debug', action='store_true',
                            help='Disabilita completamente il debugger')
        return parser.parse_args()
    
    def _start_debugger(self):
        '''Configurazione debugger'''
        if self.args.no_debug:
            return

        try: 
            debugpy.listen((self.args.debug_host, self.args.debug_port))
            logger.info(f'Debugger in ascolto su {self.args.debug_host}:{self.args.debug_port}')

            def timeout_handler():
                time.sleep(30)
                if not debugpy.is_client_connected():
                    logger.warning("Timeout connessione debugger")

            threading.Thread(target=timeout_handler, daemon=True).start()
            
            logger.info('In attesa di connessione debugger...')
            debugpy.wait_for_client()
            logger.info('Debugger connesso con successo')
        
        except Exception as e:
            logger.error(f'Errore debugger: {str(e)}', exc_info=True)
            if self.args.strict_debug:
                logger.error('Uscita per errore debugger')
                raise
        
    def _init_hyperparameters(self):
        """Carica e processa i parametri di configurazione"""

        logger.info('Inizializzazione iperparametri')
        arima_cfg = self.config['arima']
        self.order = arima_cfg['default_order']
        self.seasonal_order = arima_cfg['seasonal_order']
        self.exog_features = self.config['features']['exogenous']
        self.volatility_cfg = self.config['volatility']
        self.auto_tuning = arima_cfg['auto_tuning']
        self.regularization = self.config.get('regularization', {})
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gestione valori mancanti secondo configurazione"""
        logger.info('Gestione valori mancanti')
        missing_cfg = self.config['data_processing']['missing_values']
        data_corrected = data.copy()
        
        # Determina il metodo di interpolazione da usare
        method = missing_cfg['interpolate_method']
        if method == 'time' and not pd.api.types.is_datetime64_any_dtype(data_corrected.index):
            try:
                data_corrected.index = pd.to_datetime(data_corrected.index)
            except Exception as e:
                logger.error(f"Errore conversione indice in DatetimeIndex: {str(e)}")
                # Se la conversione fallisce, si passa a un metodo alternativo (es. 'linear')
                method = 'linear'
                # (Assicurati di importare logging o usa un logger a tua scelta)
                print("Impossibile convertire l'indice in DatetimeIndex, uso l'interpolazione lineare.")
        
        for col in data_corrected.columns:
            # Applica l'interpolazione solo alle colonne numeriche
            if pd.api.types.is_numeric_dtype(data_corrected[col]):
                data_corrected[col] = data_corrected[col].interpolate(
                    method=method,
                    limit=missing_cfg['max_consecutive_nan'],
                    limit_area='inside'
                )
            # Applica forward fill e backward fill per qualsiasi colonna (numerica o meno)
            data_corrected[col] = data_corrected[col].ffill(limit=3).bfill(limit=3)
        
        return data_corrected

    def _validate_config(self):
        """Validazione completa dei parametri di configurazione"""
        logger.info('Validazione configurazione')
        # Controllo della presenza delle sezioni principali
        required_sections = ['arima', 'features', 'volatility', 'data_processing']
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Sezione mancante nel config: {section}")
                raise ValueError(f"Sezione mancante nel config: {section}")

        # Controllo specifico per i parametri relativi ai missing values
        required_missing = ['strategy', 'rolling_window', 'max_consecutive_nan']
        for key in required_missing:
            if key not in self.config['data_processing']['missing_values']:
                logger.error(f"Parametro mancante per missing values: {key}")
                raise KeyError(f"Parametro mancante per missing values: {key}")

    def _determine_differencing(self, series: pd.Series) -> int:
        """Calcola differenziazione con fallback sicuro"""
        try:
            for d in range(self.config['arima']['auto_tuning']['max_d'] + 1):
                p_value = adfuller(series.diff(d).dropna())[1]
                if p_value < 0.05:
                    logger.info(f"Ordine differenziazione determinato: d={d}")
                    return d
            raise ValueError("Serie non stazionaria dopo differenziazione massima")
        
        except Exception as e:
            logger.error("Errore nel calcolo differenziazione: %s", str(e))
            return 1  # Fallback a differenziazione di ordine 1

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline di preprocessing con inizializzazione corretta di d"""
        # 1. Calcola l'ordine di differenziazione
        self.d = self._determine_differencing(data[self.config['features']['target']])
        
        # 2. Applica differenziazione
        data_diff = data.copy().diff(self.d).dropna()
        
        # 3. Gestione valori mancanti
        data_clean = self._handle_missing_values(data_diff)
        
        # 4. Feature engineering
        data_feat = self._create_features(data_clean)
        
        # 5. Validazione finale
        if data_feat.empty:
            logger.error("DataFrame finale vuoto dopo preprocessing")
            raise ValueError("Dati non validi per il training")
        
        return data_feat

    def _validate_config(self):
        """Validazione parametri di configurazione"""
        logger.info('Validazione configurazione')
        required_missing = ['strategy', 'rolling_window', 'max_consecutive_nan']
        for key in required_missing:
            if key not in self.config['data_processing']['missing_values']:
                logger.error(f"Parametro mancante per missing values: {key}")
                raise KeyError(f"Parametro mancante per missing values: {key}")

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applica scaling alle features esogene presenti nel DataFrame"""
        logger.info('Scaling features esogene')
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        
        # Filtra le colonne esogene che esistono nel DataFrame
        exog_cols = [col for col in self.exog_features if col in data.columns]
        if not exog_cols:
            logger.error('Nessuna feature esogena trovata per lo scaling')
            raise ValueError("Nessuna feature esogena trovata per lo scaling.")
        
        if data[exog_cols].empty or data[exog_cols].shape[0] == 0:
            print("Warning: DataFrame vuoto per le colonne esogene, salto scaling.")
            return data
        
        scaled_array = scaler.fit_transform(data[exog_cols])
        df_scaled = pd.DataFrame(scaled_array, columns=exog_cols, index=data.index)
        
        # Mantieni le altre colonne non scalate (se necessario)
        for col in data.columns:
            if col not in exog_cols:
                df_scaled[col] = data[col]
        
        return df_scaled


        scaled_array = scaler.fit_transform(data[exog_cols])
        df_scaled = pd.DataFrame(scaled_array, columns=exog_cols, index=data.index)
        
        # Mantieni le altre colonne non scalate (se necessario)
        for col in data.columns:
            if col not in exog_cols:
                df_scaled[col] = data[col]
        
        return df_scaled

    def _create_features(self, data):
        """Generazione features esogene e indicatori tecnici in base alla configurazione"""
        logger.info('Creazione features esogene e indicatori tecnici')
        target = self.config['features']['target']
        tech_cfg = self.config['features']['technical_indicators']

        # Calcola returns e volatility
        data['returns'] = data[target].pct_change()
        logger.info('Calcolati returns e volatility')
        data['volatility'] = data['returns'].rolling(self.volatility_cfg['window']).std()
        logger.info('Calcolati returns e volatility')

        # Calcola la Moving Average.
        # Se 20 non è presente nella lista dei windows, calcola esplicitamente MA_20
        if 20 not in tech_cfg['ma']['ma_windows']:
            data['MA_20'] = data[target].rolling(20).mean()
            logger.info('Calcolata MA_20')
        # Calcola le MA definite nella configurazione (eventualmente includerà MA_20 se già presente)
        for window in tech_cfg['ma']['ma_windows']:
            data[f'MA_{window}'] = data[target].rolling(window).mean()
            logger.info(f'Calcolata MA_{window}')

        # Calcola le EMA (se necessarie per ulteriori analisi, non previste nell'exogenous ma utili per il MACD)
        for ema_window in tech_cfg['ma']['ema_windows']:
            data[f'EMA_{ema_window}'] = data[target].ewm(span=ema_window, adjust=False).mean()
            logger.info(f'Calcolata EMA_{ema_window}')

        # Calcola RSI
        data['RSI'] = self._calculate_rsi(data[target], tech_cfg['rsi']['periods'][0])
        logger.info('RSI calcolato con successo')

        # Calcola MACD, MACD_Signal e MACD_hist
        data = self._add_macd(data, tech_cfg)
        logger.info('Indicatori tecnici calcolati con successo')

        # Calcola OBV
        if 'Volume' in data.columns:
            # OBV: somma cumulativa del volume moltiplicato per il segno della variazione del prezzo
            data['OBV'] = (np.sign(data[target].diff()) * data['Volume']).fillna(0).cumsum()
            logger.info(f'Calcolo OBV completato')
        else:
            logger.error("La colonna 'Volume' è necessaria per calcolare OBV.")
            raise KeyError("La colonna 'Volume' è necessaria per calcolare OBV.")

        # Calcola VWAP (Volume Weighted Average Price)
        if 'Volume' in data.columns:
            data['VWAP'] = (data[target] * data['Volume']).cumsum() / data['Volume'].cumsum()
            logger.info(f'Calcolo VWAP completato')
        else:
            logger.error("La colonna 'Volume' è necessaria per calcolare VWAP.")
            raise KeyError("La colonna 'Volume' è necessaria per calcolare VWAP.")

        return data.dropna()

    def _add_macd(self, data, cfg):
        """Aggiunge MACD, MACD_Signal e MACD_hist al feature set"""
        logger.info('Calcolo MACD')
        target = self.config['features']['target']
        exp1 = data[target].ewm(span=cfg['macd']['fast'], adjust=False).mean()
        exp2 = data[target].ewm(span=cfg['macd']['slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=cfg['macd']['signal'], adjust=False).mean()
        data['MACD'] = macd
        data['MACD_Signal'] = signal
        data['MACD_hist'] = macd - signal
        return data



    def _calculate_rsi(self, series, period):
        """Calcola Relative Strength Index"""
        logger.info('Calcolo RSI')
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _add_macd(self, data, cfg):
        """Aggiunge MACD alla feature set"""
        logger.info('Calcolo MACD')
        exp1 = data[self.config['features']['target']].ewm(
            span=cfg['macd']['fast'], adjust=False).mean()
        exp2 = data[self.config['features']['target']].ewm(
            span=cfg['macd']['slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=cfg['macd']['signal'], adjust=False).mean()
        data['MACD'] = macd
        data['MACD_Signal'] = signal
        return data

    def fit(self, X, y=None):
        """Addestramento completo del modello con ottimizzazione automatica"""
        
        if not hasattr(self, X, y=None):
            raise RuntimeError("Ordine di differenziazine non calcolato.")

        logger.info('Addestramento modello')
        data = self._preprocess_data(X)
        
        if data.empty:
            logger.error('DataFrame preprocessato vuoto')
            raise ValueError("Il DataFrame preprocessato è vuoto. Controlla i dati in input e il preprocessing.")
        
        # Seleziona solo le feature esogene presenti
        exog_cols = [col for col in self.exog_features if col in data.columns]
        if not exog_cols or data[exog_cols].empty:
            logger.error('Nessuna feature esogena valida trovata per il training')
            raise ValueError("Nessuna feature esogena valida trovata per il training.")
        exog = data[exog_cols]
        
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
        
        logger.info('Modello ARIMA addestrato con successo')

        self._fit_volatility_model(data)
        logger.info('Modello GARCH addestrato con successo')
        self._residual_diagnostics()
        logger.info('Diagnostica residui completata')
        
        return self


    def _get_regularization_params(self):
        """Parametri di regolarizzazione per SARIMAX"""
        logger.info('Parametri regolarizzazione')
        return {
            'with_intercept': self.regularization.get('intercept', True),
            'trend': self.regularization.get('trend', 'c'),
            'enforce_stationarity': self.regularization.get('enforce_stationarity', False),
            'enforce_invertibility': self.regularization.get('enforce_invertibility', False)
        }

    def _fit_volatility_model(self, data):
        """Addestramento modello GARCH per intervalli dinamici"""
        logger.info('Addestramento modello GARCH')
        self.vol_model_ = arch_model(
            data['returns'],
            vol='GARCH',
            p=self.volatility_cfg['garch_order'][0],
            q=self.volatility_cfg['garch_order'][1]
        ).fit(disp='off')

    def _residual_diagnostics(self):
        """Esegui diagnostica completa dei residui"""
        logger.info('Diagnostica residui')
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
        logger.info('Generazione previsioni')
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
        logger.info('Metadati modello')
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
        logger.info('Generazione segnali trading')
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