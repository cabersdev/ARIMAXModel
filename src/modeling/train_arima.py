# train_arima.py
import time
import joblib
import logging
import pandas as pd
import traceback
import debugpy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator

# Configurazione logger avanzata
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ARIMATrainer:
    """Pipeline di addestramento continuo con logging avanzato"""
    
    def __init__(self, config_path: str = 'configs/parameters_config.yaml'):
        self._start_debugger()  # Attiva il debugger remoto
        self.model = self._load_model(config_path)
        self.config = self.model.config
        self.current_model_version = None
        self._preflight_check()
        self._init_metrics()
        logger.info("Trainer inizializzato con successo")

    def _start_debugger(self):
        """Attiva il debugger remoto su porta 5678"""
        debugpy.listen(5678)
        logger.info("Debugger in attesa di connessione su porta 5678...")
        debugpy.wait_for_client()
        logger.info("Debugger connesso")

    def _load_model(self, config_path: str) -> BaseEstimator:
        """Carica il modello con logging diagnostico"""
        try:
            from .model import AdaptiveARIMAX
            logger.debug(f"Caricamento modello da {config_path}")
            return AdaptiveARIMAX(config_path)
        except ImportError as e:
            logger.critical("Errore nell'importazione del modello", exc_info=True)
            raise
        except Exception as e:
            logger.error("Fallimento nel caricamento del modello", exc_info=True)
            raise

    def _preflight_check(self):
        """Verifica completa della configurazione prima dell'esecuzione"""
        logger.info("Avvio preflight check...")
        required_checks = [
            ('data_processing.rolling_window', int),
            ('features.technical_indicators.rsi.periods', list),
            ('backtesting.initial_capital', (int, float)),
            ('volatility.garch.order', list)
        ]
        
        for path, expected_type in required_checks:
            keys = path.split('.')
            value = self.config
            try:
                for key in keys:
                    value = value[key]
                if not isinstance(value, expected_type):
                    logger.critical(f"Tipo errato per {path}: atteso {expected_type}, ottenuto {type(value)}")
                    raise TypeError(f"{path} deve essere {expected_type}")
                logger.debug(f"Check superato per {path}")
            except KeyError:
                logger.critical(f"Configurazione mancante: {path}")
                raise
        logger.info("Preflight check completato con successo")

    def _init_metrics(self):
        """Inizializza le metriche di performance"""
        self.metrics = {
            'training_count': 0,
            'last_success': None,
            'errors': []
        }

    def continuous_training(self) -> None:
        """Loop principale di addestramento con resilienza"""
        logger.info("Avvio pipeline di addestramento continuo")
        
        while True:
            try:
                cycle_start = datetime.now()
                logger.info(f"Ciclo di training iniziato alle {cycle_start}")
                
                raw_data = self._fetch_data_with_retry()
                if self._validate_data(raw_data):
                    processed_data = self._process_streaming_data(raw_data)
                    self._train_and_validate(processed_data)
                    self._generate_and_send_signals(processed_data)
                
                sleep_time = self.config['training']['interval']
                logger.info(f"Ciclo completato. Ripresa tra {sleep_time}s...")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Interruzione manuale ricevuta")
                break
                
            except Exception as e:
                self._handle_critical_error(e)

    def _fetch_data_with_retry(self, max_retries: int = 3) -> pd.DataFrame:
        """Recupera dati con meccanismo di retry"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"Tentativo {attempt+1} di fetch dati")
                from src.data_pipelines import fetch_data
                data = fetch_data.get_live_data()
                
                if not data.empty:
                    logger.info(f"Dati ricevuti: {data.shape[0]} righe")
                    return data
                    
                logger.warning(f"Tentativo {attempt+1}: Dati ricevuti vuoti")
                time.sleep(5)
                
            except Exception as e:
                logger.warning(f"Tentativo {attempt+1} fallito: {str(e)}")
                time.sleep(10)
        
        logger.error("Fallito il recupero dati dopo %d tentativi", max_retries)
        return pd.DataFrame()

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validazione struttura dati"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            logger.error("Dati mancanti di colonne essenziali")
            return False
        if data.isnull().mean().max() > 0.5:
            logger.error("Troppi valori mancanti nei dati")
            return False
        return True

    def _process_streaming_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Pulizia e preparazione dati in tempo reale"""
        logger.debug("Elaborazione dati in corso...")
        window_size = self.config['data_processing']['rolling_window']
        processed_data = raw_data.iloc[-window_size:] if len(raw_data) > window_size else raw_data
        
        logger.debug(f"Dati elaborati: {processed_data.shape[0]} righe")
        logger.trace("Sample dati:\n" + str(processed_data.tail(3)))  # Richiede livello TRACE
        
        return processed_data

    def _train_and_validate(self, data: pd.DataFrame) -> None:
        """Pipeline completa addestramento e validazione"""
        try:
            self.metrics['training_count'] += 1
            logger.info(f"Inizio training #{self.metrics['training_count']}")
            
            self.model.fit(data)
            logger.info("Training completato con successo")
            
            logger.debug("Avvio validazione...")
            from src.backtesting import BacktestExecutor
            backtester = BacktestExecutor(self.config)
            results = backtester.run_backtest(data)
            
            if self._validate_performance(results):
                self._deploy_new_model()
                self.metrics['last_success'] = datetime.now()
            
        except Exception as e:
            logger.error("Errore training/validazione:\n%s", traceback.format_exc())
            self.metrics['errors'].append({
                'timestamp': datetime.now(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            raise

    def _validate_performance(self, results: Dict) -> bool:
        """Valida le metriche di performance"""
        min_sharpe = self.config['backtesting']['min_sharpe']
        max_dd = self.config['backtesting']['max_drawdown']
        
        logger.debug(f"Metriche performance: {results['performance']}")
        
        if results['performance']['sharpe'] > min_sharpe:
            logger.info(f"Shapre Ratio {results['performance']['sharpe']} sopra soglia")
        else:
            logger.warning(f"Shapre Ratio {results['performance']['sharpe']} sotto soglia")
            
        return (
            results['performance']['sharpe'] > min_sharpe and
            results['performance']['max_drawdown'] < max_dd
        )

    def _deploy_new_model(self) -> None:
        """Deploy sicuro del nuovo modello"""
        try:
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = Path(f"models/arima_{version}.pkl")
            model_path.parent.mkdir(exist_ok=True)
            
            joblib.dump(self.model, model_path)
            self.current_model_version = version
            logger.info(f"Deploy nuovo modello: {model_path.name}")
            
        except Exception as e:
            logger.error("Deploy fallito:\n%s", traceback.format_exc())
            raise

    def _generate_and_send_signals(self, data: pd.DataFrame) -> None:
        """Genera e invia i segnali di trading"""
        try:
            logger.debug("Generazione segnali...")
            predictions = self.model.predict(data)
            signals = self.model.generate_signals(predictions)
            
            logger.debug(f"Segnali generati: {signals[-5:]}")
            from src.data_pipelines import fetch_data
            if hasattr(fetch_data, 'push_signals_to_api'):
                fetch_data.push_signals_to_api(signals)
                logger.info("Segnali inviati con successo")
            else:
                logger.warning("Funzione di invio segnali non disponibile")
                
        except Exception as e:
            logger.error("Errore generazione segnali:\n%s", traceback.format_exc())
            raise

    def _handle_critical_error(self, error: Exception) -> None:
        """Gestione errori critici con fallback"""
        logger.critical("Errore critico nello stack:", exc_info=True)
        self._send_alert_to_monitoring(error)
        
        retry_interval = self.config['training'].get('error_retry_interval', 600)
        logger.info(f"Riprova tra {retry_interval} secondi...")
        time.sleep(retry_interval)

    def _send_alert_to_monitoring(self, error: Exception):
        """Invia alert al sistema di monitoring"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "stack_trace": traceback.format_exc()
        }
        logger.debug(f"Inviato alert: {error_info}")

if __name__ == "__main__":
    try:
        logger.info("Avvio applicazione")
        trainer = ARIMATrainer()
        trainer.continuous_training()
    except KeyboardInterrupt:
        logger.info("Interruzione manuale dell'addestramento")
    except Exception as e:
        logger.critical("Errore fatale nell'applicazione principale", exc_info=True)
    finally:
        logger.info("Applicazione terminata")