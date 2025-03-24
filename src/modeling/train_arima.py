import os
import sys
import time
import joblib
import logging
import pandas as pd
import traceback
import debugpy
import threading
import socket
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from ipaddress import ip_address


from src.data_pipelines import fetch_data

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.DEBUG,
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
        self.args = self._parse_cli_args()
        self._start_debugger()
        self.model = self._load_model(config_path)
        self.config = self.model.config
        self.current_model_version = None
        self._preflight_check()
        self._init_metrics()
        logger.info("Trainer inizializzato con successo")

    def _parse_cli_args(self):
        """Configura parser argomenti CLI"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug-host', default='127.0.0.1', 
                          help="Indirizzo di ascolto debugger")
        parser.add_argument('--debug-port', type=int, default=5678,
                          help="Porta debugger remoto")
        parser.add_argument('--strict-debug', action='store_true',
                          help="Blocca esecuzione su errori debugger")
        parser.add_argument('--no-debug', action='store_true',
                          help="Disabilita completamente il debugger")
        return parser.parse_args()

    def _start_debugger(self):
        """Configurazione debugger"""
        if self.args.no_debug:
            logger.info("Modalità debug disabilitata")
            return

        try:
            debugpy.listen((self.args.debug_host, self.args.debug_port))
            logger.info(f"Debugger in ascolto su {self.args.debug_host}:{self.args.debug_port}")

            def timeout_handler():
                time.sleep(30)
                if not debugpy.is_client_connected():
                    logger.warning("Timeout connessione debugger")

            threading.Thread(target=timeout_handler, daemon=True).start()
            
            logger.info("In attesa connessione debugger (CTRL+C per saltare)...")
            debugpy.wait_for_client()
            logger.info("Debugger connesso con successo!")

        except Exception as e:
            logger.error(f"Errore debugger: {str(e)}", exc_info=True)
            if self.args.strict_debug:
                raise

    def _load_model(self, config_path: str) -> BaseEstimator:
        """Carica il modello con logging diagnostico"""
        try:
            from .model import AdaptiveARIMAX
            logger.debug(f"Caricamento modello da {config_path}")
            return AdaptiveARIMAX(config_path)
        except ImportError as e:
            logger.critical("Errore importazione modello", exc_info=True)
            raise
        except Exception as e:
            logger.error("Fallimento caricamento modello", exc_info=True)
            raise

    def _preflight_check(self):
        """Verifica completa della configurazione"""
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
                    self._raise_type_error(path, expected_type, value)
                logger.debug(f"Check superato: {path}")
            except KeyError:
                self._raise_missing_config_error(path)

        logger.info("Preflight check completato")

    def _raise_type_error(self, path: str, expected: type, actual: type):
        err_msg = f"Tipo errato per {path}: Atteso {expected.__name__}, ottenuto {type(actual).__name__}"
        logger.critical(err_msg)
        raise TypeError(err_msg)

    def _raise_missing_config_error(self, path: str):
        err_msg = f"Configurazione mancante: {path}"
        logger.critical(err_msg)
        raise KeyError(err_msg)

    def _init_metrics(self):
        """Inizializza le metriche di performance"""
        self.metrics = {
            'training_count': 0,
            'last_success': None,
            'errors': [],
            'start_time': datetime.now()
        }

    def continuous_training(self) -> None:
        """Loop principale di addestramento"""
        logger.info("Avvio pipeline addestramento continuo")
        
        try:
            while True:
                cycle_start = datetime.now()
                self._training_cycle(cycle_start)
        except KeyboardInterrupt:
            logger.info("Interruzione manuale ricevuta")
        finally:
            self._shutdown()

    def _training_cycle(self, cycle_start: datetime):
        """Gestione singolo ciclo di training"""
        try:
            logger.info(f"Ciclo iniziato alle {cycle_start}")
            
            raw_data = self._fetch_data_with_retry()
            logger.info('Dati ricevuti: %s righe', raw_data.shape[0])
            if self._validate_data(raw_data):
                processed_data = self._process_streaming_data(raw_data)
                logger.info('Dati elaborati: %s righe', processed_data.shape[0])
                self._train_and_validate(processed_data)
                self._generate_and_send_signals(processed_data)
            
            self._sleep_until_next_cycle()
            
        except Exception as e:
            self._handle_critical_error(e)

    def _sleep_until_next_cycle(self):
        sleep_time = self.config['training']['interval']
        logger.info(f"Pausa di {sleep_time}s...")
        time.sleep(sleep_time)

    def _fetch_data_with_retry(self, max_retries: int = 3) -> pd.DataFrame:
        """Recupero dati con validazione integrata"""
        for attempt in range(1, max_retries+1):
            try:
                data = fetch_data.get_live_data()

                if not isinstance(data, pd.DataFrame):
                    logger.error("Dati non validi: %s", type(data))
                    continue
                
                if data.empty:
                    logger.warning(f"Tentativo {attempt}: Dati vuoti da API")
                    continue
                    
                if data['Close'].isnull().all():
                    logger.error("Dati corrotti: valori Close tutti nulli")
                    continue
                    
                return data
                
            except Exception as e:
                logger.warning(f"Tentativo {attempt} fallito: {str(e)}")
                time.sleep(10)
        
        raise RuntimeError("Recupero dati fallito")

    def _fetch_live_data(self) -> pd.DataFrame:
        """Fetch dati live con import assoluto"""
        try:
            from src.data_pipelines import fetch_data
            return fetch_data.get_live_data()
        except ImportError as e:
            logger.error("Modulo fetch_data non trovato", exc_info=True)
            raise
        except AttributeError as e:
            logger.error("Funzione get_live_data non definita", exc_info=True)
            raise

    def _handle_empty_data(self, attempt: int):
        logger.warning(f"Tentativo {attempt}: Dati ricevuti vuoti")
        time.sleep(5)

    def _handle_fetch_error(self, attempt: int, error: Exception):
        logger.warning(f"Tentativo {attempt} fallito: {str(error)}")
        time.sleep(10)

    def _fail_fetch(self, max_retries: int) -> pd.DataFrame:
        logger.error(f"Fallito recupero dati dopo {max_retries} tentativi")
        return pd.DataFrame()

    # Modifica nel metodo _validate_data
    def _validate_data(self, data: pd.DataFrame) -> bool:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Controllo colonne con conversione esplicita a lista
        if not all(col in list(data.columns) for col in required_columns):
            logger.error("Colonne mancanti: %s", required_columns)
            return False
        
        # Controllo valori nulli con conversione esplicita a float
        try:
            null_percentage = data.isnull().mean().max().item()  # Converti a float
            if null_percentage > 0.5:
                logger.error("Troppi valori mancanti (>50%%)")
                return False
        except AttributeError:
            logger.error("Formato dati non valido per il controllo valori nulli")
            return False
        
        return True

    def _process_streaming_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(raw_data, pd.DataFrame):
            raise TypeError(f"Tipo dati non valido: {type(raw_data)}")

        window_size = self.config['data_processing']['rolling_window']
        processed_data = raw_data.iloc[-window_size:] if len(raw_data) > window_size else raw_data
        
        logger.debug(f"Dati elaborati: {processed_data.shape[0]} righe")
        logger.debug("Sample dati:\n%s", processed_data.tail(3))

        logger.info(f"Statistiche dati:\n{processed_data.describe()}")
        logger.debug(f"Valori nulli:\n{processed_data.isnull().sum()}")
        
        return processed_data.dropna()

    def _train_and_validate(self, data: pd.DataFrame) -> None:
        try:
            self.metrics['training_count'] += 1
            logger.info(f"Training #{self.metrics['training_count']}")
            
            self.model.fit(data)
            logger.info("Training completato")
            
            self._run_backtest(data)
            
        except Exception as e:
            self._log_training_error(e)
            raise

    def _log_training_error(self, error: Exception):
        logger.error("Errore durante il training:", exc_info=True)
        self.metrics['errors'].append({
            'timestamp': datetime.now(),
            'error': str(error),
            'traceback': traceback.format_exc()
        })

    def _run_backtest(self, data: pd.DataFrame):
        logger.debug("Avvio backtest...")
        from src.backtesting import BacktestExecutor
        backtester = BacktestExecutor(self.config)
        results = backtester.run_backtest(data)
        
        if self._validate_performance(results):
            self._deploy_new_model()
            self.metrics['last_success'] = datetime.now()

    def _validate_performance(self, results: Dict) -> bool:
        min_sharpe = self.config['backtesting']['min_sharpe']
        max_dd = self.config['backtesting']['max_drawdown']
        perf = results['performance']
        
        logger.debug(f"Metriche performance: {perf}")
        
        if perf['sharpe'] > min_sharpe:
            logger.info(f"Sharpe Ratio {perf['sharpe']:.2f} > {min_sharpe}")
        else:
            logger.warning(f"Sharpe Ratio {perf['sharpe']:.2f} < {min_sharpe}")
            
        return perf['sharpe'] > min_sharpe and perf['max_drawdown'] < max_dd

    def _deploy_new_model(self) -> None:
        try:
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            model_path = Path(f"models/arima_{version}.pkl")
            model_path.parent.mkdir(exist_ok=True)
            
            joblib.dump(self.model, model_path)
            self.current_model_version = version
            logger.info(f"Deploy modello: {model_path.name}")
            
        except Exception as e:
            logger.error("Errore deploy modello", exc_info=True)
            raise

    def _generate_and_send_signals(self, data: pd.DataFrame) -> None:
        try:
            logger.debug("Generazione segnali...")
            predictions = self.model.predict(data)
            signals = self.model.generate_signals(predictions)
            
            logger.debug(f"Ultimi segnali: {signals[-5:]}")
            self._send_to_api(signals)
            
        except Exception as e:
            logger.error("Errore generazione segnali", exc_info=True)
            raise

    def _send_to_api(self, signals):
        from src.data_pipelines import fetch_data
        if hasattr(fetch_data, 'push_signals_to_api'):
            fetch_data.push_signals_to_api(signals)
            logger.info("Segnali inviati con successo")
        else:
            logger.warning("Funzione di invio segnali non disponibile")

    def _handle_critical_error(self, error: Exception) -> None:
        logger.critical("Errore critico:", exc_info=True)
        self.metrics['errors'].append({
            'timestamp': datetime.now(),
            'error': str(error),
            'traceback': traceback.format_exc()
        })
        self._retry_after_error()

    def _retry_after_error(self):
        retry_interval = self.config['training'].get('error_retry_interval', 600)
        logger.info(f"Riprova tra {retry_interval}s...")
        time.sleep(retry_interval)

    def _shutdown(self):
        logger.info("Applicazione terminata")
        logger.info("Riepilogo metriche:")
        logger.info(f"  - Cicli completati: {self.metrics['training_count']}")
        logger.info(f"  - Ultimo successo: {self.metrics['last_success']}")
        logger.info(f"  - Errori registrati: {len(self.metrics['errors'])}")

if __name__ == "__main__":
    try:
        logger.info("Avvio applicazione")
        ARIMATrainer().continuous_training()
    except Exception as e:
        logger.critical("Errore fatale", exc_info=True)
    finally:
        logger.info("Chiusura applicazione")