import time
import joblib
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from src.modeling.model import AdaptiveARIMAX

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMATrainer:
    """Pipeline di addestramento continuo con gestione errori avanzata"""
    
    def __init__(self, config_path: str = 'configs/parameters_config.yaml'):
        self.model = AdaptiveARIMAX(config_path)
        self.config = self.model.config
        self.current_model_version = None
        self._validate_config()

    def _load_model(self, config_path: str) -> Any:
        """Carica il modello con gestione errori"""
        try:
            from .model import AdaptiveARIMAX
            return AdaptiveARIMAX(config_path)
        except ImportError as e:
            logger.error(f"Errore nel caricamento del modello: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """Verifica la presenza di tutte le chiavi di configurazione richieste"""
        required_keys = {
            'data_processing': ['rolling_window'],
            'training': ['interval', 'error_retry_interval'],
            'backtesting': ['min_sharpe', 'max_drawdown']
        }
        
        for section, keys in required_keys.items():
            if section not in self.config:
                raise KeyError(f"Sezione mancante nel config: {section}")
            for key in keys:
                if key not in self.config[section]:
                    raise KeyError(f"Chiave {key} mancante nella sezione {section}")

    def continuous_training(self) -> None:
        """Loop principale di addestramento continuo con resilienza"""
        logger.info("Avvio pipeline di addestramento continuo")
        
        while True:
            try:
                raw_data = self._fetch_data_with_retry()
                if raw_data.empty:
                    continue
                    
                processed_data = self._process_streaming_data(raw_data)
                
                if self._needs_retraining(processed_data):
                    self._train_and_validate(processed_data)
                    
                self._generate_and_send_signals(processed_data)
                
                time.sleep(self.config['training']['interval'])
                
            except Exception as e:
                self._handle_critical_error(e)

    def _fetch_data_with_retry(self, max_retries: int = 3) -> pd.DataFrame:
        """Recupera dati con meccanismo di retry"""
        for attempt in range(max_retries):
            try:
                from src.data_pipelines import fetch_data
                data = fetch_data.get_live_data()
                if not data.empty:
                    return data
                logger.warning(f"Tentativo {attempt+1}: Dati ricevuti vuoti")
            except AttributeError:
                logger.error("Funzione get_live_data non disponibile in fetch_data")
                return pd.DataFrame()
            except Exception as e:
                logger.warning(f"Tentativo {attempt+1} fallito: {str(e)}")
                time.sleep(5)
        
        logger.error("Fallito il recupero dati dopo %d tentativi", max_retries)
        return pd.DataFrame()

    def _process_streaming_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Pulizia e preparazione dati in tempo reale"""
        window_size = self.config['data_processing']['rolling_window']
        return raw_data.iloc[-window_size:] if len(raw_data) > window_size else raw_data

    def _needs_retraining(self, data: pd.DataFrame) -> bool:
        """Determina la necessità di riaddestramento"""
        return not self.current_model_version or len(data) % 100 == 0

    def _train_and_validate(self, data: pd.DataFrame) -> None:
        """Pipeline completa addestramento e validazione"""
        try:
            self.model.fit(data)
            
            from src.backtesting import BacktestExecutor
            backtester = BacktestExecutor(self.config)
            results = backtester.run_backtest(data)
            
            if self._validate_performance(results):
                self._deploy_new_model()
                
        except Exception as e:
            logger.error(f"Errore durante training/validazione: {str(e)}")
            raise

    def _validate_performance(self, results: Dict) -> bool:
        """Valida le metriche di performance"""
        min_sharpe = self.config['backtesting']['min_sharpe']
        max_dd = self.config['backtesting']['max_drawdown']
        
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
            logger.error(f"Errore durante il deploy: {str(e)}")
            raise

    def _generate_and_send_signals(self, data: pd.DataFrame) -> None:
        """Genera e invia i segnali di trading"""
        try:
            predictions = self.model.predict(data)
            signals = self.model.generate_signals(predictions)
            
            from src.data_pipelines import fetch_data
            if hasattr(fetch_data, 'push_signals_to_api'):
                fetch_data.push_signals_to_api(signals)
            else:
                logger.warning("Funzione push_signals_to_api non disponibile")
                
        except Exception as e:
            logger.error(f"Errore generazione segnali: {str(e)}")
            raise

    def _handle_critical_error(self, error: Exception) -> None:
        """Gestione errori critici con fallback"""
        logger.error(f"Errore critico: {str(error)}")
        
        retry_interval = self.config['training'].get(
            'error_retry_interval', 600  # Valore di default
        )
        
        logger.info(f"Riprova tra {retry_interval} secondi...")
        time.sleep(retry_interval)

if __name__ == "__main__":
    try:
        trainer = ARIMATrainer()
        trainer.continuous_training()
    except KeyboardInterrupt:
        logger.info("Interruzione manuale dell'addestramento")
    except Exception as e:
        logger.critical(f"Errore fatale: {str(e)}", exc_info=True)