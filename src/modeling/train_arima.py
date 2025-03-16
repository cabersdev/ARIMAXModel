import time
import joblib
from datetime import datetime
from .model import AdaptiveARIMAX
from src.data_pipelines import fetch_data
from src.backtesting import BacktestExecutor

class ARIMATrainer:
    """Pipeline di addestramento continuo con aggiornamento adattivo"""
    
    def __init__(self, config_path='configs/parameters_config.yaml'):
        self.model = AdaptiveARIMAX()
        self.backtester = BacktestExecutor(config_path)
        self.current_model_version = None
        
    def continuous_training(self):
        """Loop principale di addestramento continuo"""
        while True:
            try:
                raw_data = fetch_data.get_live_data()
                processed_data = self._process_streaming_data(raw_data)
                
                if self._needs_retraining():
                    self._train_and_validate(processed_data)
                    
                signals = self._generate_signals(processed_data)
                fetch_data.push_signals_to_api(signals)
                
                time.sleep(self.model.config['training_interval'])
                
            except Exception as e:
                self._handle_training_error(e)

    def _process_streaming_data(self, raw_data):
        """Processa dati in real-time mantenendo lo stato"""
        window_size = self.model.config['data_processing']['rolling_window']
        if len(raw_data) > window_size:
            return raw_data.iloc[-window_size:]
        return raw_data

    def _needs_retraining(self):
        """Determina se è necessario riaddestrare il modello"""
        return True

    def _train_and_validate(self, data):
        """Pipeline completa addestramento e validazione"""
        self.model.fit(data)
        backtest_results = self.backtester.run_backtest(data)
        
        if self._validate_performance(backtest_results):
            self._deploy_new_model()
            
    def _validate_performance(self, results):
        """Valida performance secondo criteri configurati"""
        min_sharpe = self.model.config['backtesting']['min_sharpe']
        max_drawdown = self.model.config['backtesting']['max_drawdown']
        return results['sharpe'] > min_sharpe and results['max_drawdown'] < max_drawdown

    def _deploy_new_model(self):
        """Deploy del nuovo modello e archiviazione versione"""
        version = datetime.now().strftime("%Y%m%d%H%M")
        joblib.dump(self.model, f'models/arima_{version}.pkl')
        self.current_model_version = version

    def _generate_signals(self, data):
        """Genera segnali con modello corrente"""
        predictions = self.model.predict(data)
        return self.model.generate_signals(predictions)

    def _handle_training_error(self, error):
        """Gestione errori e fallback"""
        print(f"Errore nell'addestramento: {str(error)}")
        time.sleep(self.model.config['error_retry_interval'])

if __name__ == "__main__":
    trainer = ARIMATrainer()
    trainer.continuous_training()