import argparse
import yfinance as yf
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import logging
import numpy as np


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetch_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
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


def load_config():
    return {
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'years': 5,
        'quantile_threshold': 0.99
    }

def fetch_stock_data(ticker: str, years: int = 5) -> pd.DataFrame | None:
    try:
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - years)
        
        logger.info(f"Scaricando dati per {ticker}... \nPeriodo: {years} anni fa")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        if data.empty:
            logger.warning(f"Nessun dato trovato per {ticker} con data di inizio {years} anni fa")
            return None  # Modifica: ritorna None invece di DataFrame vuoto
            
        return data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    except Exception as e:
        logger.error(f"Errore durante il download di {ticker}): {str(e)}")
        return None
    
def process_data(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    try:
        logger.debug(f"Elaborando dati per {ticker}")
        df = raw_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['log_returns'] = np.log(df['Close']).diff()
        
        threshold = df['log_returns'].quantile(0.99)
        filtered = df[df['log_returns'].abs() < threshold]

        logger.debug(f"Elaborazione dati per {ticker} andata a buon fine")
        return filtered.dropna()
    except Exception as e:
        logger.error(f"Errore elaborazione {ticker}: {str(e)}")
        return None

def save_data(df: pd.DataFrame, path: Path, ticker: str):
    try:
        logger.debug(f"Salvando i dati per {ticker} \n path: {path}")

        path.mkdir(parents=True, exist_ok=True)
        output_path_parquet = path / f"{ticker}.parquet"
        output_path_csv = path / f"{ticker}.csv"

        df.to_parquet(output_path_parquet)
        logger.info(f"Dati salvati in {output_path_parquet}")

        df.to_csv(output_path_csv)
        logger.info(f"Dati salvati in {output_path_csv}")

    except Exception as e:
        logger.error(f"Errore salvataggio {ticker}: {str(e)}")

def get_live_data(ticker='GOOG', interval='1m'):
    """Fetch real-time market data from Yahoo Finance"""
    try:
        logger.debug(f"Tentativo di download per {ticker} con intervall {interval}")
        data = yf.download(
            tickers=ticker,
            period="1d",
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        logger.debug(f"Scaricate {len(data)} righe")
        return data.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error scaricando dati dal vivo: {e}")
        return pd.DataFrame()

def main():
    config = load_config()
    
    Path(config['raw_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['processed_dir']).mkdir(parents=True, exist_ok=True)

    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in tickers:
        raw_data = fetch_stock_data(ticker)
        logger.debug(f"Elaborando dati per {ticker}...")

        if raw_data is not None and not raw_data.empty:
            save_data(raw_data, Path(config['raw_dir']), ticker)
            
            processed_data = process_data(raw_data, ticker)
            if processed_data is not None and not processed_data.empty:
                save_data(processed_data, Path(config['processed_dir']), ticker)

    logger.info(f"Elaborazione completa")

if __name__ == "__main__":
    main()