import yfinance as yf
import polars as pl
from pathlib import Path
import os
from datetime import datetime
import logging
from typing import Optional
import yaml
from diskcache import Cache
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"

def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

CONFIG = load_config()

RAW_DATA_DIR = Path(CONFIG['data']['raw_dir'])
PROCESSED_DATA_DIR = Path(CONFIG['data']['processed_dir'])
DEFAULT_YEARS = CONFIG['data']['years']

CACHE = Cache("cache_directory")

def cache_data(func):
    @wraps(func)
    def wrapper(ticker, *args, **kwargs):
        key = f"{ticker}_{kwargs.get('years', DEFAULT_YEARS)}"
        if key in CACHE:
            return CACHE[key]
        result = func(ticker, *args, **kwargs)
        CACHE[key] = result
        return result
    return wrapper

def create_data_dirs() -> None:
    try:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directories: {e}")  # Correzione typo
        raise

@cache_data
def fetch_stock_data(ticker: str, years: int = DEFAULT_YEARS) -> Optional[pl.DataFrame]:
    try:
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - years)
        
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            timeout=10
        )

        if data.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        return data.reset_index()
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None

def process_data(raw_df: pl.DataFrame, ticker: str) -> Optional[pl.DataFrame]:
    try:
        processed_df = raw_df.with_columns(
            pl.col("Date").str.strptime(pl.Date, fmt=DATE_FORMAT),
            pl.col("Close").cast(pl.Float64)
        )

        processed_df = processed_df.with_columns(
            pl.col("Close").log().diff().alias("log_returns")
        ).drop_nulls(subset=["log_returns"])

        q = processed_df["log_returns"].quantile(0.99)
        processed_df = processed_df.filter(pl.col("log_returns").abs() < q)

        return processed_df
    except Exception as e:
        logger.error(f"Error processing the data for {ticker}: {e}")
        return None

def save_data(df: pl.DataFrame, path: Path, ticker: str):
    try:
        df.write_parquet(path / f"{ticker}.parquet")
        logger.info(f"Data saved successfully to {path / f'{ticker}.parquet'}")
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")

def fetch_and_process_stock_data(ticker: str, years: int = DEFAULT_YEARS) -> None:
    create_data_dirs()
    
    raw_df = fetch_stock_data(ticker, years)
    if raw_df is not None:
        save_data(pl.DataFrame(raw_df), RAW_DATA_DIR, ticker)

        processed_df = process_data(pl.DataFrame(raw_df), ticker)
        if processed_df is not None and not processed_df.is_empty():
            save_data(processed_df, PROCESSED_DATA_DIR, ticker)

def process_ticker(ticker):
    try:
        logger.info(f"Starting processing for {ticker}")
        fetch_and_process_stock_data(ticker)
        return {"ticker": ticker, "status": "success"}
    except Exception as e:
        logger.error(f"Error processing {ticker}: {str(e)}")
        return {"ticker": ticker, "status": "failed", "error": str(e)}

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] 
    # tickers = ["AAPL"]
      
    if isinstance(tickers, str):
        tickers = [tickers]
    
    if len(tickers) > 1:
        logger.info(f"Starting parallel processing for {len(tickers)} tickers")
        with ThreadPoolExecutor(max_workers=min(4, len(tickers))) as executor:
            results = list(executor.map(process_ticker, tickers))
        
        success_count = sum(1 for res in results if res["status"] == "success")
        logger.info(f"Processing completed. Success: {success_count}/{len(tickers)}")
        
        for result in results:
            if result["status"] == "failed":
                logger.error(f"Failed processing {result['ticker']}: {result['error']}")
                logger.debug(f"Stack trace: {result.get('trace', 'N/A')}")
                
    elif len(tickers) == 1:
        logger.info(f"Starting single ticker processing for {tickers[0]}")
        try:
            result = process_ticker(tickers[0])
            if result["status"] == "success":
                logger.info(f"Successfully processed {tickers[0]}")
            else:
                logger.error(f"Failed to process {tickers[0]}: {result['error']}")
        except Exception as e:
            logger.error(f"Critical error processing {tickers[0]}: {str(e)}", exc_info=True)
            raise
    else:
        logger.error("No valid tickers provided")
        raise ValueError("Empty tickers list provided")