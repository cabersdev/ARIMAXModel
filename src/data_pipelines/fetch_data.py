# File: src/data_pipelines/fetch_data.py
import yfinance as yf
import polars as pl
from pathlib import Path
import os
 
# Definizione e configurazione dei percorsi relativi allo storing dei dati
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_and_process_stock_data(ticker: str, years: int = 10) -> None:
    raw_data = yfinance.download(
        tickers = ticker,
        period=f"{years}y",
        interval="1d",
        progress=False
    )

    df_raw = pl.DataFrame(raw_data.reset_index())
    df_raw.write_csv(RAW_DATA_DIR / f"{ticker}_raw.csv")

    df_processed = df_raw.with_columns(
        pl.col("Date").str.strptime(pl.Date, fmt="%Y-%m-%d"),
        pl.col("Close").log().diff().alias("log_returns")
    ).drop_nulls()

    df_processed.write_csv(PROCESSED_DATA_DIR / f"{ticker}_processed.csv")

    


