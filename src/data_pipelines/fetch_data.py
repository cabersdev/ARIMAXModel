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