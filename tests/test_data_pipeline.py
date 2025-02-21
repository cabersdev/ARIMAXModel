import pytest
import polars as pl
import pandas as pd
from datetime import datetime
from src.data_pipelines.fetch_data import fetch_stock_data, process_data
from unittest.mock import patch

@pytest.fixture
def mock_raw_data():
    # Crea un DataFrame Polars corretto
    dates = pl.datetime_range(
        start=datetime(2020, 1, 1), 
        end=datetime(2020, 1, 5), 
        interval="1d",
        eager=True
    )
    return pl.DataFrame({
        "Date": dates,
        "Close": [100.0, 101.0, 102.0, 101.0, 103.0]
    })

def test_fetch_data_success():
    with patch('yfinance.download') as mock_download:
        # Simula ritorno di un DataFrame pandas
        mock_download.return_value = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=5),
            'Close': [100, 101, 102, 101, 103]
        })
        data = fetch_stock_data("AAPL")
        assert data is not None
        assert data.height == 5

def test_process_data(mock_raw_data):
    processed = process_data(mock_raw_data, "AAPL")
    assert processed is not None
    assert "log_returns" in processed.columns
    assert processed.shape[0] == 4  # 5 rows - 1 null
    assert all(processed["log_returns"].abs() < processed["log_returns"].quantile(0.99))