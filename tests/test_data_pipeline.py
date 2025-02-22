import pytest
import polars as pl
import pandas as pd
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
import logging
from diskcache import Cache

from data_pipelines.fetch_data import (
    load_config,
    fetch_stock_data,
    process_data,
    save_data,
    fetch_and_process_stock_data,
    process_ticker,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

@pytest.fixture(autouse=True)
def setup_teardown():
    Cache("cache_directory").clear()
    yield
    if RAW_DATA_DIR.exists():
        for f in RAW_DATA_DIR.glob("*"):
            f.unlink()
        RAW_DATA_DIR.rmdir()
    if PROCESSED_DATA_DIR.exists():
        for f in PROCESSED_DATA_DIR.glob("*"):
            f.unlink()
        PROCESSED_DATA_DIR.rmdir()

@pytest.fixture
def mock_config(tmp_path):
    config = {
        'data': {
            'raw_dir': str(tmp_path / "raw"),
            'processed_dir': str(tmp_path / "processed"),
            'years': 2,
            'interval': '1d'
        }
    }
    with patch('src.data_pipelines.fetch_data.load_config') as mock_load:
        mock_load.return_value = config
        yield config

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', periods=5)
    return pd.DataFrame({
        'Date': dates,
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [95, 96, 97, 98, 99],
        'Close': [100, 101, 102, 101, 103],
        'Volume': [1000, 2000, 3000, 4000, 5000]
    })

def test_load_config(mock_config):
    config = load_config()
    assert config['data']['years'] == 2
    assert Path(config['data']['raw_dir']).name == "raw"

def test_fetch_stock_data_success(sample_data, mock_config):
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = sample_data
        result = fetch_stock_data("AAPL")
        
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 6)
        assert "Volume" in result.columns
        assert result["Date"][0] == datetime(2023, 1, 1)

def test_fetch_stock_data_failure(mock_config):
    with patch('yfinance.download', side_effect=Exception("API Error")):
        result = fetch_stock_data("INVALID")
        assert result is None

def test_process_data_normal_case(sample_data, mock_config):
    raw_df = pl.from_pandas(sample_data)
    processed = process_data(raw_df, "TEST")
    
    assert processed.schema["Date"] == pl.Date
    assert "log_returns" in processed.columns
    assert processed.null_count().sum_horizontal() == 0
    
    returns = processed["log_returns"].to_list()
    expected = [0.00995, 0.00985, -0.00985, 0.01961]  # Valori approssimati
    assert pytest.approx(returns, rel=1e-2) == expected

def test_process_data_edge_cases():
    assert process_data(pl.DataFrame(), "TEST") is None
    
    df = pl.DataFrame({
        "Date": [datetime(2023,1,1), datetime(2023,1,2)],
        "Close": [100.0, None]
    })
    processed = process_data(df, "TEST")
    assert processed is None

def test_save_data(tmp_path, mock_config):
    test_df = pl.DataFrame({"Close": [100, 101], "log_returns": [0.01, -0.005]})
    save_data(test_df, Path(mock_config['data']['raw_dir']), "TEST")
    
    saved_file = Path(mock_config['data']['raw_dir']) / "TEST.parquet"
    assert saved_file.exists()
    
    df = pl.read_parquet(saved_file)
    assert df.equals(test_df)

def test_full_pipeline_integration(mock_config, sample_data):
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = sample_data
        
        fetch_and_process_stock_data("AAPL")
        
        raw_files = list(Path(mock_config['data']['raw_dir']).glob("*"))
        processed_files = list(Path(mock_config['data']['processed_dir']).glob("*"))
        
        assert len(raw_files) == 1
        assert len(processed_files) == 1
        assert "AAPL" in str(processed_files[0])

def test_process_ticker_success(mock_config, sample_data):
    with patch('yfinance.download', return_value=sample_data):
        result = process_ticker("AAPL")
        assert result["status"] == "success"
        assert Path(mock_config['data']['raw_dir']).exists()

def test_process_ticker_failure(mock_config):
    with patch('yfinance.download', side_effect=Exception("Error")):
        result = process_ticker("INVALID")
        assert result["status"] == "failed"
        assert "Error" in result["error"]

def test_main_execution_with_multiple_tickers(mock_config, sample_data, caplog):
    with patch('yfinance.download', return_value=sample_data):
        from src.data_pipelines.fetch_data import __name__ as module_name
        with patch(f'{module_name}.tickers', ["AAPL", "MSFT"]):
            with patch(f'{module_name}.DEFAULT_YEARS', 1):
                import __main__
                assert "Success: 2/2" in caplog.text

def test_main_execution_with_single_ticker(mock_config, sample_data, caplog):
    with patch('yfinance.download', return_value=sample_data):
        from src.data_pipelines.fetch_data import __name__ as module_name
        with patch(f'{module_name}.tickers', ["AAPL"]):
            import __main__
            
            assert "Successfully processed AAPL" in caplog.text

def test_cache_functionality(mock_config, sample_data):
    with patch('yfinance.download', return_value=sample_data):
        result1 = fetch_stock_data("AAPL")
        result2 = fetch_stock_data("AAPL")
        
        assert result1 is result2

def test_logging_output(caplog, mock_config):
    with patch('yfinance.download', side_effect=Exception("Error")):
        process_ticker("INVALID")
        
        assert "Error processing INVALID" in caplog.text
        assert "Traceback" not in caplog.text 
def test_directory_creation_errors(mock_config):
    with patch('os.makedirs', side_effect=PermissionError("Access denied")):
        with pytest.raises(PermissionError):
            from src.data_pipelines.fetch_data import create_data_dirs
            create_data_dirs()
