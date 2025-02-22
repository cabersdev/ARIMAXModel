import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from data_pipelines.fetch_data import (
    fetch_stock_data,
    process_data,
    save_data,
    main,
    load_config
)

@pytest.fixture
def mock_config():
    return {
        'raw_dir': 'test_data/raw',
        'processed_dir': 'test_data/processed',
        'years': 1,
        'quantile_threshold': 0.99
    }

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

def test_load_config():
    config = load_config()
    assert 'raw_dir' in config
    assert 'processed_dir' in config
    assert isinstance(config['years'], int)

@patch('yfinance.download')
def test_fetch_stock_data_success(mock_download, sample_data):
    mock_download.return_value = sample_data
    result = fetch_stock_data("AAPL")
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert len(result) == 5

@patch('yfinance.download')
def test_fetch_stock_data_failure(mock_download):
    mock_download.side_effect = Exception("API Error")
    result = fetch_stock_data("INVALID")
    
    assert result is None

def test_process_data_valid(sample_data):
    processed = process_data(sample_data, "AAPL")
    
    assert not processed.empty
    assert 'log_returns' in processed.columns
    assert processed['log_returns'].isna().sum() == 0
    assert len(processed) < len(sample_data)

def test_process_data_invalid():
    invalid_df = pd.DataFrame()
    result = process_data(invalid_df, "TEST")
    
    assert result is None

def test_save_data_success(tmp_path, sample_data):
    test_path = tmp_path / "test_dir"
    save_data(sample_data, test_path, "TEST")
    
    saved_file = test_path / "TEST.parquet"
    assert saved_file.exists()
    
    df = pd.read_parquet(saved_file)
    assert not df.empty

def test_save_data_failure(tmp_path, caplog):
    invalid_df = pd.DataFrame({'test': [1, 2, 3]})
    invalid_path = tmp_path / "invalid.txt"
    
    invalid_path.write_text("dummy content") 
    
    save_data(invalid_df, invalid_path, "TEST")
    
    assert "Errore salvataggio TEST" in caplog.text

@patch('data_pipelines.fetch_data.fetch_stock_data')
@patch('data_pipelines.fetch_data.process_data')
def test_main_flow(mock_process, mock_fetch, tmp_path, mock_config):
    mock_config['raw_dir'] = str(tmp_path / "raw")
    mock_config['processed_dir'] = str(tmp_path / "processed")
    
    sample_data = pd.DataFrame({
        'Date': [datetime.now()],
        'Open': [100],
        'High': [101],
        'Low': [99],
        'Close': [100],
        'Volume': [1000]
    })
    
    mock_fetch.return_value = sample_data
    mock_process.return_value = sample_data

    with patch('data_pipelines.fetch_data.load_config', return_value=mock_config):
        main()
    
    raw_files = list((tmp_path / "raw").glob("*.parquet"))
    processed_files = list((tmp_path / "processed").glob("*.parquet"))
    
    assert len(raw_files) == 3
    assert len(processed_files) == 3

@patch('yfinance.download')
def test_main_with_failures(mock_download, tmp_path, mock_config, caplog):
    mock_config['raw_dir'] = str(tmp_path / "raw")
    mock_config['processed_dir'] = str(tmp_path / "processed")

    # Configura i diversi comportamenti per i ticker
    def download_side_effect(ticker, start, end, progress, auto_adjust):
        if ticker == "AAPL":
            return pd.DataFrame()  # DataFrame vuoto
        elif ticker == "MSFT":
            return pd.DataFrame({
                'Date': [datetime.now()],
                'Open': [100],
                'High': [101],
                'Low': [99],
                'Close': [100],
                'Volume': [1000]
            })
        elif ticker == "GOOGL":
            raise Exception("API Error")

    mock_download.side_effect = download_side_effect

    with patch('data_pipelines.fetch_data.load_config', return_value=mock_config):
        main()

    assert "Nessun dato trovato per AAPL" in caplog.text
    assert "Errore elaborazione MSFT" not in caplog.text
    assert "Errore durante il download di GOOGL" in caplog.text

def test_date_conversion(sample_data):
    processed = process_data(sample_data, "TEST")
    assert pd.api.types.is_datetime64_any_dtype(processed['Date'])

def test_log_returns_calculation(sample_data):
    processed = process_data(sample_data, "TEST")
    expected_returns = np.log(sample_data['Close']).diff()
    
    expected_returns = expected_returns[expected_returns.abs() < expected_returns.quantile(0.99)]
    
    assert np.allclose(processed['log_returns'].dropna(), expected_returns.dropna())