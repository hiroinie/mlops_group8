import os
import pytest
import pandas as pd
from data_load import data_loader

# Paths for test files
# Keeping paths relative to the test file ensures portability and independence from the user's working directory
TEST_DIR = os.path.dirname(__file__)
MOCK_DATA_DIR = os.path.join(TEST_DIR, "mock_data")
MOCK_CSV = os.path.join(MOCK_DATA_DIR, "mock_data.csv")
MOCK_XLSX = os.path.join(MOCK_DATA_DIR, "mock_data.xlsx")
BAD_FILE = os.path.join(MOCK_DATA_DIR, "nonexistent.csv")

# Example minimal config dicts
# These are "inline configs" for test isolation, so tests do not depend on or modify real production configs
CSV_CONFIG = {
    "data_source": {
        "raw_path": MOCK_CSV,
        "type": "csv",
        "delimiter": ",",
        "header": 0,
        "encoding": "utf-8"
    }
}

EXCEL_CONFIG = {
    "data_source": {
        "raw_path": MOCK_XLSX,
        "type": "excel",
        "sheet_name": "Sheet1",
        "header": 0,
        "encoding": "utf-8"
    }
}


def test_load_data_csv_success(monkeypatch):
    """
    Test: Successful data loading from a CSV file.
    'Happy path' scenario, ensuring basic functionality works with correct config and valid data.
    """
    # Patch load_config so the data loader uses the *test* config, not a real config file
    monkeypatch.setattr(data_loader, "load_config", lambda _: CSV_CONFIG)
    # Run the data loading process
    df = data_loader.get_data(config_path="dummy.yaml", env_path=None)
    # Assert the returned object is a DataFrame and is not empty
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_missing_file(monkeypatch):
    """
    Test: Data loader should raise FileNotFoundError if the data file is missing.
    Tests error handling 
    """
    config = dict(CSV_CONFIG)
    # Point to a file that doesn't exist
    config["data_source"]["raw_path"] = BAD_FILE
    monkeypatch.setattr(data_loader, "load_config", lambda _: config)
    with pytest.raises(FileNotFoundError):
        data_loader.get_data(config_path="dummy.yaml", env_path=None)


def test_load_data_unsupported_type(monkeypatch):
    """
    Test: Data loader should raise ValueError for unsupported file types.
    Ensures the module fails early and clearly when receiving bad input, preventing hidden bugs downstream.
    """
    config = dict(CSV_CONFIG)
    # Not implemented in the data_loader
    config["data_source"]["type"] = "parquet"
    config["data_source"]["raw_path"] = MOCK_CSV
    monkeypatch.setattr(data_loader, "load_config", lambda _: config)
    with pytest.raises(ValueError):
        data_loader.get_data(config_path="dummy.yaml", env_path=None)


def test_load_data_no_path(monkeypatch):
    """
    Test: Data loader should raise ValueError if no data file path is provided.
    Verifies config validation logic is enforced, preventing ambiguous failures or cryptic error messages.
    """
    config = dict(CSV_CONFIG)
    config["data_source"].pop("raw_path")  # Remove 'path' key
    monkeypatch.setattr(data_loader, "load_config", lambda _: config)
    with pytest.raises(ValueError):
        data_loader.get_data(config_path="dummy.yaml", env_path=None)
