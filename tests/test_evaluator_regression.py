import numpy as np
from sklearn.linear_model import LinearRegression
from evaluation import evaluator


def test_evaluate_regression_basic(tmp_path):
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10) * 2.0
    model = LinearRegression().fit(X, y)
    config = {"metrics": ["mse", "mae", "r2"]}
    metrics = evaluator.evaluate_regression(model, X, y, config, save_path=tmp_path / "metrics.json")
    assert set(metrics.keys()) == {"MSE", "MAE", "R2"}
    assert (tmp_path / "metrics.json").is_file()
