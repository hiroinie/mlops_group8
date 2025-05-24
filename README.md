# MLOps Group 8

This repository contains an example MLOps pipeline used for coursework. The
pipeline reads the dataset configured in `config.yaml`, applies preprocessing and
trains a scikit-learn model. The configuration now supports a linear regression
model which we use for the Spotify dataset.

## Running the pipeline

```bash
python -m src.main --config config.yaml --stage all
```

## Running tests

```bash
pytest -q
```
