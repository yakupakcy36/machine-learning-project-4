# Car Price Prediction

Small learning project. The notebook `car_price_prediction.ipynb` trains linear and lasso regression.

Data: Used car data (car data.csv), predicting the selling price. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`benchmark.py` compares ridge, lasso and gradient boosting regression with 5-fold cross-validation.
It does not use the real dataset. It runs on synthetic data generated with numpy/scikit-learn (850 samples, 7 features), so the numbers below only compare the models on that data and say nothing about the real dataset.

## How to run

```
pip install numpy scikit-learn jupyter
jupyter notebook
python benchmark.py
```

## Results (5-fold CV, mean)

| Model | r2 | rmse | mae |
|---|---|---|---|
| Gradient Boosting Regressor | 0.9203 | 1.1472 | 0.4913 |
| Ridge Regression (L2) | 0.8073 | 1.7841 | 1.0778 |
| Lasso Regression (L1) | 0.8067 | 1.787 | 1.0721 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `car_price_prediction.ipynb`: notebook
- `benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
