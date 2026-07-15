import logging
import mlflow
import numpy as np 
import pandas as pd 
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


logger = logging.getLogger(__name__)


def select_book_features(features: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Filter Feature Bank to one ISBN ref for the modelling window,
    returning features (X) and target (y) matrix. String datatypes
    are dropped.
    """
    isbn = str(params['isbn'])
    start = pd.Timestamp(params["model_start_date"])

    #create feature slice for isbn
    df = features[features['isbn'] == isbn].copy()
    df['end_date'] = pd.to_datetime(df["end_date"])
    df = df[df['end_date'] >= start]
    df = df.sort_values('end_date').reset_index(drop=True)

    #create the splits and check against required minimum
    n_splits = params["cv"]["n_splits"]
    test_horizon = params["cv"]["test_horizon"]
    min_rows = n_splits * test_horizon + 1
    assert len(df) >= min_rows, (
        f"ISBN {isbn}: {len(df)} in-window rows < required {min_rows} "
        f"(n_splits={n_splits} * test_horizon={test_horizon} + 1)"
    )

    #assign features and target
    y = df['volume']
    X = df.select_dtypes(include="number").drop(columns=["volume"])
    logger.info("select_book_features: ISBN=%s rows=%d features=%d", isbn, len(X), X.shape[1])
    
    return X, y


def walk_forward_evaluate_xgb(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    """
    TimeSeriesSplit walk forward cross validation with XGBoost.
    Log-per fold and aggregate MAE/RMSE/MAPE to the active MLflow run. 
    MAE is primary; MAPE is computed over non-zero actuals only"""
    cv = params['cv']
    model_params = params['model_params'][params['model_type']]
    splitter = TimeSeriesSplit(n_splits=cv["n_splits"], test_size=cv["test_horizon"])

    #init the metrics and perform k fold CV
    fold_mae, fold_rmse, fold_mape = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**model_params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        #calc metrics
        mae = float(mean_absolute_error(y_te, pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        #conv to array
        actual = y_te.to_numpy()
        nz = actual != 0
        excluded = int((~nz).sum())
        mape = float(mean_absolute_percentage_error(actual[nz], pred[nz])) if nz.any() else float("nan")

        #update metrics for fold
        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_mape.append(mape)
        mlflow.log_metric("fold_mae", mae, step=fold)
        mlflow.log_metric("fold_rmse", rmse, step=fold)
        mlflow.log_metric("fold_mape_excluded_rows", excluded, step=fold)
        if np.isfinite(mape):
            mlflow.log_metric("fold_mape", mape, step=fold)
        logger.info(
            "fold %d: MAE=%.3f RMSE=%.3f MAPE=%.4f (excluded %d zero-actual rows)",
            fold, mae, rmse, mape, excluded,
        )
    
    metrics = {
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std": float(np.std(fold_rmse)),
        "mape_mean": float(np.nanmean(fold_mape)),
        "mape_std": float(np.nanstd(fold_mape)),
        "n_folds": splitter.n_splits,
    }
    mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
    logger.info("walk_forward aggregate: %s", metrics)
    
    return metrics


def train_final_xgb_model(X: pd.DataFrame, y: pd.Series, params: dict) -> XGBRegressor:
    """Fit the final xgb model on the full training history; the returned estimator is 
    logged + registered in the catalogue (MLFlowModelTrackingDataset)."""
    #load and log params
    model_params = params["model_params"][params["model_type"]]
    mlflow.log_params(model_params)
    
    #train
    model = XGBRegressor(**model_params)
    model.fit(X, y)
    logger.info(
        "train_final_model: fitted %s on %d rows x %d features",
        params["model_type"], len(X), X.shape[1],
    )
    return model  

