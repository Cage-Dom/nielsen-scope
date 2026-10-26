import logging
import re

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor

from .models import get_model


logger = logging.getLogger(__name__)


LAG_COLUMN = re.compile(r".+_lag_(\d+)$")


def apply_feature_policy(X: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Drop columns that are excluded by config or unavailable at forecast time.

    min_lag=None keeps every column (one-step-ahead, the Stage 2 behaviour).
    min_lag=L keeps *_lag_N only when N >= L, drops every *_roll_* column (their
    windows include lag 1) and keeps calendar/static columns. L below
    cv.test_horizon raises, because the lags would read actuals inside the test
    window.
    """
    min_lag = params.get("min_lag")
    exclude = list(params.get("exclude_features") or [])

    missing = [c for c in exclude if c not in X.columns]
    if missing:
        raise ValueError(f"exclude_features not in feature bank: {missing}")

    drop = set(exclude)
    if min_lag is not None:
        test_horizon = params["cv"]["test_horizon"]
        if min_lag < test_horizon:
            raise ValueError(
                f"min_lag={min_lag} < cv.test_horizon={test_horizon}: lag "
                "features would read actuals inside the test window"
            )
        for col in X.columns:
            match = LAG_COLUMN.match(col)
            if "_roll_" in col or (match and int(match.group(1)) < min_lag):
                drop.add(col)
    return X.drop(columns=sorted(drop))


def select_book_features(
    features: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Filter Feature Bank to one ISBN ref for the modelling window,
    returning features (X), target (y) and the week-ending dates. String
    datatypes are dropped; params min_lag / exclude_features then apply the
    feature policy (see apply_feature_policy).
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
    X = apply_feature_policy(X, params)
    logger.info("select_book_features: ISBN=%s rows=%d features=%d", isbn, len(X), X.shape[1])

    return X, y, df['end_date']


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    dates: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    TimeSeriesSplit cross validation, model chosen by params['model_type'] via
    models.get_model. Logs per-fold and aggregate MAE/RMSE/MAPE to the active
    MLflow run. MAE is primary; MAPE is computed over non-zero actuals only.
    Returns (metrics, predictions); predictions has one row per test week
    (fold, end_date, y_true, y_pred); end_date is NaT when dates is None.
    When params['min_lag'] is None the figure is one-step-ahead (A-055) and
    mae_mean is also logged as one_step_mae.
    """
    cv = params['cv']
    model_type = params['model_type']
    model_params = params['model_params'][model_type]
    fit, predict = get_model(model_type)
    splitter = TimeSeriesSplit(n_splits=cv["n_splits"], test_size=cv["test_horizon"])

    #init the metrics and perform k fold CV
    fold_mae, fold_rmse, fold_mape, fold_preds = [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = fit(X_tr, y_tr, model_params)
        pred = predict(model, X_te)

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
        fold_preds.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "end_date": dates.iloc[test_idx].to_numpy() if dates is not None else pd.NaT,
                    "y_true": actual,
                    "y_pred": pred,
                }
            )
        )
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
    if params.get("min_lag") is None:
        mlflow.log_metric("one_step_mae", metrics["mae_mean"])
    logger.info("walk_forward aggregate: %s", metrics)

    return metrics, pd.concat(fold_preds, ignore_index=True)


def objective(trial, params: dict, X: pd.DataFrame, y: pd.Series, model_type: str) -> float:
    """Optuna objective: sample xgboost hyperparams, score via cv_xgb's
    walk-forward MAE (lower is better)."""
    if model_type != 'xgboost':
        raise NotImplementedError(f"Hyperparam tuning for {model_type} not implemented")

    #assign param limits for xgboost models
    tuning_cfg = params['hyperparam_tuning'][model_type]
    trial_params = dict(params)
    trial_params['model_type'] = model_type

    #set the model params for the trial
    trial_params['model_params'] = {
        model_type: {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                tuning_cfg['n_estimators']['lower'],
                tuning_cfg['n_estimators']['upper'],
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                tuning_cfg['max_depth']['lower'],
                tuning_cfg['max_depth']['upper'],
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                tuning_cfg['learning_rate']['lower'],
                tuning_cfg['learning_rate']['upper'],
                log=True,
            ),
        }
    }

    metrics = cv_xgb(X, y, trial_params)
    return metrics["mae_mean"]

def optuna_hyperparam_tune_xgb(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    """Optuna search over the XGBoost grid in params['hyperparam_tuning'], scored by
    cv_xgb's walk-forward mae_mean. Returns the best trial's params, shaped for
    params['model_params']['xgboost']. Expects a merged dict: the 'training' block
    plus the top-level 'hyperparam_tuning' block."""
    tuning_cfg = params["hyperparam_tuning"]["optuna"]
    seed = params["model_params"]["xgboost"].get("random_state", 42)
    study = optuna.create_study(
        direction=tuning_cfg["direction"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        lambda trial: objective(trial, params, X, y, "xgboost"),
        n_trials=tuning_cfg["n_trials"],
    )
    logger.info("optuna best_value=%.3f best_params=%s", study.best_value, study.best_params)
    
    return study.best_params



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

