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
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from .models import NAIVE_LAG_COLUMN, get_interval_predictor, get_model, predict_naive

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


def _score_fold(actual: np.ndarray, pred: np.ndarray) -> dict:
    """MAE, RMSE, MAPE (non-zero actuals only) and the zero-actual row count of a fold."""
    nz = actual != 0
    mape = float("nan")
    if nz.any():
        mape = float(mean_absolute_percentage_error(actual[nz], pred[nz]))
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mape": mape,
        "mape_excluded_rows": int((~nz).sum()),
    }


def _log_fold(row: dict) -> None:
    """Log one fold's metrics to the active MLflow run and the module logger."""
    fold = row["fold"]
    mlflow.log_metric("fold_mae", row["mae"], step=fold)
    mlflow.log_metric("fold_rmse", row["rmse"], step=fold)
    mlflow.log_metric("fold_mape_excluded_rows", row["mape_excluded_rows"], step=fold)
    if np.isfinite(row["mape"]):
        mlflow.log_metric("fold_mape", row["mape"], step=fold)
    if np.isfinite(row["rel_mae"]):
        mlflow.log_metric("fold_relative_mae", row["rel_mae"], step=fold)
    logger.info(
        "fold %d: MAE=%.3f RMSE=%.3f MAPE=%.4f (excluded %d zero-actual rows)",
        fold, row["mae"], row["rmse"], row["mape"], row["mape_excluded_rows"],
    )


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    dates: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    TimeSeriesSplit cross validation, model chosen by params['model_type'] via
    models.get_model. Logs per-fold and aggregate MAE/RMSE/MAPE to the active
    MLflow run. MAE is primary; MAPE is computed over non-zero actuals only and
    mape_excluded_rows counts the zero-actual rows left out. relative_mae is
    mae_mean divided by the seasonal-naive mae (volume_lag_52 over the same folds);
    NaN when that column is not in X or the naive error is 0.
    Returns (metrics, predictions, fold_metrics). predictions has one row per test
    week (fold, end_date, y_true, y_pred, lo_95, hi_95); end_date is NaT when dates is
    None and lo_95/hi_95 are NaN for models without an interval function.
    fold_metrics has one row per fold (fold, mae, rmse, mape, mape_excluded_rows,
    naive_mae, rel_mae, fell_back); fell_back is <NA> for models that cannot fall back.
    When params['min_lag'] is None the figure is one-step-ahead (A-055) and
    mae_mean is also logged as one_step_mae.
    """
    cv = params['cv']
    model_type = params['model_type']
    model_params = params['model_params'].get(model_type, {})
    fit, predict = get_model(model_type)
    interval_predict = get_interval_predictor(model_type)
    splitter = TimeSeriesSplit(n_splits=cv["n_splits"], test_size=cv["test_horizon"])
    has_naive_ref = NAIVE_LAG_COLUMN in X.columns

    #perform k fold CV
    fold_rows, fold_preds = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = fit(X_tr, y_tr, model_params)
        pred = predict(model, X_te)
        if interval_predict is None:
            lower = upper = np.full(len(pred), np.nan)
        else:
            lower, upper = interval_predict(model, X_te)

        #score the fold against the actuals and the seasonal naive
        actual = y_te.to_numpy()
        naive_mae = float("nan")
        if has_naive_ref:
            naive_mae = float(mean_absolute_error(actual, predict_naive(None, X_te)))
        scores = _score_fold(actual, pred)
        row = {"fold": fold} | scores | {
            "naive_mae": naive_mae,
            "rel_mae": scores["mae"] / naive_mae if naive_mae > 0 else float("nan"),
            "fell_back": getattr(model, "fell_back", None),
        }
        fold_rows.append(row)
        fold_preds.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "end_date": dates.iloc[test_idx].to_numpy() if dates is not None else pd.NaT,
                    "y_true": actual,
                    "y_pred": pred,
                    "lo_95": lower,
                    "hi_95": upper,
                }
            )
        )
        _log_fold(row)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics["fell_back"] = fold_metrics["fell_back"].astype("boolean")
    fold_mae = fold_metrics["mae"].to_numpy()
    fold_rmse = fold_metrics["rmse"].to_numpy()
    fold_mape = fold_metrics["mape"].to_numpy()
    fold_naive_mae = fold_metrics["naive_mae"].to_numpy()

    mae_mean = float(np.mean(fold_mae))
    relative_mae = float("nan")
    if not has_naive_ref:
        logger.warning("relative_mae undefined: %s is not in the features", NAIVE_LAG_COLUMN)
    elif np.mean(fold_naive_mae) > 0:
        relative_mae = mae_mean / float(np.mean(fold_naive_mae))
    else:
        logger.warning("relative_mae undefined: naive MAE is 0 over all folds")

    metrics = {
        "mae_mean": mae_mean,
        "mae_std": float(np.std(fold_mae)),
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std": float(np.std(fold_rmse)),
        "mape_mean": float(np.nanmean(fold_mape)),
        "mape_std": float(np.nanstd(fold_mape)),
        "relative_mae": relative_mae,
        "mape_excluded_rows": int(fold_metrics["mape_excluded_rows"].sum()),
        "n_folds": splitter.n_splits,
    }
    mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
    if params.get("min_lag") is None:
        mlflow.log_metric("one_step_mae", metrics["mae_mean"])
    logger.info("walk_forward aggregate: %s", metrics)

    return metrics, pd.concat(fold_preds, ignore_index=True), fold_metrics


def objective(trial, params: dict, X: pd.DataFrame, y: pd.Series, model_type: str) -> float:
    """Optuna objective: sample xgboost hyperparams, score via cross_validate's
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

    metrics, _, _ = cross_validate(X, y, trial_params)
    return metrics["mae_mean"]

def optuna_hyperparam_tune_xgb(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    """Optuna search over the XGBoost grid in params['hyperparam_tuning'], scored by
    cross_validate's walk-forward mae_mean. Returns the best trial's params, shaped for
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