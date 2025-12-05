# -*- coding: utf-8 -*-
"""
modules/hpo.py

Optuna 기반 하이퍼파라미터 최적화 래퍼.
지원 모델 (model_name):
  - RandomForestClassifier
  - RandomForestRegressor
  - GradientBoostingClassifier
  - GradientBoostingRegressor
  - SVC (RBF)
  - LogisticRegression

기본 scoring:
  - classification: accuracy
  - regression: r2
"""
from typing import Any, Dict, Optional, Tuple
import time
import numpy as np
import traceback
import logging

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.pruners import MedianPruner
except Exception as e:
    raise ImportError("optuna가 설치되어 있지 않습니다. requirements.txt를 확인하세요.") from e

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    raise ImportError("scikit-learn 의존성이 누락되었습니다.") from e


MODEL_CHOICES = [
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "SVC",
    "LogisticRegression",
]


def _task_from_y(y: np.ndarray) -> str:
    try:
        if np.issubdtype(y.dtype, np.number):
            nuniq = len(np.unique(y[~np.isnan(y)]))
            if nuniq <= 20:
                return "classification"
            else:
                return "regression"
        else:
            return "classification"
    except Exception:
        return "unknown"


def _space_rf_classifier(trial: optuna.trial.Trial, random_state: int):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": random_state,
    }
    return RandomForestClassifier(**params)


def _space_rf_regressor(trial: optuna.trial.Trial, random_state: int):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": random_state,
    }
    return RandomForestRegressor(**params)


def _space_gb_classifier(trial: optuna.trial.Trial, random_state: int):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": random_state,
    }
    return GradientBoostingClassifier(**params)


def _space_gb_regressor(trial: optuna.trial.Trial, random_state: int):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": random_state,
    }
    return GradientBoostingRegressor(**params)


def _space_svc(trial: optuna.trial.Trial, random_state: int):
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
        "kernel": "rbf",
        "probability": True,
        "random_state": random_state,
    }
    return SVC(**params)


def _space_logreg(trial: optuna.trial.Trial, random_state: int):
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l2"]),
        "solver": "lbfgs",
        "max_iter": 500,
        "random_state": random_state,
    }
    return LogisticRegression(**params)


MODEL_BUILDERS = {
    "RandomForestClassifier": _space_rf_classifier,
    "RandomForestRegressor": _space_rf_regressor,
    "GradientBoostingClassifier": _space_gb_classifier,
    "GradientBoostingRegressor": _space_gb_regressor,
    "SVC": _space_svc,
    "LogisticRegression": _space_logreg,
}


def _study_summary(study: optuna.study.Study) -> Dict[str, Any]:
    trials = []
    for t in study.trials:
        trials.append(
            {
                "number": t.number,
                "state": str(t.state),
                "value": None if t.value is None else float(t.value),
                "params": {
                    k: (
                        None
                        if v is None
                        else (int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v)
                    )
                    for k, v in t.params.items()
                },
            }
        )
    best = None
    try:
        best = {"number": study.best_trial.number, "value": float(study.best_trial.value), "params": study.best_trial.params}
    except Exception:
        best = None
    return {"n_trials": len(study.trials), "best_trial": best, "trials": trials}


def run_hpo(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    time_budget: int = 120,
    n_trials: Optional[int] = None,
    cv: int = 3,
    random_state: int = 0,
    direction: Optional[str] = None,
    n_jobs: int = 1,
    task_override: Optional[str] = None,
    callbacks: Optional[list] = None,
    patience: Optional[int] = None,
    min_trials_before_stop: int = 5,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """
    주어진 X, y에 대해 Optuna 기반 HPO 수행.
    model_name은 MODEL_CHOICES 중 하나.

    Args:
        time_budget: 전체 탐색 시간(초)
        n_trials: 최대 trial 수 (None이면 시간 기반)
        patience: 연속 미개선 trial 허용 횟수 (None/0이면 사용 안 함)
        min_trials_before_stop: 조기중단 발동 전 최소 trial 수
        tolerance: 개선으로 인정할 최소 차이 (val > best + tolerance)
    """
    start_time = time.time()
    res: Dict[str, Any] = {"best_params": None, "best_value": None, "study_summary": None, "error": None, "model_name": model_name}

    try:
        task = task_override if task_override in {"classification", "regression"} else _task_from_y(y)
        if task == "unknown":
            raise ValueError("타깃(y)을 분류/회귀로 구분할 수 없습니다.")

        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"지원하지 않는 model_name: {model_name}")

        if task == "classification":
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            scoring = "accuracy"
            direction_auto = "maximize"
        else:
            cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            scoring = "r2"
            direction_auto = "maximize"
        if direction is None:
            direction = direction_auto

        builder = MODEL_BUILDERS[model_name]
        best_val_so_far = None
        no_improve_count = 0

        def objective(trial: optuna.trial.Trial):
            clf = builder(trial, random_state=random_state)
            try:
                scores = cross_val_score(clf, X, y, cv=cv_split, scoring=scoring, n_jobs=n_jobs, error_score=np.nan)
            except TypeError:
                scores = cross_val_score(clf, X, y, cv=cv_split, scoring=scoring, error_score=np.nan)
            return float(np.nanmean(scores))

        def _early_stop_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            nonlocal best_val_so_far, no_improve_count
            val = trial.value
            if val is None or (isinstance(val, float) and np.isnan(val)):
                no_improve_count += 1
            else:
                if best_val_so_far is None or val > best_val_so_far + tolerance:
                    best_val_so_far = val
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            if patience and patience > 0:
                if len(study.trials) >= max(min_trials_before_stop, patience) and no_improve_count >= patience:
                    study.stop()

        cb_list = callbacks[:] if callbacks else []
        if patience and patience > 0:
            cb_list.append(_early_stop_callback)

        study = optuna.create_study(direction=direction, pruner=MedianPruner())
        study.optimize(
            objective,
            timeout=time_budget,
            n_trials=n_trials,
            callbacks=cb_list if cb_list else None,
        )

        best_trial = study.best_trial
        res["best_params"] = best_trial.params if best_trial is not None else None
        res["best_value"] = None if best_trial is None else float(best_trial.value)
        res["study_summary"] = _study_summary(study)
        return res

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("run_hpo 실패: %s\n%s", e, tb)
        res["error"] = f"{e}"
        return res


def run_optuna_sample(df, time_budget: int = 60, model_name: str = "RandomForestClassifier") -> Dict[str, Any]:
    """
    간이 샘플: df에서 target 컬럼을 찾고 지정된 model_name으로 HPO 수행.
    """
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise ValueError("df는 pandas.DataFrame 이어야 합니다.")
        # find target col
        target_col = None
        for cand in ["target", "label", "y", "class"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col is None:
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    nunique = int(df[c].nunique(dropna=True))
                    if 2 <= nunique <= min(20, max(2, int(df.shape[0] * 0.2))):
                        target_col = c
                        break
        if target_col is None:
            raise ValueError("타깃 컬럼을 찾을 수 없습니다.")

        X = df.select_dtypes(include=["number"]).drop(columns=[target_col], errors="ignore").fillna(0).values
        y = df[target_col].values
        return run_hpo(model_name=model_name, X=X, y=y, time_budget=time_budget)
    except Exception as e:
        logger.error("run_optuna_sample 실패: %s", e)
        return {"error": str(e)}
