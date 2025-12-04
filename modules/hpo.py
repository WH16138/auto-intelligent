"""
modules/hpo.py

Optuna 기반 하이퍼파라미터 튜닝 유틸.

주요 함수
- run_hpo(model_name, X, y, time_budget, cv=3, direction=None)
    : model_name (str)에 따라 탐색 공간을 구성하고 Optuna로 최적화 수행.
      반환: dict { 'best_params', 'best_value', 'study_summary', 'trials' }

- run_optuna_sample(df, time_budget)
    : (하위 호환) DataFrame을 받아 내부에서 X,y를 추출(타깃 컬럼 'target' 우선)하여 간단히 RandomForest 튜닝 수행.
      반환 형식은 run_hpo와 동일.

설치: optuna가 requirements.txt에 포함되어 있어야 합니다.
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
    raise ImportError("optuna가 설치되어 있어야 합니다. requirements.txt를 확인하세요.") from e

try:
    import sklearn
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, make_scorer
except Exception as e:
    raise ImportError("scikit-learn이 설치되어 있어야 합니다.") from e

# ---------------------------------------
# Helpers
# ---------------------------------------
def _task_from_y(y: np.ndarray) -> str:
    """간단한 y 기반 문제 유형 추정"""
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


def _default_search_space(trial: optuna.trial.Trial):
    """
    RandomForest를 위한 기본 탐색 공간(공용).
    확장하고 싶으면 edit 하세요.
    """
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "min_samples_split": min_samples_split,
    }


def _study_summary(study: optuna.study.Study) -> Dict[str, Any]:
    """간단한 study summary를 생성"""
    trials = []
    for t in study.trials:
        trials.append({
            "number": t.number,
            "state": str(t.state),
            "value": None if t.value is None else float(t.value),
            "params": {k: (None if v is None else (int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v)) for k, v in t.params.items()}
        })
    best = None
    try:
        best = {"number": study.best_trial.number, "value": float(study.best_trial.value), "params": study.best_trial.params}
    except Exception:
        best = None
    return {"n_trials": len(study.trials), "best_trial": best, "trials": trials}


# ---------------------------------------
# Primary HPO function
# ---------------------------------------
def run_hpo(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    time_budget: int = 120,
    cv: int = 3,
    random_state: int = 0,
    direction: Optional[str] = None,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    주어진 X,y에 대해 Optuna로 하이퍼파라미터를 탐색합니다.

    Args:
        model_name: 튜닝할 모델 이름(예: 'RandomForest', 'RandomForestClassifier', 'RandomForestRegressor')
        X: numpy array (n_samples, n_features)
        y: numpy array (n_samples,)
        time_budget: 타임아웃(초)
        cv: 교차검증 분할 수
        direction: 'maximize' 또는 'minimize' (자동 결정 가능)
        n_jobs: cross_val_score에 넘길 n_jobs (주의: 병렬 실행 시 프로세스/메모리 고려)

    Returns:
        dict: {
          'best_params': {...},
          'best_value': float,
          'study_summary': {...},
          'error': None or str
        }
    """
    start_time = time.time()
    res: Dict[str, Any] = {"best_params": None, "best_value": None, "study_summary": None, "error": None}

    try:
        task = _task_from_y(y)
        if task == "unknown":
            raise ValueError("타깃(y)의 타입을 판별하지 못했습니다.")

        # choose default scorer & cv
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

        # define objective
        def objective(trial: optuna.trial.Trial):
            # only RandomForest space for now; can be extended for different model_name strings
            params = _default_search_space(trial)
            if task == "classification":
                clf = RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]) if params["max_depth"] is not None else None,
                    max_features=params["max_features"],
                    min_samples_split=int(params["min_samples_split"]),
                    random_state=random_state,
                )
            else:
                clf = RandomForestRegressor(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]) if params["max_depth"] is not None else None,
                    max_features=params["max_features"],
                    min_samples_split=int(params["min_samples_split"]),
                    random_state=random_state,
                )

            # cross_val_score; use n_jobs argument carefully
            try:
                scores = cross_val_score(clf, X, y, cv=cv_split, scoring=scoring, n_jobs=n_jobs, error_score=np.nan)
            except TypeError:
                # older sklearn may not accept n_jobs in cross_val_score
                scores = cross_val_score(clf, X, y, cv=cv_split, scoring=scoring, error_score=np.nan)
            # Use mean score as trial value
            mean_score = float(np.nanmean(scores))
            # Optuna by default maximizes if direction='maximize'
            return mean_score

        # Optuna study
        study = optuna.create_study(direction=direction, pruner=MedianPruner())
        # run optimize with timeout
        study.optimize(objective, timeout=time_budget)

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


# ---------------------------------------
# Backward-compatible wrapper: run_optuna_sample
# ---------------------------------------
def run_optuna_sample(df, time_budget: int = 60) -> Dict[str, Any]:
    """
    하위호환 함수: modules.pipeline에서 예전 스켈레톤이 호출할 수 있도록 제공.
    - df: pandas.DataFrame, 'target' 컬럼을 우선으로 사용
    - time_budget: 초

    반환: run_hpo와 같은 형식의 dict
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
            # try heuristic: non-numeric with small cardinality
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    nunique = int(df[c].nunique(dropna=True))
                    if 2 <= nunique <= min(20, max(2, int(df.shape[0] * 0.2))):
                        target_col = c
                        break
        if target_col is None:
            raise ValueError("타깃 컬럼을 찾을 수 없습니다. 'target' 컬럼을 넣거나 target_col을 명시하세요.")

        X = df.select_dtypes(include=["number"]).drop(columns=[target_col], errors="ignore").fillna(0).values
        y = df[target_col].values
        # call run_hpo (default model_name RandomForest)
        return run_hpo(model_name="RandomForest", X=X, y=y, time_budget=time_budget)
    except Exception as e:
        logger.error("run_optuna_sample 실패: %s", e)
        return {"error": str(e)}
