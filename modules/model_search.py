"""
modules/model_search.py

기능 요약:
- 데이터프레임으로부터 X, y 추출(get_X_y)
- 문제 유형 감지(detect_task_type): classification / regression / unknown
- 간단하고 빠른 베이스라인 모델 비교(quick_baselines)
- 모델 학습(학습된 모델 딕셔너리 반환) 및 교차검증 메트릭 리포트 반환

디자인 원칙:
- UI(예: Streamlit)에 의존하지 않고 재사용 가능하게 설계
- numeric 피처만 우선적으로 사용 (전처리/피처엔지니어링이 선행되면 더 정확)
- 타깃 열 자동 감지: 'target' 우선, 없으면 ingest fingerprint에서 가져오도록 호출자에게 target_col 전달 가능
"""

from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
import warnings
import logging

logger = logging.getLogger(__name__)


def detect_task_type(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """
    간단한 문제 유형 감지.
    - classification: 타깃이 범주형이거나(정수/문자 중 고유값이 적음) 고유값 수가 적은 숫자형
    - regression: 타깃이 연속적인 숫자형(고유값이 많음)
    - unknown: 타깃이 없거나 판별 불가

    반환값: 'classification' | 'regression' | 'unknown'
    """
    if target_col is None or target_col not in df.columns:
        return "unknown"
    s = df[target_col]
    if pd.api.types.is_numeric_dtype(s):
        nunique = int(s.nunique(dropna=True))
        # heuristic: 숫자형이지만 고유값이 적으면 classification
        if nunique <= 20:
            return "classification"
        else:
            return "regression"
    else:
        # non-numeric -> classification
        return "classification"


def get_X_y(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    DataFrame으로부터 X (numpy), y (numpy), task_type, feature_names 반환.
    기본적으로 숫자형 컬럼만 사용함(전처리·인코딩이 있으면 더 좋은 결과).
    target_col 미지정 시 'target' 컬럼을 우선으로 시도.

    반환:
        X: (n_samples, n_features) numpy 배열
        y: (n_samples,) numpy 배열
        task_type: 'classification'|'regression'|'unknown'
        feature_names: 사용된 피처명 리스트
    """
    if target_col is None:
        if "target" in df.columns:
            target_col = "target"
        else:
            # try common names
            for c in ["label", "y", "class"]:
                if c in df.columns:
                    target_col = c
                    break

    if target_col is None or target_col not in df.columns:
        raise ValueError("타깃 컬럼이 지정되지 않았거나 존재하지 않습니다. target_col을 지정하세요.")

    # select numeric features only
    features = df.select_dtypes(include=["number"]).columns.tolist()
    # remove target if present among numeric features
    features = [c for c in features if c != target_col]
    if len(features) == 0:
        raise ValueError("사용 가능한 숫자형 피처가 없습니다. 전처리로 숫자형 피처를 확보하세요.")

    X = df[features].fillna(0).values
    y = df[target_col].values
    task = detect_task_type(df, target_col)
    return X, y, task, features


def _safe_cross_val_score(estimator, X, y, cv, scoring):
    """
    cross_val_score wrapper that catches exceptions and returns nan if failing.
    """
    try:
        scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, error_score=np.nan)
        return np.array(scores)
    except Exception as e:
        logger.warning("cross_val_score 실패: %s", e)
        return np.array([np.nan] * (cv if hasattr(cv, "__len__") else (cv if isinstance(cv, int) else 3)))


def quick_baselines(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    cv: int = 3,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    빠른 베이스라인 모델 비교.
    - classification: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC
    - regression: LinearRegression, RandomForestRegressor, GradientBoostingRegressor

    반환: (results_df, trained_models)
      results_df: pandas.DataFrame (model, mean_score, std_score, task, notes)
      trained_models: dict 모델명->fit된 estimator (fit on full data)
    """
    X, y, task, feature_names = get_X_y(df, target_col=target_col)

    results = []
    models = {}

    if task == "classification":
        # use stratified kfold
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        # choose scorings: accuracy, f1_macro; add roc_auc for binary
        scorers = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
        try:
            is_binary = len(np.unique(y[~pd.isnull(y)])) == 2
        except Exception:
            is_binary = False
        if is_binary:
            scorers["roc_auc"] = "roc_auc"

        candidate_models = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
            "SVC": SVC(probability=is_binary, random_state=random_state),
        }

        for name, clf in candidate_models.items():
            row = {"model": name, "task": "classification"}
            # compute primary metric (accuracy)
            primary = _safe_cross_val_score(clf, X, y, cv=cv_split, scoring="accuracy")
            row["mean_accuracy"] = float(np.nanmean(primary)) if primary.size > 0 else np.nan
            row["std_accuracy"] = float(np.nanstd(primary)) if primary.size > 0 else np.nan
            # compute f1_macro
            f1 = _safe_cross_val_score(clf, X, y, cv=cv_split, scoring="f1_macro")
            row["mean_f1_macro"] = float(np.nanmean(f1))
            row["std_f1_macro"] = float(np.nanstd(f1))
            # roc_auc if binary
            if is_binary:
                try:
                    roc = _safe_cross_val_score(clf, X, y, cv=cv_split, scoring="roc_auc")
                    row["mean_roc_auc"] = float(np.nanmean(roc))
                    row["std_roc_auc"] = float(np.nanstd(roc))
                except Exception:
                    row["mean_roc_auc"] = np.nan
                    row["std_roc_auc"] = np.nan
            # fit on full data for return (best-effort)
            try:
                clf.fit(X, y)
                models[name] = {"model": clf, "feature_names": feature_names}
            except Exception as e:
                logger.warning("모델 학습 실패(%s): %s", name, e)
                models[name] = {"model": None, "feature_names": feature_names, "error": str(e)}
            results.append(row)

    elif task == "regression":
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        candidate_models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        }
        for name, reg in candidate_models.items():
            row = {"model": name, "task": "regression"}
            # use R^2 as primary metric
            r2 = _safe_cross_val_score(reg, X, y, cv=cv_split, scoring="r2")
            row["mean_r2"] = float(np.nanmean(r2)) if r2.size > 0 else np.nan
            row["std_r2"] = float(np.nanstd(r2)) if r2.size > 0 else np.nan
            # fit on full data
            try:
                reg.fit(X, y)
                models[name] = {"model": reg, "feature_names": feature_names}
            except Exception as e:
                logger.warning("모델 학습 실패(%s): %s", name, e)
                models[name] = {"model": None, "feature_names": feature_names, "error": str(e)}
            results.append(row)

    else:
        raise ValueError("타깃을 찾을 수 없거나 문제 유형 판별 실패했습니다. target_col을 지정하세요.")

    results_df = pd.DataFrame(results)
    # sort by primary metric (classification: mean_accuracy desc, regression: mean_r2 desc)
    if not results_df.empty:
        if task == "classification" and "mean_accuracy" in results_df.columns:
            results_df = results_df.sort_values(by="mean_accuracy", ascending=False).reset_index(drop=True)
        if task == "regression" and "mean_r2" in results_df.columns:
            results_df = results_df.sort_values(by="mean_r2", ascending=False).reset_index(drop=True)

    return results_df, models
