"""
modules/feature_engineering.py

간단한 자동/반자동 특징공학 유틸 (MVP 용).

주요 함수:
- auto_generate_features(df, max_new=10, methods=None, datetime_extract=True)
    : 입력 DataFrame에서 안전한 규칙 기반 파생 피처를 생성하여 확장된 DataFrame 반환.
      반환값: (df_extended, meta) where meta describes newly created features.

- select_features_by_importance(df, target_col, model='RandomForest', top_k=20)
    : 임시 RandomForest 기반으로 피처 중요도를 계산하고 상위 top_k 피처명을 반환.

- featuretools_generate(df, target_col=None, max_depth=1, max_features=20)
    : optional: featuretools가 설치되어 있으면 Deep Feature Synthesis를 사용해 피처 생성 (없으면 에러 아님).

설계 원칙:
- 숫자형 중심의 안전한 조합(차, 비율, 로그)과 datetime 추출 제공
- 생성 피처 수를 max_new로 제한
- 고카디널리티 범주에는 주의(빈도/타깃 인코딩 옵션 제공하지는 않음)
- 외부 heavy dependency(featuretools)는 optional
"""
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# -------------------------
# Basic transform helpers
# -------------------------
def _safe_log1p(series: pd.Series) -> pd.Series:
    """log1p 변환, 음수/무한 처리 안전하게"""
    try:
        arr = pd.to_numeric(series, errors='coerce').fillna(0).astype(float)
        # shift if negatives exist
        minv = arr.min()
        if minv <= -1:
            shift = abs(minv) + 1.0
            arr = arr + shift
        return np.log1p(arr)
    except Exception as e:
        logger.debug("log1p 실패: %s", e)
        return pd.Series(np.zeros(len(series)), index=series.index)

def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    """0으로 나누는 경우 안전 처리"""
    a_num = pd.to_numeric(a, errors='coerce').fillna(0)
    b_num = pd.to_numeric(b, errors='coerce').fillna(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = a_num / b_num
        res = res.replace([np.inf, -np.inf], np.nan).fillna(0)
    return res

# -------------------------
# Core: auto feature gen
# -------------------------
def auto_generate_features(
    df: pd.DataFrame,
    max_new: int = 10,
    methods: Optional[List[str]] = None,
    datetime_extract: bool = True,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    규칙 기반 자동 피처 생성.

    Args:
        df: 입력 DataFrame (원본을 복사해 반환)
        max_new: 생성할 최대 피처 수 (초과하지 않음)
        methods: 사용할 변환 목록. 지원값: ['ratio','diff','sum','log','datetime']
                 None이면 기본 ['ratio','diff','log','datetime'] 사용 (datetime은 datetime_extract에 따름)
        datetime_extract: datetime 타입 컬럼에서 연/월/일/요일/시간 추출
        random_state: 내부 샘플링 시드

    Returns:
        (df_extended, meta) where meta = {'new_features': [names], 'method': {name: method}}
    """
    if methods is None:
        methods = ['ratio', 'diff', 'log', 'datetime']
    df_out = df.copy()
    new_features = []
    meta = {"new_features": [], "method": {}}

    # numeric candidate columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # datetime candidate columns
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    # 1) datetime extraction
    if datetime_extract and 'datetime' in methods and dt_cols:
        for c in dt_cols:
            if len(new_features) >= max_new:
                break
            try:
                s = pd.to_datetime(df[c], errors='coerce')
                for attr in ["year", "month", "day", "hour", "weekday"]:
                    colname = f"{c}__{attr}"
                    if colname in df_out.columns:
                        continue
                    if attr == "weekday":
                        df_out[colname] = s.dt.weekday.fillna(-1).astype(int)
                    elif attr == "hour":
                        df_out[colname] = s.dt.hour.fillna(-1).astype(int)
                    else:
                        df_out[colname] = getattr(s.dt, attr).fillna(-1).astype(int)
                    new_features.append(colname)
                    meta["new_features"].append(colname)
                    meta["method"][colname] = "datetime_extract"
                    if len(new_features) >= max_new:
                        break
            except Exception as e:
                logger.debug("datetime 추출 실패(%s): %s", c, e)

    # refresh numeric columns after potential new features
    num_cols = [c for c in df_out.select_dtypes(include=["number"]).columns.tolist() if c not in (new_features)]

    # 2) pairwise combos: ratio, difference, sum
    # choose a small set of pairs to avoid explosion: pick top-k numeric cols by variance
    if num_cols and any(m in methods for m in ['ratio', 'diff', 'sum']):
        variances = df_out[num_cols].var().sort_values(ascending=False)
        candidates = variances.index.tolist()
        # limit candidate pool to top 10
        pool = candidates[:min(10, len(candidates))]
        pairs = []
        for i in range(len(pool)):
            for j in range(i+1, len(pool)):
                pairs.append((pool[i], pool[j]))
        # deterministic order
        pairs = pairs[:max(0, max_new * 5)]
        for (a, b) in pairs:
            if len(new_features) >= max_new:
                break
            # ratio a/b
            if 'ratio' in methods:
                try:
                    fname = f"{a}__div__{b}"
                    if fname not in df_out.columns:
                        df_out[fname] = _safe_divide(df_out[a], df_out[b])
                        new_features.append(fname)
                        meta["new_features"].append(fname)
                        meta["method"][fname] = "ratio"
                        if len(new_features) >= max_new:
                            break
                except Exception as e:
                    logger.debug("ratio 생성 실패 %s/%s: %s", a, b, e)
            # diff a-b
            if 'diff' in methods and len(new_features) < max_new:
                try:
                    fname = f"{a}__minus__{b}"
                    if fname not in df_out.columns:
                        df_out[fname] = pd.to_numeric(df_out[a], errors='coerce').fillna(0) - pd.to_numeric(df_out[b], errors='coerce').fillna(0)
                        new_features.append(fname)
                        meta["new_features"].append(fname)
                        meta["method"][fname] = "diff"
                        if len(new_features) >= max_new:
                            break
                except Exception as e:
                    logger.debug("diff 생성 실패 %s-%s: %s", a, b, e)
            # sum a+b
            if 'sum' in methods and len(new_features) < max_new:
                try:
                    fname = f"{a}__plus__{b}"
                    if fname not in df_out.columns:
                        df_out[fname] = pd.to_numeric(df_out[a], errors='coerce').fillna(0) + pd.to_numeric(df_out[b], errors='coerce').fillna(0)
                        new_features.append(fname)
                        meta["new_features"].append(fname)
                        meta["method"][fname] = "sum"
                        if len(new_features) >= max_new:
                            break
                except Exception as e:
                    logger.debug("sum 생성 실패 %s+%s: %s", a, b, e)

    # 3) log transforms for skewed numeric features
    if 'log' in methods and len(new_features) < max_new:
        # compute skew and pick top skewed numeric cols
        try:
            num_cols_all = df_out.select_dtypes(include=["number"]).columns.tolist()
            skews = []
            for c in num_cols_all:
                try:
                    s = pd.to_numeric(df_out[c], errors='coerce').dropna()
                    if s.size > 2:
                        skews.append((c, float(s.skew())))
                except Exception:
                    continue
            skews.sort(key=lambda x: -abs(x[1]))
            for c, sk in skews:
                if len(new_features) >= max_new:
                    break
                fname = f"{c}__log1p"
                if fname in df_out.columns:
                    continue
                try:
                    df_out[fname] = _safe_log1p(df_out[c])
                    new_features.append(fname)
                    meta["new_features"].append(fname)
                    meta["method"][fname] = "log1p"
                except Exception as e:
                    logger.debug("log 변환 실패 %s: %s", c, e)
        except Exception as e:
            logger.debug("log 변환 단계 실패: %s", e)

    # finalize: ensure dtype sanity, limit total new features
    if len(new_features) > max_new:
        # keep first max_new
        to_drop = new_features[max_new:]
        for col in to_drop:
            try:
                df_out.drop(columns=[col], inplace=True)
                meta["new_features"].remove(col)
                meta["method"].pop(col, None)
            except Exception:
                pass
        new_features = new_features[:max_new]

    meta["n_created"] = len(meta["new_features"])
    return df_out, meta

# -------------------------
# Feature selection by importance
# -------------------------
def select_features_by_importance(
    df: pd.DataFrame,
    target_col: str,
    model: str = 'RandomForest',
    top_k: int = 20,
    random_state: int = 0
) -> List[str]:
    """
    간단한 feature importance 기반 선택. 숫자형 피처를 사용하여 RandomForest 학습 후 중요도 상위 top_k 반환.

    Args:
        df: DataFrame (target_col 포함)
        target_col: 타깃 컬럼명
        model: 'RandomForest' (현재 한정)
        top_k: 반환할 피처 수
    Returns:
        List of feature names (length <= top_k)
    """
    if target_col not in df.columns:
        raise ValueError("target_col이 DataFrame에 존재하지 않습니다.")
    # use numeric features
    X = df.select_dtypes(include=["number"]).drop(columns=[target_col], errors='ignore')
    if X.shape[1] == 0:
        # fallback: try to convert object columns via factorize (not ideal)
        obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not obj_cols:
            return []
        X = pd.DataFrame()
        for c in obj_cols:
            X[c] = pd.factorize(df[c])[0]
    y = df[target_col]
    # drop constant columns
    nunique = X.nunique(dropna=True)
    nonconst_cols = nunique[nunique > 1].index.tolist()
    X = X[nonconst_cols]
    if X.shape[1] == 0:
        return []

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        # detect classification vs regression
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            # regression
            clf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        clf.fit(X.fillna(0).values, y.values)
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return X.columns.tolist()[:top_k]
        imp_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        return imp_series.head(top_k).index.tolist()
    except Exception as e:
        logger.warning("feature importance 계산 실패: %s", e)
        # fallback: return top_k numeric cols by variance
        vars_ = X.var().sort_values(ascending=False)
        return vars_.head(top_k).index.tolist()

# -------------------------
# Optional integration with featuretools (if installed)
# -------------------------
def featuretools_generate(df: pd.DataFrame, target_col: Optional[str] = None, max_depth: int = 1, max_features: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    featuretools의 Deep Feature Synthesis를 사용해 피처를 생성합니다.
    이 함수는 featuretools가 설치되어 있을 때만 동작하며, 설치되지 않은 경우 ImportError를 발생시킵니다.

    Args:
        df: DataFrame
        target_col: optional (featuretools EntitySet/relationship을 만들 때 사용)
        max_depth: dfs max depth
        max_features: 생성 최대 피처 수(결과를 자름)

    Returns:
        (df_extended, meta)
    """
    try:
        import featuretools as ft
    except Exception as e:
        raise ImportError("featuretools가 설치되어 있지 않습니다. pip install featuretools") from e

    # simple single-table DFS
    es = ft.EntitySet(id="es")
    # ensure index
    if "__index" not in df.columns:
        df2 = df.reset_index().rename(columns={"index": "__index"})
    else:
        df2 = df.copy()
    es = es.add_dataframe(dataframe_name="df", dataframe=df2, index="__index")
    # perform dfs
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name="df", max_depth=max_depth, n_jobs=1)
    # limit features
    if feature_matrix.shape[1] > max_features:
        feature_matrix = feature_matrix.iloc[:, :max_features]
    meta = {"n_created": feature_matrix.shape[1], "feature_defs": [str(fd) for fd in feature_defs[:max_features]]}
    return feature_matrix.reset_index(drop=True), meta
