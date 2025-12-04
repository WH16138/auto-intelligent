"""
modules/preprocessing.py

전처리 파이프라인 빌더 및 간단한 전처리 래퍼.

주요 함수
- build_preprocessor(df, numeric_imputer='median', categorical_imputer='most_frequent', scale_numeric=True, onehot_threshold=20)
    : DataFrame으로부터 ColumnTransformer 기반 preprocessor를 구성하고 반환.

- simple_preprocess(df)
    : demo용, build_preprocessor로 preprocessor를 만들고 fit_transform하여 (df_transformed, preprocessor) 반환.

- apply_preprocessor(preprocessor, df)
    : 이미 학습된 preprocessor에 대해 transform하고 DataFrame으로 반환.

- save_pipeline(preprocessor, path)
    : joblib으로 저장.
"""
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib
import logging

logger = logging.getLogger(__name__)


def _detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # datetime detection
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    # exclude datetime from cat/num lists
    num_cols = [c for c in num_cols if c not in datetime_cols]
    cat_cols = [c for c in cat_cols if c not in datetime_cols]
    return {"numeric": num_cols, "categorical": cat_cols, "datetime": datetime_cols}


def _make_feature_names(num_cols: List[str], cat_cols: List[str], onehot_encoder: Optional[OneHotEncoder]) -> List[str]:
    """
    ColumnTransformer의 출력에 맞는 feature names 생성기.
    OneHotEncoder가 있으면 범주형 컬럼의 카테고리를 사용해서 확장된 이름을 생성합니다.
    """
    feature_names: List[str] = []
    # numeric keep same names
    feature_names.extend(num_cols)
    # categorical expansion
    if cat_cols:
        if onehot_encoder is None:
            # ordinal encoder case - keep original names
            feature_names.extend(cat_cols)
        else:
            # onehot encoder present: extract categories_
            try:
                categories = onehot_encoder.categories_
                for col, cats in zip(cat_cols, categories):
                    # sanitize category names to string
                    for cat in cats:
                        feature_names.append(f"{col}___{str(cat)}")
            except Exception as e:
                # fallback: use col names
                logger.warning("OneHotEncoder 카테고리 추출 실패: %s", e)
                feature_names.extend(cat_cols)
    return feature_names


def build_preprocessor(
    df: pd.DataFrame,
    numeric_imputer: str = "median",
    categorical_imputer: str = "most_frequent",
    scale_numeric: bool = True,
    use_onehot: bool = True,
    onehot_threshold: int = 20,
) -> ColumnTransformer:
    """
    DataFrame을 입력으로 받아 ColumnTransformer를 구성하여 반환.

    인자:
    - df: 데이터 샘플 (DataFrame)
    - numeric_imputer: 숫자 결측 대체 전략 ('median'|'mean'|'constant')
    - categorical_imputer: 범주형 결측 대체 전략 ('most_frequent'|'constant')
    - scale_numeric: 숫자형에 StandardScaler 적용 여부
    - use_onehot: 범주형에 OneHotEncoder 사용 여부 (False면 OrdinalEncoder 사용)
    - onehot_threshold: 유일값 수가 이 값을 초과하는 범주형은 Ordinal로 처리 (카디널리티 완화)

    반환:
    - sklearn.compose.ColumnTransformer (fitted하지 않은 상태)
    """
    types = _detect_column_types(df)
    num_cols = types["numeric"]
    cat_cols = types["categorical"]

    # Filter categorical columns by cardinality for one-hot
    onehot_cols = []
    ordinal_cols = []
    for c in cat_cols:
        nuniq = int(df[c].nunique(dropna=True))
        if use_onehot and nuniq <= onehot_threshold:
            onehot_cols.append(c)
        else:
            ordinal_cols.append(c)

    # numeric pipeline
    num_steps = []
    if numeric_imputer:
        num_steps.append(("imputer", SimpleImputer(strategy=numeric_imputer)))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    from sklearn.pipeline import Pipeline as SkPipeline
    num_pipeline = SkPipeline(num_steps) if num_steps else "passthrough"

    # categorical pipelines
    cat_pipelines = []
    if onehot_cols:
        cat_pipelines.append(
            ("onehot", Pipeline([("imputer", SimpleImputer(strategy=categorical_imputer, fill_value="__missing__")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))]), onehot_cols)
        )
    if ordinal_cols:
        cat_pipelines.append(
            ("ordinal", Pipeline([("imputer", SimpleImputer(strategy=categorical_imputer, fill_value="__missing__")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), ordinal_cols)
        )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    # add categorical transformers
    transformers.extend(cat_pipelines)

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)
    return preprocessor


def _get_feature_names_from_column_transformer(ct: ColumnTransformer, df_input: pd.DataFrame) -> List[str]:
    """
    ColumnTransformer로부터 출력 feature names를 추출.
    scikit-learn 버전 차이를 고려해 시도/폴백 처리.
    """
    feature_names: List[str] = []
    # numeric columns
    types = _detect_column_types(df_input)
    num_cols = types["numeric"]
    cat_cols = types["categorical"]

    # try to extract onehot encoder if present
    onehot_enc = None
    for name, transformer, cols in ct.transformers_:
        # transformers_ may include ('remainder', 'drop', ...) entries
        if name == "num":
            continue
        if isinstance(transformer, Pipeline):
            # the last step may be OneHotEncoder or OrdinalEncoder
            last_step = transformer.steps[-1][1]
            if isinstance(last_step, OneHotEncoder):
                onehot_enc = last_step
                # find which cols correspond to this encoder
                # assume cols variable above corresponds
                onehot_cols_for_encoder = cols
                # feature names for this encoder will be handled below
                break

    # build list combining numeric and categorical expansion
    # Build mapping for onehot encoders if available
    # The ColumnTransformer may contain multiple categorical transformers; we inspect each
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            feature_names.extend(cols)
        else:
            if isinstance(transformer, Pipeline):
                last = transformer.steps[-1][1]
                if isinstance(last, OneHotEncoder):
                    # categories_ available
                    try:
                        cats = last.categories_
                        for col_name, cats_for_col in zip(cols, cats):
                            for cat in cats_for_col:
                                feature_names.append(f"{col_name}___{cat}")
                    except Exception:
                        # fallback: append column name
                        feature_names.extend(cols)
                else:
                    # ordinal or other: keep original col names
                    feature_names.extend(cols)
            else:
                # passthrough or unknown
                feature_names.extend(cols)
    return feature_names


def apply_preprocessor(preprocessor: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """
    이미 생성된(preprocessor) ColumnTransformer에 대해 transform을 수행하고 DataFrame으로 반환.
    반환된 DataFrame의 컬럼명은 전처리 후 피처명(OneHot 확장 포함)입니다.
    """
    if preprocessor is None:
        raise ValueError("preprocessor가 None입니다.")
    # fit_transform if not fitted, else transform (best-effort)
    try:
        # check if fitted by accessing attribute
        fitted = hasattr(preprocessor, "transformers_") and all([hasattr(t, "transform") for _, t, _ in preprocessor.transformers_ if t != "drop"])
    except Exception:
        fitted = False

    if not fitted:
        X_trans = preprocessor.fit_transform(df)
    else:
        X_trans = preprocessor.transform(df)

    # Ensure numpy array
    if isinstance(X_trans, pd.DataFrame):
        return X_trans
    if hasattr(X_trans, "toarray"):  # sparse matrix
        X_arr = X_trans.toarray()
    else:
        X_arr = np.asarray(X_trans)

    # derive feature names
    try:
        feature_names = _get_feature_names_from_column_transformer(preprocessor, df)
    except Exception as e:
        logger.warning("전처리 후 feature names 생성 실패: %s", e)
        # fallback: generate generic names
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

    df_out = pd.DataFrame(X_arr, columns=feature_names)
    return df_out


def simple_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    데모용 간단 전처리: 자동 컬럼 감지 -> ColumnTransformer 빌드 -> fit_transform -> 반환
    반환: (df_transformed, preprocessor)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("DataFrame을 입력으로 주세요.")
    pre = build_preprocessor(df)
    df_trans = apply_preprocessor(pre, df)
    return df_trans, pre


def save_pipeline(preprocessor: Any, path: str) -> None:
    """
    preprocessor (또는 전체 Pipeline)을 joblib로 저장.
    """
    joblib.dump(preprocessor, path)
    logger.info("파이프라인을 %s에 저장했습니다.", path)
