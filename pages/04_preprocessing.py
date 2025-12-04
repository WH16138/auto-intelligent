# pages/04_preprocessing.py
"""
04 - Preprocessing 페이지

기능:
- 전처리 파이프라인 옵션 선택 (숫자/범주 결측치, 스케일링, OneHot 여부, OneHot threshold)
- 간이(preview) 전처리 적용 (simple_preprocess 또는 build_preprocessor + apply_preprocessor)
- 전처리 결과 미리보기(상위 N행)
- 전처리 파이프라인 저장 (joblib)
- 세션 상태 업데이트: st.session_state['df_preprocessed'], ['preprocessor']
"""
import streamlit as st
import pandas as pd
import io
import json
from typing import Optional

st.set_page_config(layout="wide")
st.title("4 - Preprocessing (전처리)")

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_preprocessed" not in st.session_state:
    st.session_state["df_preprocessed"] = None
if "preprocessing_pipeline" not in st.session_state:
    st.session_state["preprocessing_pipeline"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None

# imports (fallback)
try:
    from modules import preprocessing as prep
except Exception:
    import modules.preprocessing as prep  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore

df: Optional[pd.DataFrame] = st.session_state.get("df")
if df is None:
    st.warning("먼저 Upload 페이지에서 데이터를 업로드하거나 샘플 데이터를 로드하세요.")
    st.stop()

# Sidebar: options
with st.sidebar:
    st.header("전처리 기본 옵션")
    numeric_imputer = st.selectbox("숫자형 결측 대체 전략", options=["median", "mean", "constant"], index=0)
    categorical_imputer = st.selectbox("범주형 결측 대체 전략", options=["most_frequent", "constant"], index=0)
    scale_numeric = st.checkbox("숫자형 스케일링 적용 (StandardScaler)", value=True)
    use_onehot = st.checkbox("범주형에 OneHot 인코더 사용", value=True)
    onehot_threshold = st.number_input("OneHot 허용 유일값 최대치 (이상이면 Ordinal 처리)", min_value=2, max_value=1000, value=20, step=1)
    apply_on_preview = st.checkbox("미리보기에서 옵션을 적용하여 결과 확인", value=True)
    save_default_pipeline_name = st.text_input("저장할 파이프라인 파일명 (artifacts 폴더)", value="artifacts/preprocessor.joblib")

st.markdown("### 현재 데이터 요약")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric("행 수", df.shape[0])
with col2:
    st.metric("열 수", df.shape[1])
with col3:
    n_missing = int(df.isnull().sum().sum())
    st.metric("전체 결측 개수", n_missing)

st.markdown("---")
st.subheader("전처리 미리보기")

# choose preview rows
preview_n = st.number_input("미리보기 행 수", min_value=5, max_value=1000, value=50, step=5)

# Build preprocessor preview (not persisted until '적용' 클릭)
if st.button("미리보기 전처리 적용"):
    try:
        # Build ColumnTransformer based on options
        pre = prep.build_preprocessor(
            df,
            numeric_imputer=numeric_imputer,
            categorical_imputer=categorical_imputer,
            scale_numeric=scale_numeric,
            use_onehot=use_onehot,
            onehot_threshold=int(onehot_threshold),
        )
        # Apply transform (fit_transform)
        df_trans = prep.apply_preprocessor(pre, df)
        st.session_state["df_preprocessed_preview"] = df_trans
        st.success("미리보기 전처리 적용 완료 — 아래에서 결과를 확인하세요.")
    except Exception as e:
        st.error(f"미리보기 전처리 실패: {e}")

# Show original and preview side-by-side
col_orig, col_new = st.columns(2)
with col_orig:
    st.caption("원본 데이터 (상위)")
    st.dataframe(df.head(preview_n))
with col_new:
    st.caption("전처리 미리보기 (상위)")
    if st.session_state.get("df_preprocessed_preview") is not None:
        st.dataframe(st.session_state["df_preprocessed_preview"].head(preview_n))
    else:
        st.info("미리보기 적용 버튼을 눌러 전처리 결과를 확인하세요.")

st.markdown("---")
st.subheader("전처리 적용 및 저장")

col_apply, col_actions = st.columns([2,1])
with col_apply:
    if st.button("전처리 적용하고 세션에 저장 (Fit & Transform)"):
        try:
            pre = prep.build_preprocessor(
                df,
                numeric_imputer=numeric_imputer,
                categorical_imputer=categorical_imputer,
                scale_numeric=scale_numeric,
                use_onehot=use_onehot,
                onehot_threshold=int(onehot_threshold),
            )
            df_trans = prep.apply_preprocessor(pre, df)
            # store pipeline and transformed df in session for later pages
            st.session_state["preprocessing_pipeline"] = pre
            st.session_state["df_preprocessed"] = df_trans
            st.success("전처리 적용 및 세션 저장 완료 (st.session_state['df_preprocessed']에 저장됨).")
            # show small preview
            st.dataframe(df_trans.head(preview_n))
        except Exception as e:
            st.error(f"전처리 적용 실패: {e}")

with col_actions:
    st.markdown("파이프라인 저장/내보내기")
    if st.button("파이프라인을 artifacts에 저장"):
        pre = st.session_state.get("preprocessing_pipeline")
        if pre is None:
            st.warning("저장할 파이프라인이 없습니다. 먼저 '전처리 적용'을 수행하세요.")
        else:
            try:
                # use provided path or default
                path = save_default_pipeline_name or "artifacts/preprocessor.joblib"
                # attempt to use preprocessing.save_pipeline if present
                if hasattr(prep, "save_pipeline"):
                    prep.save_pipeline(pre, path)
                else:
                    # fallback to io_utils.save_model
                    io_utils.save_model(pre, path)
                st.success(f"파이프라인이 저장되었습니다: {path}")
            except Exception as e:
                st.error(f"파이프라인 저장 실패: {e}")

    st.markdown("---")
    st.markdown("전처리 결과 다운로드")
    if st.session_state.get("df_preprocessed") is not None:
        csv_bytes = st.session_state["df_preprocessed"].to_csv(index=False).encode("utf-8")
        st.download_button("전처리된 데이터 다운로드 (CSV)", data=csv_bytes, file_name="df_preprocessed.csv", mime="text/csv")
    else:
        st.info("전처리된 데이터가 없습니다. 먼저 '전처리 적용'을 실행하세요.")

st.markdown("---")
st.write("다음 단계: 특징공학(Feature Engineering) 또는 모델 선택 페이지로 이동하세요.")
