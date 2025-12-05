# pages/04_preprocessing.py
"""
4단계 - 전처리

안내:
- 타깃 컬럼은 전처리 입력에서 제외하고, 변환 후 다시 붙입니다(누수 방지).
- 전처리 설정을 본문 상단에서 바로 조정할 수 있습니다.
"""
import pandas as pd
import streamlit as st
from typing import Optional

st.set_page_config(layout="wide")
st.title("4 - 전처리")

# Session defaults
st.session_state.setdefault("df", None)
st.session_state.setdefault("df_preprocessed", None)
st.session_state.setdefault("preprocessing_pipeline", None)
st.session_state.setdefault("target_col", None)
st.session_state.setdefault("df_features", None)  # downstream 특징공학/모델 단계에서 사용

# Imports (fallbacks)
try:
    from modules import preprocessing as prep
except Exception:
    import modules.preprocessing as prep  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore

df: Optional[pd.DataFrame] = st.session_state.get("df")
target_col = st.session_state.get("target_col")

if df is None:
    st.warning("Upload 페이지에서 데이터를 불러온 후 다시 시도하세요.")
    st.stop()

# Config in main body
st.markdown("### 전처리 설정")
col_a, col_b, col_c = st.columns(3)
with col_a:
    numeric_imputer = st.selectbox("수치 결측 대치", options=["median", "mean", "constant"], index=0)
    scale_numeric = st.checkbox("수치 표준화 (StandardScaler)", value=True)
with col_b:
    categorical_imputer = st.selectbox("범주 결측 대치", options=["most_frequent", "constant"], index=0)
    use_onehot = st.checkbox("범주형 One-Hot 인코딩", value=True)
with col_c:
    onehot_threshold = st.number_input("One-Hot 최대 범주 수(초과 시 Ordinal)", min_value=2, max_value=1000, value=20, step=1)
    preview_n = st.number_input("미리보기 행수", min_value=5, max_value=1000, value=50, step=5)
save_default_pipeline_name = st.text_input("파이프라인 저장 경로", value="artifacts/preprocessor.joblib")

st.info("타깃 컬럼은 전처리 입력에서 제외하고, 변환 후 다시 붙입니다.")

st.markdown("---")
st.subheader("데이터 개요")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("행 수", df.shape[0])
with col2:
    st.metric("열 수", df.shape[1])
with col3:
    n_missing = int(df.isnull().sum().sum())
    st.metric("결측 총합", n_missing)

def _apply_pipeline(df_input: pd.DataFrame):
    """Build and apply preprocessor excluding target; return (df_trans, preprocessor)."""
    df_work = df_input.copy()
    tgt = None
    if target_col and target_col in df_work.columns:
        tgt = df_work[target_col].copy()
        df_work = df_work.drop(columns=[target_col])
    pre = prep.build_preprocessor(
        df_work,
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        scale_numeric=scale_numeric,
        use_onehot=use_onehot,
        onehot_threshold=int(onehot_threshold),
    )
    df_trans = prep.apply_preprocessor(pre, df_work)
    if tgt is not None:
        df_trans[target_col] = tgt.values
    return df_trans, pre

st.markdown("---")
st.subheader("전처리 미리보기")

if st.button("미리보기 생성"):
    try:
        df_trans, pre = _apply_pipeline(df)
        st.session_state["df_preprocessed_preview"] = df_trans
        st.success("미리보기 전처리를 완료했습니다.")
    except Exception as e:
        st.error(f"미리보기 실패: {e}")

col_orig, col_new = st.columns(2)
with col_orig:
    st.caption("원본 데이터 (상위)")
    st.dataframe(df.head(preview_n))
with col_new:
    st.caption("전처리 미리보기 (상위)")
    if st.session_state.get("df_preprocessed_preview") is not None:
        st.dataframe(st.session_state["df_preprocessed_preview"].head(preview_n))
    else:
        st.info("미리보기 버튼을 눌러주세요.")

st.markdown("---")
st.subheader("전처리 적용 및 저장")
col_apply, col_actions = st.columns([2, 1])
with col_apply:
    if st.button("전처리 적용 (Fit & Transform)"):
        try:
            df_trans, pre = _apply_pipeline(df)
            st.session_state["preprocessing_pipeline"] = pre
            st.session_state["df_preprocessed"] = df_trans
            # 전처리 결과를 기본 특징 데이터로 초기화
            st.session_state["df_features"] = df_trans
            st.success("전처리 완료: st.session_state['df_preprocessed'] 에 저장되었습니다.")
            st.dataframe(df_trans.head(preview_n))
        except Exception as e:
            st.error(f"전처리 적용 실패: {e}")

    if st.button("전처리 생략하고 원본 유지"):
        st.session_state["preprocessing_pipeline"] = None
        st.session_state["df_preprocessed"] = df.copy()
        st.session_state["df_features"] = df.copy()
        st.success("원본 데이터를 그대로 다음 단계에 사용합니다.")

with col_actions:
    st.markdown("파이프라인 저장")
    if st.button("파이프라인 저장 (artifacts)"):
        pre = st.session_state.get("preprocessing_pipeline")
        if pre is None:
            st.warning("먼저 '전처리 적용'을 수행하세요.")
        else:
            try:
                path = save_default_pipeline_name or "artifacts/preprocessor.joblib"
                if hasattr(prep, "save_pipeline"):
                    prep.save_pipeline(pre, path)
                else:
                    io_utils.save_model(pre, path)
                st.success(f"파이프라인을 저장했습니다: {path}")
            except Exception as e:
                st.error(f"파이프라인 저장 실패: {e}")

    st.markdown("---")
    st.markdown("전처리 결과 다운로드")
    if st.session_state.get("df_preprocessed") is not None:
        csv_bytes = st.session_state["df_preprocessed"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "전처리 결과 다운로드 (CSV)",
            data=csv_bytes,
            file_name="df_preprocessed.csv",
            mime="text/csv",
        )
    else:
        st.info("아직 전처리 결과가 없습니다.")

st.markdown("---")
st.write("다음 단계: Feature Engineering 페이지로 이동하세요.")
