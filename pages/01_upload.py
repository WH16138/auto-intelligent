# pages/01_upload.py
"""
Upload page for AutoML-Edu Streamlit app.

Features:
- CSV 업로드 (file-like or path)
- 샘플 데이터 로드 (breast_cancer, iris)
- 안전한 읽기: modules.ingestion.safe_read_csv 사용 (있을 경우)
- 데이터 미리보기 및 간단 통계
- 데이터 지문(fingerprint) 생성 버튼 (modules.ingestion.generate_fingerprint)
- 타깃(target) 컬럼 선택/수정 (st.session_state['target_col'])
"""

import streamlit as st
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title("1 - Upload (데이터 수집)")
st.caption("CSV를 올리거나 샘플 데이터로 바로 연습하세요. 업로드하면 기본 train/test 분할이 자동 설정됩니다.")

# Ensure session keys used by app exist
for key in [
    "df_original",
    "df_dropped",
    "df_preprocessed",
    "df_features",
    "df",
    "fingerprint",
    "target_col",
    "train_idx",
    "test_idx",
    "split_meta",
]:
    st.session_state.setdefault(key, None)

# Try import ingestion module (required)
try:
    from modules import ingestion
except Exception:
    import modules.ingestion as ingestion  # fallback

col1, col2 = st.columns([1.4, 1])

with col1:
    st.header("CSV 업로드")
    uploaded = st.file_uploader("CSV 파일을 업로드하세요 (또는 아래 샘플 데이터 사용)", type=["csv"], accept_multiple_files=False, key="uploader_page")
    st.markdown("- 인코딩 문제 시 `safe_read_csv`가 여러 인코딩을 시도합니다.\n- 업로드 후 즉시 train/test 분할을 저장합니다.")
    if uploaded is not None:
        try:
            # uploaded is a file-like object; pass directly to safe_read_csv which supports file-like
            try:
                df = ingestion.safe_read_csv(uploaded)
            except Exception:
                # fallback to pandas default read (attempt with utf-8, cp949)
                try:
                    df = pd.read_csv(uploaded)
                except Exception:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, encoding="utf-8", engine="python")
            # Set all stage data to the original upload
            st.session_state["df_original"] = df.copy()
            st.session_state["df_dropped"] = df.copy()
            st.session_state["df_preprocessed"] = df.copy()
            st.session_state["df_features"] = df.copy()
            st.session_state["df"] = df.copy()  # backward compatibility
            try:
                idx = list(range(len(df)))
                train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
                st.session_state["train_idx"] = train_idx
                st.session_state["test_idx"] = test_idx
                st.session_state["split_meta"] = {"test_size": 0.2, "random_state": 42, "stratify": False}
            except Exception:
                st.session_state["train_idx"] = None
                st.session_state["test_idx"] = None
                st.session_state["split_meta"] = {"test_size": 0.2, "random_state": 42, "stratify": False}
            st.success(f"CSV 업로드 성공: {df.shape[0]}행 × {df.shape[1]}열 (train/test 분할 저장)")
        except Exception as e:
            st.session_state["df"] = None
            st.error(f"CSV 읽기 실패: {e}")

    st.markdown("---")
    st.header("샘플 데이터")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("샘플 로드: breast_cancer"):
            try:
                df = ingestion.load_sample("breast_cancer")
                st.session_state["df_original"] = df.copy()
                st.session_state["df_dropped"] = df.copy()
                st.session_state["df_preprocessed"] = df.copy()
                st.session_state["df_features"] = df.copy()
                st.session_state["df"] = df.copy()
                try:
                    idx = list(range(len(df)))
                    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
                    st.session_state["train_idx"] = train_idx
                    st.session_state["test_idx"] = test_idx
                    st.session_state["split_meta"] = {"test_size": 0.2, "random_state": 42, "stratify": False}
                except Exception:
                    st.session_state["train_idx"] = None
                    st.session_state["test_idx"] = None
                    st.session_state["split_meta"] = {"test_size": 0.2, "random_state": 42, "stratify": False}
                st.success("breast_cancer 샘플 로드 완료 (train/test 분할 저장)")
            except Exception as e:
                st.error(f"샘플 로드 실패: {e}")
    with c2:
        if st.button("샘플 로드: iris"):
            try:
                df = ingestion.load_sample("iris")
                st.session_state["df_original"] = df.copy()
                st.session_state["df_dropped"] = df.copy()
                st.session_state["df_preprocessed"] = df.copy()
                st.session_state["df_features"] = df.copy()
                st.session_state["df"] = df.copy()
                try:
                    idx = list(range(len(df)))
                    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
                    st.session_state["train_idx"] = train_idx
                    st.session_state["test_idx"] = test_idx
                    st.session_state["split_meta"] = {"test_size": 0.2, "random_state": 42, "stratify": False}
                except Exception:
                    st.session_state["train_idx"] = None
                    st.session_state["test_idx"] = None
                    st.session_state["split_meta"] = {"test_size": 0.2, "random_state": 42, "stratify": False}
                st.success("iris 샘플 로드 완료 (train/test 분할 저장)")
            except Exception as e:
                st.error(f"샘플 로드 실패: {e}")

    st.markdown("---")
    if st.button("세션 데이터 제거 (완전 초기화)"):
        # Clear all session entries then set expected keys to None so 다른 페이지도 안전하게 동작
        st.session_state.clear()
        for k in [
            "df_original",
            "df_dropped",
            "df_preprocessed",
            "df_features",
            "df",
            "fingerprint",
            "target_col",
            "preprocessing_pipeline",
            "feature_engineering_meta",
            "baselines_df",
            "baseline_models",
            "best_model_name",
            "trained_model",
            "problem_type",
            "hpo_result",
            "validation_result",
            "train_idx",
            "test_idx",
        ]:
            st.session_state[k] = None
        st.success("세션 데이터를 모두 삭제했습니다. 이어서 새 데이터를 업로드하거나 샘플을 로드하세요.")

with col2:
    st.header("업로드 상태")
    if st.session_state.get("df") is None:
        st.info("아직 업로드된 데이터가 없습니다.")
    else:
        df = st.session_state["df"]
        st.metric("행 수", df.shape[0])
        st.metric("열 수", df.shape[1])
        n_missing = int(df.isnull().sum().sum())
        st.metric("전체 결측 개수", n_missing)
        st.write("---")
        st.subheader("타깃 & 지문")

        # 타깃 설정을 가까이 배치
        cols = list(df.columns)
        current_target = st.session_state.get("target_col")
        sel = st.selectbox("타깃 컬럼 (모델 학습 시 사용)", options=["(없음)"] + cols, index=(0 if current_target is None else (cols.index(current_target) + 1)), key="target_select_box")
        if sel == "(없음)":
            st.session_state["target_col"] = None
        else:
            st.session_state["target_col"] = sel
        if st.session_state.get("target_col"):
            st.caption(f"현재 선택된 타깃: `{st.session_state.get('target_col')}`")

        # Fingerprint 버튼과 요약을 한 곳에
        fp = st.session_state.get("fingerprint")
        col_fp_btn, col_fp_preview = st.columns([1, 2])
        with col_fp_btn:
            if st.button("데이터 지문 생성/갱신", help="컬럼 통계·결측·추정 타깃 정보를 담은 요약을 만듭니다."):
                try:
                    fp = ingestion.generate_fingerprint(df)
                    st.session_state["fingerprint"] = fp
                    inferred = fp.get("target_column")
                    if inferred and st.session_state.get("target_col") is None:
                        st.session_state["target_col"] = inferred
                    st.success("지문 생성 완료")
                except Exception as e:
                    st.error(f"지문 생성 실패: {e}")
        with col_fp_preview:
            if fp:
                fp_preview = {
                    "n_rows": fp.get("n_rows"),
                    "n_cols": fp.get("n_cols"),
                    "missing_ratio": fp.get("missing_ratio"),
                    "inferred_target": fp.get("target_column"),
                }
                st.caption("지문 요약: 행/열/결측률/추정 타깃")
                st.json(fp_preview)
                with st.expander("지문 전체 보기 (길면 접어서 확인)", expanded=False):
                    st.json(fp)
            else:
                st.info("지문이 없습니다. 위 버튼으로 생성하세요.")

# Full width preview and fingerprint controls
st.write("### 데이터 미리보기 / 지문 생성")
if st.session_state.get("df") is None:
    st.info("왼쪽에서 CSV를 업로드하거나 샘플 데이터를 로드하세요.")
else:
    df = st.session_state["df"]
    # show head and basic describe
    st.subheader("샘플 미리보기 (상위 100행)")
    st.dataframe(df.head(100))

    # basic stats
    with st.expander("기본 통계 (describe, dtypes)"):
        try:
            st.write(df.describe(include="all").T)
            dtypes = {c: str(df[c].dtype) for c in df.columns}
            st.write("dtypes:", dtypes)
        except Exception as e:
            st.write("기본 통계 생성 실패:", e)

# Download preview as CSV
if st.session_state.get("df") is not None:
    csv_bytes = st.session_state["df"].head(100).to_csv(index=False).encode("utf-8")
    st.download_button("미리보기 CSV 다운로드 (상위 100행)", data=csv_bytes, file_name="preview.csv", mime="text/csv")

# small troubleshooting / help
st.write("---")
st.write("**도움말**")
st.markdown(
    """
- CSV 업로드 중 인코딩 오류가 발생하면, 파일을 UTF-8로 변환하거나 다른 인코딩으로 재시도하세요.
- `safe_read_csv`는 여러 인코딩을 시도하므로 대부분의 CSV에서 동작합니다.
- 타깃 컬럼은 이후 모델 선택/튜닝 단계에서 중요하니 정확히 선택하세요.
"""
)
