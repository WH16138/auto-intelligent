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

st.set_page_config(layout="wide")

st.title("1 - Upload (데이터 수집)")

# Ensure session keys used by app exist
if "df" not in st.session_state:
    st.session_state["df"] = None
if "fingerprint" not in st.session_state:
    st.session_state["fingerprint"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None

# Try import ingestion module (required)
try:
    from modules import ingestion
except Exception:
    import modules.ingestion as ingestion  # fallback

col1, col2 = st.columns([2, 1])

with col1:
    st.header("CSV 업로드")
    uploaded = st.file_uploader("CSV 파일을 업로드하세요 (또는 아래 샘플 데이터 사용)", type=["csv"], accept_multiple_files=False, key="uploader_page")
    st.markdown("**옵션:** 파일이 크거나 인코딩 문제가 의심되면 `safe_read_csv`가 여러 인코딩을 시도합니다.")
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
            st.session_state["df"] = df
            st.success(f"CSV 업로드 성공: {df.shape[0]}행 × {df.shape[1]}열")
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
                st.session_state["df"] = df
                st.success("breast_cancer 샘플 로드 완료")
            except Exception as e:
                st.error(f"샘플 로드 실패: {e}")
    with c2:
        if st.button("샘플 로드: iris"):
            try:
                df = ingestion.load_sample("iris")
                st.session_state["df"] = df
                st.success("iris 샘플 로드 완료")
            except Exception as e:
                st.error(f"샘플 로드 실패: {e}")

    st.markdown("---")
    if st.button("세션 데이터 제거 (df 초기화)"):
        st.session_state["df"] = None
        st.session_state["fingerprint"] = None
        st.session_state["target_col"] = None
        st.success("세션의 데이터가 초기화되었습니다.")

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
        st.subheader("데이터 지문 (Fingerprint)")
        if st.session_state.get("fingerprint") is not None:
            st.json(st.session_state["fingerprint"])
        else:
            st.info("지문이 생성되지 않았습니다. 아래 '데이터 지문 생성' 버튼을 사용하세요.")

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

    # fingerprint generation and target selection
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if st.button("데이터 지문 생성 (Fingerprint)"):
            try:
                fp = ingestion.generate_fingerprint(df)
                st.session_state["fingerprint"] = fp
                # if fingerprint suggests target and no target selected yet, prefill
                inferred = fp.get("target_column")
                if inferred and st.session_state.get("target_col") is None:
                    st.session_state["target_col"] = inferred
                st.success("지문 생성 완료")
                st.json(fp)
            except Exception as e:
                st.error(f"지문 생성 실패: {e}")
    with col_b:
        # Target selection UI
        st.markdown("**타깃(target) 설정**")
        cols = list(df.columns)
        current_target = st.session_state.get("target_col")
        sel = st.selectbox("타깃 컬럼 선택 (모델 학습 시 사용)", options=["(없음)"] + cols, index=(0 if current_target is None else (cols.index(current_target) + 1)), key="target_select_box")
        if sel == "(없음)":
            st.session_state["target_col"] = None
        else:
            st.session_state["target_col"] = sel
        if st.session_state.get("target_col"):
            st.caption(f"현재 선택된 타깃: `{st.session_state.get('target_col')}`")

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
