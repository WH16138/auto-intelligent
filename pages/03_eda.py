# pages/03_eda.py
"""
03 - EDA (시각화 및 자동 요약) 페이지

기능:
- 자동 EDA 요약 텍스트(모듈 modules.eda.generate_eda_summary 사용)
- 타깃 분포 시각화(선택한 target이 있을 때)
- 컬럼 선택 기반 히스토그램 / 박스플롯 표시
- 상관관계 히트맵, 결측 행렬 표시
- EDA 리포트 HTML 생성 및 다운로드 (modules.io_utils.save_report_html 사용)
"""
import streamlit as st
import pandas as pd
from typing import Optional

st.set_page_config(layout="wide")
st.title("3 - EDA (탐색적 데이터 분석)")

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "fingerprint" not in st.session_state:
    st.session_state["fingerprint"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None

# imports (fallbacks)
try:
    from modules import eda as eda_mod
except Exception:
    import modules.eda as eda_mod  # type: ignore

try:
    from modules import visualization as viz
except Exception:
    import modules.visualization as viz  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore

df = st.session_state.get("df")
fingerprint = st.session_state.get("fingerprint")
target_col = st.session_state.get("target_col")

if df is None:
    st.warning("먼저 Upload 페이지에서 데이터를 업로드하거나 샘플 데이터를 로드하세요.")
    st.stop()

# --- Top summary and quick metrics ---
st.subheader("자동 요약")
with st.expander("자동 EDA 요약(요약문 + 구조화 정보)"):
    try:
        text_summary, summary_struct = eda_mod.generate_eda_summary(df, fingerprint=fingerprint, target_col=target_col, top_k=5)
        st.markdown("**요약:**")
        st.write(text_summary)
        st.markdown("**구조화된 요약:**")
        st.json(summary_struct)
    except Exception as e:
        st.error(f"EDA 요약 생성 실패: {e}")

# If target exists, show distribution
if target_col and target_col in df.columns:
    st.subheader(f"타깃 분포: `{target_col}`")
    try:
        fig_target = viz.hist_plotly(df, target_col, nbins=50, title=f"Target `{target_col}` distribution")
        st.plotly_chart(fig_target, use_container_width=True)
    except Exception as e:
        st.warning(f"타깃 분포 시각화 실패: {e}")

st.markdown("---")

# --- Column-based interactive plotting ---
st.subheader("컬럼별 시각화")
cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

col1, col2 = st.columns([2, 1])
with col1:
    sel_col = st.selectbox("플롯할 컬럼 선택", options=cols, index=0)
    if sel_col:
        try:
            fig_hist = eda_mod.plot_histogram(df, sel_col)
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.warning(f"히스토그램 생성 실패: {e}")

        if sel_col in numeric_cols:
            try:
                fig_box = eda_mod.plot_box(df, sel_col)
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.info("박스플롯 생성 불가 또는 해당 컬럼이 숫자형이 아님")

with col2:
    st.write("간단 통계")
    try:
        st.write(df[sel_col].describe(include="all"))
    except Exception:
        st.write("통계 계산 실패")

st.markdown("---")

# --- Correlation & Missing ---
st.subheader("상관관계 및 결측 패턴")
c1, c2 = st.columns(2)
with c1:
    try:
        fig_corr = eda_mod.plot_correlation_heatmap(df, numeric_only=True, top_n=30)
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"상관관계 히트맵 생성 실패: {e}")

with c2:
    try:
        fig_missing = eda_mod.plot_missing_matrix(df, max_cols=80)
        st.plotly_chart(fig_missing, use_container_width=True)
    except Exception as e:
        st.warning(f"결측 행렬 시각화 실패: {e}")

st.markdown("---")

# --- Quick pairwise sample (heavy; optional) ---
st.subheader("샘플 기반 산점도 매트릭스 (선택 사항)")
if st.button("샘플 산점도 생성 (시간 소요 가능)"):
    try:
        fig_pair = viz.pair_sample_plotly(df)
        st.plotly_chart(fig_pair, use_container_width=True)
    except Exception as e:
        st.warning(f"pairwise plot 생성 실패: {e}")

st.markdown("---")

# --- Export / Save report ---
st.subheader("EDA 리포트 생성 및 다운로드")
report_name = st.text_input("리포트 파일명 (확장자 제외)", value="eda_report")
if st.button("HTML 리포트 생성 및 다운로드"):
    try:
        html_report = eda_mod.create_eda_report_html(df, fingerprint=fingerprint, target_col=target_col)
        file_path = f"reports/{report_name}.html"
        io_utils.save_report_html(html_report, file_path)
        # read bytes for download
        with open(file_path, "rb") as f:
            data = f.read()
        st.download_button("EDA 리포트 다운로드 (HTML)", data=data, file_name=f"{report_name}.html", mime="text/html")
        st.success(f"리포트가 생성되어 reports/{report_name}.html 로 저장되었습니다.")
    except Exception as e:
        st.error(f"리포트 생성 실패: {e}")

st.markdown("---")
st.write("다음 단계: 전처리 페이지로 이동하여 결측/인코딩/스케일링을 적용하세요.")
