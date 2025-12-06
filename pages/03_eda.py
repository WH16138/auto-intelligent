"""
03 - EDA (시각화 및 자동 요약) 페이지

기능:
- 자동 EDA 요약 텍스트(모듈 modules.eda.generate_eda_summary 사용)
- 타깃 분포 시각화(선택한 target이 있을 때)
- 컬럼 선택 기반 히스토그램 / 박스플롯 표시
- 상관관계 히트맵, 결측 행렬 표시
"""
import streamlit as st
import pandas as pd
from typing import Optional

st.set_page_config(layout="wide")
st.title("3 - EDA (탐색적 데이터 분석)")
st.info("순서 추천: ① 자동 요약 → ② 타깃 분포(있다면) → ③ 단일 컬럼 분포/박스 → ④ 상관·결측 → ⑤ (옵션) pairwise 그래프는 버튼을 눌러야 실행됩니다.")

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "fingerprint" not in st.session_state:
    st.session_state["fingerprint"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None

try:
    from modules import eda as eda_mod
except Exception:
    import modules.eda as eda_mod

try:
    from modules import visualization as viz
except Exception:
    import modules.visualization as viz

df = st.session_state.get("df")
fingerprint = st.session_state.get("fingerprint")
target_col = st.session_state.get("target_col")

if df is None:
    st.warning("먼저 Upload 페이지에서 데이터를 업로드하거나 샘플 데이터를 로드하세요.")
    st.stop()

# Top summary and quick metrics
st.subheader("자동 요약")
with st.expander("자동 EDA 요약(요약문 + 구조화 정보)", expanded=True):
    try:
        text_summary, summary_struct = eda_mod.generate_eda_summary(df, fingerprint=fingerprint, target_col=target_col, top_k=5)
        st.markdown("**요약(텍스트):**")
        st.write(text_summary)
        st.markdown("**요약(구조화)** — 길 경우 접어서 확인하세요.")
        with st.expander("구조화 요약 펼쳐보기", expanded=False):
            st.json(summary_struct)
    except Exception as e:
        st.error(f"EDA 요약 생성 실패: {e}")

# If target exists, show distribution
if target_col and target_col in df.columns:
    st.subheader(f"타깃 분포: `{target_col}`")
    st.caption("분류면 클래스 불균형 여부, 회귀면 분포 왜도 등을 빠르게 확인하세요.")
    try:
        fig_target = viz.hist_plotly(df, target_col, nbins=50, title=f"Target `{target_col}` distribution")
        st.plotly_chart(fig_target, use_container_width=True)
    except Exception as e:
        st.warning(f"타깃 분포 시각화 실패: {e}")

st.markdown("---")

# Column-based interactive plotting
st.subheader("컬럼별 시각화")
st.caption("컬럼을 하나 선택해 분포/박스플롯·기본 통계를 확인합니다.")
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

# Correlation & Missin
st.subheader("상관관계 및 결측 패턴")
st.caption("숫자형 상관관계와 결측 패턴을 확인해 다중공선성·결측 처리를 가이드합니다.")
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

# Quick pairwise sample (heavy; optional)
st.subheader("샘플 기반 산점도 매트릭스 (선택 사항)")
st.caption("주의: 컬럼/행이 많으면 느려질 수 있습니다. 소규모 데이터나 일부 컬럼만 있을 때 실행하세요.")
if st.button("샘플 산점도 생성 (시간 소요 가능)"):
    try:
        fig_pair = viz.pair_sample_plotly(df)
        st.plotly_chart(fig_pair, use_container_width=True)
    except Exception as e:
        st.warning(f"pairwise plot 생성 실패: {e}")

st.markdown("---")
st.info("다음 단계: Preprocessing 페이지로 이동하여 결측/인코딩/스케일링을 적용하세요.")
