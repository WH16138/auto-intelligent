# -*- coding: utf-8 -*-
# app.py - Auto-Intelligent landing page
# Usage: streamlit run app.py
import streamlit as st

st.set_page_config(
    page_title="Auto-Intelligent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Auto-Intelligent")
st.caption("입문자도 따라올 수 있는 가벼운 ML 파이프라인 안내")

st.success("좌측 사이드바에서 01~09 단계를 순서대로 눌러보세요. 각 단계에 간단 힌트와 기본값을 준비했습니다.")

st.markdown("## 프로젝트 한눈에 보기")
st.write(
    """
Auto-Intelligent는 CSV 업로드 → EDA → 전처리 → 특징 생성 → 모델 선택 → HPO → 검증 → 리포트까지 이어지는 교육용 파이프라인입니다.
각 단계는 `pages/`에 UI가, `modules/`에 핵심 로직이 담겨 있으며, 이 화면은 흐름을 안내하는 출발점입니다.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("누구를 위해?", "입문자·학생", "설명/추천 포함")
with col2:
    st.metric("단계 수", "9 스텝", "EDA·HPO 포함")
with col3:
    st.metric("형태", "Streamlit", "간단 UI")

st.markdown("## 주요 기능")
st.markdown(
    """
- CSV 업로드 & 자동 요약, 기본 EDA 시각화(분포/결측/상관)
- 전처리 파이프라인 + 특징 엔지니어링(규칙 기반)
- 베이스라인 비교, Optuna/그리드 HPO, 클래스 불균형 시 샘플링 옵션
- 검증 지표/중요도/리포트 다운로드
    """
)

st.markdown("## 빠른 따라하기")
st.markdown(
    """
1. **01_upload**: CSV 업로드 또는 샘플 데이터 사용 → 자동 split.
2. **02~03**: 컬럼 요약과 EDA로 데이터 파악.
3. **04~05**: 전처리/특징 생성 → 기본값으로 실행 후 필요 시 수정.
4. **06~07**: 베이스라인 비교 → Optuna/HPO로 튜닝(추천값 제공).
5. **08~09**: 검증 지표 확인 → 리포트/모델 다운로드.
    """
)

st.markdown("## 페이지 안내")
st.markdown(
    """
- 01_upload: CSV 업로드, 샘플 로드, 기본 split
- 02_overview: 컬럼 타입/통계 요약, 분할 전략 추천
- 03_eda: 기본 시각화(분포, 결측, 상관)
- 04_preprocessing: 전처리 파이프라인 구성
- 05_feature_engineering: 특징 생성/선택
- 06_model_selection: 베이스라인 모델 비교
- 07_hpo: Optuna/그리드 HPO + 샘플링 옵션
- 08_validation: 검증 지표/중요도/설명
- 09_report: 결과 요약/다운로드
    """
)

with st.expander("코드 구조 살펴보기"):
    st.markdown(
        """
        - `modules/`: ingestion, preprocessing, feature_engineering, model_search, hpo, explain, eda, io_utils, visualization 등 핵심 로직
        - `pages/`: Streamlit 페이지별 UI(01~09)
        - `project_overview.md`: 전체 아키텍처 개요
        - `README.md`: 설치 및 실행 안내
        """
    )

st.markdown("##local 환경 실행 가이드")
st.code(
    """python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
""",
    language="bash",
)

with st.expander("자주 묻는 질문"):
    st.markdown(
        """
        - **사용할 데이터가 없어요.** 01_upload에서 샘플 버튼을 눌러 기본 데이터를 사용해 보세요.
        - **설정값이 많아 어려워요.** 각 단계는 안전한 기본 설정을 제공합니다. 먼저 실행 후, 필요 시 세부 조정하세요.
        - **긴 출력이 부담될 ** 각 페이지에서 미리보기/expander를 활용해 요약만 확인할 수 있습니다.
        """
    )

st.info("각 단계에서 설명/힌트를 적극 활용해 주세요. 페이지 알림과 경고를 따라가면 초보자도 전체 흐름을 이해할 수 있습니다.")
