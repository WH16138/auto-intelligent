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
st.caption("가벼운 ML 데이터 이해와 추천 흐름을 안내하는 Streamlit 앱")

st.success(
    "이 화면은 프로젝트 소개용입니다. 좌측 사이드바의 단계별 페이지(01~09)를 선택해 전체 워크플로우를 체험하세요."
)

st.markdown("## 프로젝트 한눈에 보기")
st.write(
    """
Auto-Intelligent는 CSV 업로드부터 EDA, 전처리, 특징 생성, 모델 선택, 하이퍼파라미터 최적화(HPO), 검증, 리포트까지 이어지는 학습용 파이프라인을 제공합니다.
각 단계의 코드는 `pages/`와 `modules/` 디렉터리에 분리되어 있으며, 이 화면은 흐름을 안내하는 진입점입니다.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("대상", "입문자/학생", "빠른 실습")
with col2:
    st.metric("주요 단계", "9", "+EDA/HPO")
with col3:
    st.metric("형태", "Streamlit", "가벼운 UI")

st.markdown("## 주요 기능")
st.markdown(
    """
- CSV 업로드와 자동 데이터 요약
- 기본 EDA 시각화(분포, 결측, 상관)
- 전처리 파이프라인 구성 및 특징 엔지니어링
- 베이스라인 모델 비교 + Optuna 기반 HPO
- 검증(메트릭/중요도) 및 리포트 다운로드
    """
)

st.markdown("## 추천 워크플로우")
st.markdown(
    """
1. 좌측에서 01_upload를 열어 CSV를 업로드하거나 샘플 데이터를 선택합니다.
2. 02~03에서 컬럼 개요와 EDA를 확인합니다.
3. 04~05에서 전처리/특징 생성으로 입력 데이터를 준비합니다.
4. 06~07에서 모델 베이스라인을 비교하고 HPO로 튜닝합니다.
5. 08~09에서 검증 지표와 중요도를 확인하고 리포트를 다운로드합니다.
    """
)

st.markdown("## 페이지 안내")
st.markdown(
    """
- 01_upload: CSV 업로드 및 샘플 데이터 불러오기
- 02_overview: 컬럼 타입/통계 요약, 간단한 추천
- 03_eda: 기본 시각화
- 04_preprocessing: 전처리 파이프라인 구성
- 05_feature_engineering: 특징 생성 및 저장
- 06_model_selection: 베이스라인 모델 비교
- 07_hpo: Optuna 기반 HPO 실행
- 08_validation: 모델 검증 및 중요도 확인
- 09_report: 결과 리포트 다운로드
    """
)

st.markdown("## 코드 구조")
st.markdown(
    """
- `modules/`: ingestion, preprocessing, feature_engineering, model_search, hpo, explain, eda, io_utils, visualization 등 핵심 로직
- `pages/`: Streamlit 페이지별 UI 흐름(01~09)
- `project_overview.md`: 전체 아키텍처 개요
- `README.md`: 설치 및 실행 안내
    """
)

st.markdown("## 실행 가이드")
st.code("""python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
""", language="bash")

st.info("샘플 데이터나 실습 힌트는 각 페이지의 안내/버튼을 따라 진행하세요. 문제가 발생하면 터미널 로그 또는 페이지 알림을 확인하세요.")
