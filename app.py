# app.py - AutoML-Edu main entry (enhanced)
# Usage: streamlit run app.py
import streamlit as st
import pandas as pd
import traceback
import time
from typing import Any, Dict

st.set_page_config(page_title="AutoML-Edu", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Session state initialization
# -------------------------
def init_session_state():
    defaults = {
        "df": None,
        "df_preprocessed": None,
        "df_features": None,
        "fingerprint": None,
        "preprocessing_pipeline": None,
        "models": {},            # store models / baseline results / best_model meta
        "hpo_result": None,
        "current_step": 1,
        "tutorial_mode": False,
        "time_budget_seconds": 120,
        "cpu_limit": None,
        "last_message": None,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# -------------------------
# Helper utilities
# -------------------------
def set_message(msg: str, level: str = "info"):
    """Store message in session and also display."""
    st.session_state["last_message"] = {"ts": time.time(), "msg": msg, "level": level}
    if level == "info":
        st.info(msg)
    elif level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.write(msg)

def safe_execute(func, *args, error_message="오류가 발생했습니다", **kwargs):
    """Run func and capture exceptions; set session_state['error'] on failure."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        tb = traceback.format_exc()
        st.session_state["error"] = {"message": str(e), "traceback": tb}
        st.error(f"{error_message}: {e}")
        # also show expandable trace for debugging
        with st.expander("오류 상세(traceback)"):
            st.code(tb)
        return None

def upload_sample_dataset(name: str = "breast_cancer"):
    """Provide small sample datasets to test flows."""
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = pd.concat([data.frame, pd.Series(data.target, name="target")], axis=1) if hasattr(data, "frame") else pd.DataFrame(data.data, columns=data.feature_names).assign(target=data.target)
        return df
    elif name == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame
        df["target"] = d.target
        return df
    else:
        return None

# -------------------------
# Sidebar: global controls & navigation
# -------------------------
with st.sidebar:
    st.title("AutoML-Edu")
    st.markdown("CSV 업로드 → EDA → 전처리 → 특징공학 → 모델선택 → HPO → 검증 → 리포트")
    st.write("---")

    # Navigation steps
    steps = [
        "1. 업로드", "2. 데이터 개요", "3. 시각화(EDA)", "4. 전처리",
        "5. 특징공학", "6. 모델선택", "7. HPO", "8. 검증", "9. 리포트"
    ]
    current = st.radio("단계 선택", options=list(range(1, len(steps) + 1)), index=st.session_state["current_step"] - 1)
    st.session_state["current_step"] = current

    st.write("---")
    st.checkbox("튜토리얼 모드 (학습 설명 표출)", value=st.session_state["tutorial_mode"], key="tutorial_mode")
    st.write("리소스 제약")
    st.session_state["time_budget_seconds"] = st.slider("HPO 시간(초)", min_value=30, max_value=3600, value=st.session_state["time_budget_seconds"], step=30)
    st.session_state["cpu_limit"] = st.number_input("CPU 제한(없으면 0)", min_value=0, value=st.session_state["cpu_limit"] or 0, step=1)

    st.write("---")
    st.markdown("### 빠른 시작")
    if st.button("샘플 데이터 로드 (breast_cancer)"):
        df_sample = upload_sample_dataset("breast_cancer")
        st.session_state["df"] = df_sample
        set_message("샘플 데이터가 로드되었습니다.", "success")
    if st.button("샘플 데이터 로드 (iris)"):
        df_sample = upload_sample_dataset("iris")
        st.session_state["df"] = df_sample
        set_message("샘플 데이터가 로드되었습니다.", "success")

    st.write("---")
    if st.button("앱 재시작(세션 초기화)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# -------------------------
# Main content: step router
# -------------------------
st.title("AutoML-Edu — 단계별 학습형 AutoML 데모")

# show last message if exists
if st.session_state.get("last_message"):
    lm = st.session_state["last_message"]
    st.info(f"최근: {lm['msg']}")

step = st.session_state["current_step"]

# Brief educational banner at top of each step
def edu_banner(step:int):
    banners = {
        1: ("업로드 단계", "데이터 파일을 업로드하고 기본 스키마와 품질(결측/타깃)을 확인합니다."),
        2: ("데이터 개요", "컬럼 타입, 유일값, 결측률 등 데이터 지문을 생성하여 다음 단계 권장 전략을 만듭니다."),
        3: ("EDA", "분포와 상관관계를 시각화하여 이상치/불균형을 발견합니다."),
        4: ("전처리", "결측/인코딩/스케일링을 자동 혹은 수동으로 적용해 모델 준비를 합니다."),
        5: ("특징공학", "유의미한 파생 피처를 만들고 선택합니다."),
        6: ("모델선택", "간단한 베이스라인 비교로 후보 모델을 선별합니다."),
        7: ("HPO", "리소스 제약 내에서 하이퍼파라미터를 탐색합니다."),
        8: ("검증", "교차검증, 학습곡선, 혼동행렬 등으로 일반화 성능을 점검합니다."),
        9: ("리포트", "모든 설정과 결과를 포함하는 재현가능한 리포트를 생성합니다."),
    }
    title, desc = banners.get(step, ("", ""))
    st.subheader(f"{step}. {title}")
    st.caption(desc)
    if st.session_state["tutorial_mode"]:
        st.markdown(f"**학습 포인트:** {desc}")

edu_banner(step)

# Step implementations: light-weight delegations to pages under /pages if present
# If the pages are present Streamlit will also show them in the UI; here we include
# basic inline fallbacks so the main app can run alone for quick demos.

if step == 1:
    st.header("1. 업로드 (데이터 수집)")
    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"], key="uploader_main")
    if uploaded:
        df = safe_execute(pd.read_csv, uploaded, error_message="CSV 읽기 실패")
        if df is not None:
            st.session_state["df"] = df
            set_message(f"데이터 업로드 완료 ({df.shape[0]}행 × {df.shape[1]}열)", "success")
            st.markdown("미리보기 (상위 100행)")
            st.dataframe(df.head(100))
            if st.button("데이터 지문 생성"):
                from pipeline.ingestion import generate_fingerprint
                fp = safe_execute(generate_fingerprint, df, error_message="지문 생성 실패")
                if fp:
                    st.session_state["fingerprint"] = fp
                    st.json(fp)
    else:
        st.info("CSV 파일을 업로드하거나 좌측의 샘플 데이터 버튼으로 시작하세요.")

elif step == 2:
    st.header("2. 데이터 개요")
    df = st.session_state.get("df")
    if df is None:
        st.warning("먼저 1) 업로드에서 데이터를 올려주세요.")
    else:
        st.write("컬럼별 정보")
        col_info = {}
        for c in df.columns:
            col_info[c] = {"dtype": str(df[c].dtype), "n_unique": int(df[c].nunique()), "n_null": int(df[c].isnull().sum())}
        st.json(col_info)
        if st.button("데이터 유형 추천(간이)"):
            # simple heuristic recommendations
            fp = st.session_state.get("fingerprint") or {}
            recs = {"recommended_model_family": "classification" if "target" in df.columns else "try to specify target column"}
            if fp:
                recs["note"] = "fingerprint 사용됨"
            st.json(recs)

elif step == 3:
    st.header("3. EDA (시각화)")
    df = st.session_state.get("df")
    if df is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        col = st.selectbox("시각화할 컬럼 선택(숫자형)", options=numeric if numeric else ["(없음)"])
        if col and col != "(없음)":
            import plotly.express as px
            fig = px.histogram(df, x=col, nbins=50, title=f"{col} 분포")
            st.plotly_chart(fig, use_container_width=True)
        if categorical:
            cat = st.selectbox("범주형 컬럼(빈도)", options=categorical, key="cat_col")
            if cat:
                import plotly.express as px
                fig = px.bar(df[cat].value_counts().reset_index().rename(columns={cat:"count","index":cat}), x=cat, y="count", title=f"{cat} 빈도")
                st.plotly_chart(fig, use_container_width=True)

elif step == 4:
    st.header("4. 전처리")
    df = st.session_state.get("df")
    if df is None:
        st.warning("데이터를 먼저 업로드하세요.")
    else:
        st.write("간단 전처리 샘플(숫자형 중앙값 보간 + 표준화)")
        if st.button("간이 전처리 적용"):
            from pipeline.preprocessing import simple_preprocess
            df2, pre = safe_execute(simple_preprocess, df, error_message="전처리 실패")
            if df2 is not None:
                st.session_state["df_preprocessed"] = df2
                st.session_state["preprocessing_pipeline"] = pre
                set_message("전처리 완료: df_preprocessed에 저장됨", "success")
                st.dataframe(df2.head(50))

elif step == 5:
    st.header("5. 특징공학")
    df = st.session_state.get("df_preprocessed") or st.session_state.get("df")
    if df is None:
        st.warning("데이터를 먼저 업로드/전처리하세요.")
    else:
        st.write("자동 피처 생성 (간이 예시)")
        if st.button("자동 피처 생성"):
            from pipeline.feature_engineering import generate_sample_features
            df_feat = safe_execute(generate_sample_features, df, error_message="특징공학 실패")
            if df_feat is not None:
                st.session_state["df_features"] = df_feat
                st.dataframe(df_feat.head(50))

elif step == 6:
    st.header("6. 모델선택 (간이 베이스라인)")
    df = st.session_state.get("df_features") or st.session_state.get("df_preprocessed") or st.session_state.get("df")
    if df is None:
        st.warning("데이터를 먼저 준비하세요.")
    else:
        st.write("간단한 베이스라인 모델들을 빠르게 비교합니다.")
        if st.button("간이 베이스라인 실행"):
            from pipeline.model_search import quick_baselines
            res = safe_execute(quick_baselines, df, error_message="베이스라인 실패")
            if res is not None:
                st.session_state["models"]["baselines"] = res
                st.table(res)

elif step == 7:
    st.header("7. HPO (하이퍼파라미터 탐색)")
    df = st.session_state.get("df_features") or st.session_state.get("df_preprocessed") or st.session_state.get("df")
    if df is None:
        st.warning("데이터를 먼저 준비하세요.")
    else:
        st.write(f"HPO 시간 예산: {st.session_state['time_budget_seconds']}초")
        if st.button("HPO 시작 (샘플)"):
            from pipeline.hpo import run_optuna_sample
            hpo_res = safe_execute(run_optuna_sample, df, st.session_state["time_budget_seconds"], error_message="HPO 실패")
            if hpo_res is not None:
                st.session_state["hpo_result"] = hpo_res
                st.json(hpo_res)

elif step == 8:
    st.header("8. 검증")
    st.write("최종 모델 검증 및 시각화 (샘플)")
    if st.session_state["models"].get("baselines") is not None:
        st.write("저장된 베이스라인 결과:")
        st.table(st.session_state["models"]["baselines"])
    else:
        st.info("6번에서 베이스라인을 실행해보세요.")

elif step == 9:
    st.header("9. 리포트 및 내보내기")
    st.write("모든 단계의 요약 리포트를 생성하고 모델/스크립트를 다운로드할 수 있습니다.")
    if st.button("리포트(샘플) 생성"):
        # Simple example: assemble small html report
        report_html = "<html><body><h1>AutoML-Edu Report (샘플)</h1>"
        if st.session_state.get("fingerprint"):
            report_html += f"<h2>Fingerprint</h2><pre>{st.session_state['fingerprint']}</pre>"
        if st.session_state.get("models").get("baselines") is not None:
            report_html += "<h2>Baselines</h2>" + st.session_state["models"]["baselines"].to_html()
        report_html += "</body></html>"
        st.download_button("리포트 다운로드 (HTML)", data=report_html, file_name="automl_edu_report.html", mime="text/html")
        set_message("리포트가 생성되어 다운로드 가능합니다.", "success")

# -------------------------
# Footer / Debug info
# -------------------------
st.write("---")
col1, col2 = st.columns([3,1])
with col1:
    st.caption("AutoML-Edu — 교육용 및 데모용 템플릿. 프로덕션 사용 전 적절한 검증과 보안/프라이버시 검토 필요.")
with col2:
    if st.session_state.get("error"):
        st.error("최근 에러가 있습니다. '오류 상세(traceback)'를 확인하세요.")
    else:
        st.success("상태 정상")

# End of app.py
