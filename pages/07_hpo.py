# pages/07_hpo.py
"""
Step 7 - Hyperparameter Tuning (Optuna)

Notes:
- Explicitly excludes the target column from feature inputs.
- Chooses dataset deterministically (preprocessed vs raw) to avoid DataFrame truthiness errors.
- Drops rows with missing target before HPO.
"""
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("7 - Hyperparameter Tuning (HPO)")

# Session defaults
st.session_state.setdefault("df", None)
st.session_state.setdefault("df_preprocessed", None)
st.session_state.setdefault("target_col", None)
st.session_state.setdefault("baseline_models", None)
st.session_state.setdefault("hpo_result", None)

# Imports (fallbacks)
try:
    from modules import hpo as hpo_mod
except Exception:
    hpo_mod = None

try:
    from modules import model_search as ms
except Exception:
    import modules.model_search as ms  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore


def get_df(prefer_preprocessed: bool = True) -> Tuple[Optional[pd.DataFrame], bool]:
    """Return dataframe and flag indicating if preprocessed was used."""
    df_feat = st.session_state.get("df_features")
    if prefer_preprocessed and df_feat is not None:
        return df_feat, True
    df_pre = st.session_state.get("df_preprocessed")
    if prefer_preprocessed and df_pre is not None:
        return df_pre, True
    df_raw = st.session_state.get("df")
    if df_raw is not None:
        return df_raw, False
    if df_feat is not None:
        return df_feat, True
    if df_pre is not None:
        return df_pre, True
    return None, False


# Choose data
df_pre, _ = get_df(prefer_preprocessed=True)
df_raw, _ = get_df(prefer_preprocessed=False)
if df_pre is None and df_raw is None:
    st.warning("Upload / Preprocessing 페이지에서 데이터를 준비한 후 다시 시도하세요.")
    st.stop()

data_choice = st.radio(
    "HPO에 사용할 데이터",
    options=[
        "특징공학/전처리 반영 데이터(df_features/df_preprocessed)",
        "원본 데이터(df)",
    ],
    index=0 if df_pre is not None else 1,
)
if data_choice.startswith("특징공학") and df_pre is not None:
    df = df_pre
else:
    df = df_raw

# Target selection
target_session = st.session_state.get("target_col")
columns_list = df.columns.tolist()
target_override = st.selectbox(
    "타깃 컬럼 선택",
    options=["(세션 값 사용)"] + columns_list,
    index=0 if target_session is None else columns_list.index(target_session) + 1 if target_session in columns_list else 0,
)
target_col = target_session if target_override == "(세션 값 사용)" else target_override
if target_col is None or target_col not in df.columns:
    st.error("타깃 컬럼을 선택하세요. Overview에서 설정했거나 여기서 선택할 수 있습니다.")
    st.stop()

# Diagnostics
target_ser = df[target_col]
target_nulls = int(target_ser.isnull().sum())
unique_vals = target_ser.dropna().nunique()
numeric_features = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
st.markdown(
    f"- 타깃 고유값: **{unique_vals}** | 결측치: **{target_nulls}**\n"
    f"- 사용 가능한 수치형 특징 수: **{len(numeric_features)}**"
)
if target_nulls > 0:
    st.warning("타깃 결측이 있습니다. 결측 행을 제거하고 HPO를 진행합니다.")
if unique_vals <= 1:
    st.error("타깃이 단일 값입니다. 최소 2개 이상의 클래스/값이 필요합니다.")
    st.stop()
if len(numeric_features) == 0:
    st.error("사용 가능한 수치형 특징이 없습니다. 전처리/인코딩 단계를 확인하세요.")
    st.stop()

# Drop rows with missing target
df_hpo = df.dropna(subset=[target_col]).copy()
if df_hpo.empty:
    st.error("타깃 결측 제거 후 데이터가 비어 있습니다.")
    st.stop()

st.write(f"HPO 데이터셋 크기: {df_hpo.shape[0]} 행 × {df_hpo.shape[1]} 열")
st.write(f"타깃 컬럼: `{target_col}`")

# HPO availability
if hpo_mod is None:
    st.error("modules.hpo가 로드되지 않았습니다. requirements.txt의 optuna가 설치되어 있는지 확인하세요.")
    st.stop()

# Controls
st.subheader("HPO 설정")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    model_to_tune = st.selectbox("튜닝할 모델", options=["RandomForest"], index=0)
with col2:
    time_budget = st.number_input("시간 제한 (초, 0=기본 60초)", min_value=0, value=60, step=10)
    n_jobs = st.number_input("교차검증 n_jobs", min_value=1, value=1, step=1)
with col3:
    cv = st.number_input("CV 분할 수", min_value=2, max_value=10, value=3, step=1)
    random_state = st.number_input("random_state", min_value=0, value=0, step=1)

st.markdown("---")
st.subheader("HPO 실행")
if st.button("HPO 실행"):
    try:
        with st.spinner("HPO 실행 중..."):
            X, y, task, feature_names = ms.get_X_y(df_hpo, target_col=target_col)
            tb = int(time_budget) if time_budget and time_budget > 0 else 60
            hpo_result = hpo_mod.run_hpo(
                model_name=model_to_tune,
                X=X,
                y=y,
                time_budget=tb,
                cv=int(cv),
                random_state=int(random_state),
                n_jobs=int(n_jobs),
            )
            st.session_state["hpo_result"] = hpo_result
            st.session_state["target_col"] = target_col  # persist
            st.success("HPO가 완료되었습니다.")
    except Exception as e:
        st.error(f"HPO 실행 실패: {e}")

st.markdown("---")
st.subheader("HPO 결과")
hpo_res = st.session_state.get("hpo_result")
if hpo_res is None:
    st.info("아직 HPO 결과가 없습니다. 먼저 실행하세요.")
else:
    st.write("best_params:")
    st.json(hpo_res.get("best_params"))
    st.write("best_value:", hpo_res.get("best_value"))
    if hpo_res.get("study_summary"):
        st.markdown("### Study Summary")
        st.json(hpo_res.get("study_summary"))
    if isinstance(hpo_res.get("trials"), list):
        st.markdown("### Trials (상위 20개)")
        st.dataframe(pd.DataFrame(hpo_res.get("trials")).head(20))

    st.markdown("---")
    st.subheader("결과 저장 / 스크립트 생성")
    save_name = st.text_input("best_params 저장 경로", value="artifacts/hpo_best_params.json")
    generate_script = st.checkbox("훈련 스크립트 생성", value=True)
    script_name = st.text_input("스크립트 경로", value="scripts/train_best_hpo.py")

    if st.button("best_params 저장"):
        try:
            best_params = hpo_res.get("best_params") or {}
            io_utils.save_json(best_params, save_name)
            st.success(f"저장 완료: {save_name}")
        except Exception as e:
            st.error(f"저장 실패: {e}")

    if generate_script and st.button("훈련 스크립트 생성"):
        try:
            best_params = hpo_res.get("best_params") or {}
            preproc_obj = st.session_state.get("preprocessing_pipeline")
            snapshot_paths = io_utils.snapshot_artifacts(
                model_obj=None,
                preprocessor_obj=preproc_obj,
                params=best_params,
                base_dir="artifacts",
                prefix="hpo",
            )
            model_filename = snapshot_paths.get("model_path", "artifacts/hpo_best_model.pkl")
            pipeline_filename = snapshot_paths.get("pipeline_path")
            io_utils.generate_train_script(
                params=best_params,
                model_filename=model_filename,
                pipeline_filename=pipeline_filename,
                script_path=script_name,
                target_col=target_col,
            )
            st.success(f"스크립트 생성 완료: {script_name}")
            try:
                with open(script_name, "r", encoding="utf-8") as f:
                    st.code(f.read()[:800] + "\n...\n", language="python")
            except Exception:
                pass
        except Exception as e:
            st.error(f"스크립트 생성 실패: {e}")

st.markdown("---")
st.write("다음 단계: Validation 페이지에서 검증을 진행하세요.")
