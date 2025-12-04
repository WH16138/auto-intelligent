# pages/07_hpo.py
"""
07 - Hyperparameter Tuning (Optuna 기반 HPO) 페이지

기능:
- 튜닝에 사용할 데이터 / 타깃 선택
- 모델 선택(현재 기본: RandomForest) — 확장 가능
- 시간 예산(timeout) 또는 trial 수로 HPO 실행
- 실행 중 상태(스피너) 및 결과 표시 (best_params, best_value, study_summary)
- 결과 저장(JSON) 및 재현 스크립트 자동 생성 (modules.io_utils.generate_train_script)
"""
import streamlit as st
import pandas as pd
import json
import time
from typing import Optional, Dict, Any

st.set_page_config(layout="wide")
st.title("7 - Hyperparameter Tuning (HPO)")

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_preprocessed" not in st.session_state:
    st.session_state["df_preprocessed"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None
if "baseline_models" not in st.session_state:
    st.session_state["baseline_models"] = None
if "hpo_result" not in st.session_state:
    st.session_state["hpo_result"] = None

# imports (optional)
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

# check available data
df = st.session_state.get("df_preprocessed") or st.session_state.get("df")
if df is None:
    st.warning("먼저 Upload / Preprocessing 단계를 완료하세요.")
    st.stop()

target_col = st.session_state.get("target_col")
st.write(f"현재 세션 데이터: {df.shape[0]}행 × {df.shape[1]}열")
st.write(f"설정된 타깃: `{target_col}`" if target_col else "타깃 컬럼이 설정되어 있지 않습니다. Overview에서 설정하세요.")

# HPO availability
if hpo_mod is None:
    st.error("modules.hpo가 설치/로딩되어 있지 않습니다. `modules/hpo.py`가 있어야 HPO를 실행할 수 있습니다.")
    st.info("현재 구현은 Optuna 기반. requirements.txt에 optuna를 추가하세요.")
    st.stop()

# Page controls
st.subheader("HPO 설정")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    model_to_tune = st.selectbox("튜닝할 모델 선택 (현재 지원 제한적)", options=["RandomForest"], index=0)
with col2:
    time_budget = st.number_input("시간 예산 (초) — timeout (0이면 사용 안함)", min_value=0, value=60, step=10)
    n_jobs = st.number_input("교차검증 n_jobs (병렬)", min_value=1, value=1, step=1)
with col3:
    cv = st.number_input("CV 분할 수 (n_splits)", min_value=2, max_value=10, value=3, step=1)
    random_state = st.number_input("랜덤 시드", min_value=0, value=0, step=1)

st.markdown("---")
st.write("HPO 입력 데이터 설정:")
hpo_data_choice = st.radio("HPO에 사용할 데이터", options=["세션의 전처리된 데이터(df_preprocessed) 사용", "원본 세션 데이터(df) 사용"], index=0)
if hpo_data_choice.startswith("세션의") and st.session_state.get("df_preprocessed") is not None:
    df_hpo = st.session_state.get("df_preprocessed")
else:
    df_hpo = st.session_state.get("df")

# allow explicit override of target for this run
target_override = st.selectbox("이 HPO 실행에서 사용할 타깃 컬럼 (오버라이드)", options=["(세션값 사용)"] + list(df_hpo.columns), index=0)
if target_override == "(세션값 사용)":
    page_target = target_col
else:
    page_target = target_override

if page_target is None:
    st.warning("타깃이 지정되어 있지 않습니다. HPO를 수행하려면 타깃을 지정해야 합니다.")
    st.stop()

# show small class/target info
try:
    tc_ser = df_hpo[page_target]
    st.write("타깃 샘플 분포:")
    st.write(tc_ser.value_counts(dropna=False).head(10))
except Exception:
    pass

st.markdown("---")
st.subheader("HPO 실행")
col_run_a, col_run_b = st.columns([3,1])
with col_run_a:
    run_mode = st.radio("실행 모드", options=["시간 기반(timeout)","trial 기반(n_trials)"], index=0)
    if run_mode == "trial 기반(n_trials)":
        n_trials = st.number_input("n_trials", min_value=1, value=50, step=1)
    else:
        n_trials = None

with col_run_b:
    save_name = st.text_input("HPO 결과 저장 파일명 (params JSON)", value="artifacts/hpo_best_params.json")
    generate_script = st.checkbox("최적 파라미터로 재현 train script 생성", value=True)
    script_name = st.text_input("생성될 스크립트 경로", value="scripts/train_best_hpo.py")

# run button
if st.button("HPO 실행"):
    try:
        with st.spinner("HPO를 실행합니다 — 진행 중 (콘솔/로그를 확인하세요)..."):
            # prepare X,y via model_search helper
            X, y, task, feature_names = ms.get_X_y(df_hpo, target_col=page_target)
            if X is None or y is None:
                st.error("HPO를 수행할 수 있는 X,y 데이터를 준비하지 못했습니다. 전처리 상태를 확인하세요.")
            else:
                # call run_hpo — respect run_mode
                if run_mode == "trial 기반(n_trials)" and n_trials is not None:
                    # run_hpo in our hpo_mod expects time_budget, so we call with timeout approx via time_budget or implement trials? fallback: use time_budget with approx time per trial
                    # hpo_mod.run_hpo supports time_budget param. To support n_trials, we'll run with time_budget ~ n_trials*2s heuristic if time_budget==0
                    tb = int(time_budget) if time_budget and time_budget > 0 else max(30, int(n_trials) * 3)
                else:
                    tb = int(time_budget) if time_budget and time_budget > 0 else 60

                # call hpo_mod.run_hpo
                hpo_result = hpo_mod.run_hpo(model_name=model_to_tune, X=X, y=y, time_budget=tb, cv=int(cv), random_state=int(random_state), n_jobs=int(n_jobs))
                st.session_state["hpo_result"] = hpo_result
                st.success("HPO 실행 완료")
    except Exception as e:
        st.error(f"HPO 실행 중 오류가 발생했습니다: {e}")

st.markdown("---")
st.subheader("HPO 결과")
hpo_res = st.session_state.get("hpo_result")
if hpo_res is None:
    st.info("아직 HPO 실행 결과가 없습니다. 위에서 실행하세요.")
else:
    # pretty display
    try:
        st.write("최고 파라미터 (best_params):")
        st.json(hpo_res.get("best_params"))
    except Exception:
        st.write(hpo_res.get("best_params"))

    try:
        st.write("최고 값 (best_value):", hpo_res.get("best_value"))
    except Exception:
        pass

    if hpo_res.get("study_summary"):
        st.markdown("### Study Summary (간단)")
        try:
            st.json(hpo_res.get("study_summary"))
        except Exception:
            st.write(hpo_res.get("study_summary"))

    if hpo_res.get("trials") is not None and isinstance(hpo_res.get("trials"), list):
        st.markdown("### Trials (최대 일부 표시)")
        st.dataframe(pd.DataFrame(hpo_res.get("trials")).head(20))

    # actions: save params, snapshot, generate train script
    st.markdown("---")
    st.subheader("결과 저장 및 재현")
    if st.button("최적 파라미터 JSON으로 저장"):
        try:
            best_params = hpo_res.get("best_params") or {}
            io_utils.save_json(best_params, save_name)
            st.success(f"최적 파라미터를 저장했습니다: {save_name}")
            # offer download
            with open(save_name, "rb") as f:
                b = f.read()
            st.download_button("파라미터 다운로드", data=b, file_name=save_name.split("/")[-1], mime="application/json")
        except Exception as e:
            st.error(f"파라미터 저장 실패: {e}")

    if generate_script:
        if st.button("재현 스크립트 생성 (generate_train_script 호출)"):
            try:
                best_params = hpo_res.get("best_params") or {}
                # attempt to find pipeline path from session artifacts if present
                preproc_path = None
                # if preprocessing pipeline stored in session, try to snapshot first to obtain pipeline path
                preproc_obj = st.session_state.get("preprocessing_pipeline")
                # snapshot minimal artifacts to get pipeline path
                snapshot_paths = io_utils.snapshot_artifacts(model_obj=None, preprocessor_obj=preproc_obj, params=best_params, base_dir="artifacts", prefix="hpo")
                # choose model filename placeholder
                model_filename = snapshot_paths.get("model_path", "artifacts/hpo_best_model.pkl")
                pipeline_filename = snapshot_paths.get("pipeline_path")
                io_utils.generate_train_script(params=best_params, model_filename=model_filename, pipeline_filename=pipeline_filename, script_path=script_name, target_col=page_target)
                st.success(f"재현 스크립트를 생성했습니다: {script_name}")
                try:
                    with open(script_name, "r", encoding="utf-8") as f:
                        st.code(f.read()[:1000] + "\n...\n", language="python")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"재현 스크립트 생성 실패: {e}")

st.markdown("---")
st.write("다음 단계: Evaluation 페이지에서 최종 모델 성능을 평가하거나, Model Selection으로 돌아가 모델을 채택하세요.")
