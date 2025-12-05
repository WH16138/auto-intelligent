# -*- coding: utf-8 -*-
# pages/07_hpo.py
"""
7단계 - Hyperparameter Tuning (Optuna)

- 특징공학 반영본(df_features)만 사용
- 모델 셀렉션 결과(베이스라인)에서 선택하거나 지원 모델 목록 중 선택
- modules/hpo.py가 지원하는 모델: RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor, SVC, LogisticRegression
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide")
st.title("7 - Hyperparameter Tuning (HPO)")

# Session defaults
for key in [
    "df_original",
    "df_dropped",
    "df_preprocessed",
    "df_features",
    "df_features_train",
    "df_features_test",
    "df",
    "target_col",
    "baseline_models",
    "best_model_name",
    "hpo_result",
]:
    st.session_state.setdefault(key, None)

# Imports
try:
    from modules import hpo as hpo_mod
    HPO_MODELS = getattr(hpo_mod, "MODEL_CHOICES", ["RandomForestClassifier"])
except Exception:
    hpo_mod = None
    HPO_MODELS = ["RandomForestClassifier"]

try:
    from modules import model_search as ms
except Exception:
    import modules.model_search as ms  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore


def get_df() -> Optional[pd.DataFrame]:
    # HPO는 특징공학 반영본을 사용
    return st.session_state.get("df_features")


df = get_df()
problem_type_override = st.session_state.get("problem_type")
if df is None:
    st.error("특징공학 결과(df_features)가 없습니다. 05단계에서 확정 후 진행하세요.")
    st.stop()
train_idx = st.session_state.get("train_idx")
if isinstance(train_idx, list) and len(train_idx) > 0:
    df_train = df.iloc[train_idx].copy()
else:
    df_train = df

st.write(f"사용 데이터(훈련용): {df_train.shape[0]} 행 × {df_train.shape[1]} 열")
if isinstance(train_idx, list) and len(train_idx) > 0:
    st.caption(f"train 인덱스 {len(train_idx)}개 사용 (전체 {df.shape[0]} 중)")

# Target
target_session = st.session_state.get("target_col")
columns_list = df.columns.tolist()
target_override = st.selectbox(
    "타깃 컬럼",
    options=["(세션 값 사용)"] + columns_list,
    index=0 if target_session is None else (columns_list.index(target_session) + 1 if target_session in columns_list else 0),
)
target_col = target_session if target_override == "(세션 값 사용)" else target_override
if target_col is None or target_col not in df.columns:
    st.error("타깃 컬럼을 선택하세요. Overview에서 지정했거나 여기서 선택할 수 있습니다.")
    st.stop()

# Diagnostics
target_ser = df_train[target_col]
target_nulls = int(target_ser.isnull().sum())
unique_vals = target_ser.dropna().nunique()
numeric_features = [c for c in df_train.select_dtypes(include=["number"]).columns if c != target_col]
st.markdown(
    f"- 타깃 고유값: **{unique_vals}** | 타깃 결측: **{target_nulls}** | 수치형 특징 수: **{len(numeric_features)}**"
)
if unique_vals <= 1:
    st.error("타깃이 단일 값입니다. 최소 2개 이상의 클래스/값이 필요합니다.")
    st.stop()
if len(numeric_features) == 0:
    st.error("사용 가능한 수치형 특징이 없습니다.")
    st.stop()
if target_nulls > 0:
    st.warning("타깃 결측 행을 제거하고 진행합니다.")

# Clean data
if problem_type_override:
    st.info(f"문제 유형 설정: {problem_type_override} (Overview에서 지정)")
df_hpo = df_train.dropna(subset=[target_col]).copy()
if df_hpo.empty:
    st.error("타깃 결측 제거 후 데이터가 비어 있습니다.")
    st.stop()

# HPO availability
if hpo_mod is None:
    st.error("modules.hpo를 불러오지 못했습니다. optuna 설치를 확인하세요.")
    st.stop()

# 모델 후보
baseline_models = st.session_state.get("baseline_models") or {}
best_baseline = st.session_state.get("best_model_name")
baseline_names = list(baseline_models.keys())

def _choose_default_model_name() -> str:
    if best_baseline and best_baseline in HPO_MODELS:
        return best_baseline
    for name in baseline_names:
        if name in HPO_MODELS:
            return name
    return HPO_MODELS[0]

st.subheader("HPO 설정")
col1, col2, col3 = st.columns([1.4, 1, 1])
with col1:
    options_model = []
    default_model = _choose_default_model_name()
    for name in baseline_names:
        if name in HPO_MODELS and name not in options_model:
            options_model.append(name)
    for name in HPO_MODELS:
        if name not in options_model:
            options_model.append(name)
    try:
        default_index = options_model.index(default_model)
    except Exception:
        default_index = 0
    model_option = st.selectbox("튜닝할 모델", options=options_model, index=default_index)
with col2:
    time_budget = st.number_input("시간 제한 (초, 0=기본 60초)", min_value=0, value=60, step=10)
    n_jobs = st.number_input("교차검증 n_jobs", min_value=1, value=1, step=1)
with col3:
    cv = st.number_input("CV 분할 수", min_value=2, max_value=10, value=3, step=1)
    random_state = st.number_input("random_state", min_value=0, value=0, step=1)

# Early stopping & search controls
col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    n_trials = st.number_input("n_trials (0=시간 기반)", min_value=0, value=0, step=10)
with col_e2:
    patience = st.number_input("조기중단 patience(연속 미개선 trial 수)", min_value=0, value=0, step=1)
with col_e3:
    tolerance = st.number_input("개선 허용 오차(tolerance)", min_value=0.0, value=0.0, step=0.001, format="%.3f")
if patience > 0:
    st.caption(f"연속 {int(patience)}회 개선 없으면 탐색 중단 (최소 trial 확보 후). tolerance={tolerance:.3f}")

# Run HPO
st.markdown("---")
st.subheader("HPO 실행")

# 실시간 진행 모니터 영역
progress_bar = st.progress(0, text="HPO 대기 중...")
history_chart = st.empty()
metric_placeholder = st.empty()
trial_values = []

if st.button("HPO 실행"):
    try:
        import time

        X, y, task, feature_names = ms.get_X_y(df_hpo, target_col=target_col, forced_task=problem_type_override)
        tb = int(time_budget) if time_budget and time_budget > 0 else 60
        total_budget = tb if tb > 0 else 60

        def _on_trial(study, trial):
            val = trial.value
            if val is not None:
                trial_values.append(val)
                history_chart.line_chart(
                    pd.DataFrame(trial_values, columns=["score"]),
                    height=200,
                )
            elapsed = time.time() - start_ts
            ratio = min(1.0, elapsed / max(1, total_budget))
            best_val = study.best_value if study.best_value is not None else "N/A"
            val_fmt = f"{val:.4f}" if val is not None else "N/A"
            progress_bar.progress(ratio, text=f"진행 중... trial {trial.number} | best={best_val}")
            metric_placeholder.markdown(f"**Trial {trial.number} 완료** | 값: {val_fmt} | 경과: {elapsed:.1f}s")

        start_ts = time.time()
        hpo_result = hpo_mod.run_hpo(
            model_name=model_option,
            X=X,
            y=y,
            time_budget=tb,
            n_trials=int(n_trials) if n_trials and n_trials > 0 else None,
            cv=int(cv),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            task_override=problem_type_override,
            callbacks=[_on_trial],
            patience=int(patience) if patience and patience > 0 else None,
            tolerance=float(tolerance),
        )
        st.session_state["hpo_result"] = hpo_result
        st.session_state["hpo_model_name"] = model_option
        st.session_state["target_col"] = target_col
        progress_bar.progress(1.0, text="HPO 완료")
        st.success("HPO가 완료되었습니다.")

        # Train model on full train set with best_params and save to session
        best_params = hpo_result.get("best_params") if isinstance(hpo_result, dict) else None
        if best_params:
            try:
                if model_option == "RandomForestClassifier":
                    model = RandomForestClassifier(random_state=int(random_state), **best_params)
                elif model_option == "RandomForestRegressor":
                    model = RandomForestRegressor(random_state=int(random_state), **best_params)
                elif model_option == "GradientBoostingClassifier":
                    model = GradientBoostingClassifier(random_state=int(random_state), **best_params)
                elif model_option == "GradientBoostingRegressor":
                    model = GradientBoostingRegressor(random_state=int(random_state), **best_params)
                elif model_option == "SVC":
                    params = {**best_params}
                    params.setdefault("probability", True)
                    params.setdefault("random_state", int(random_state))
                    model = SVC(**params)
                elif model_option == "LogisticRegression":
                    params = {**best_params}
                    params.setdefault("max_iter", 500)
                    params.setdefault("random_state", int(random_state))
                    model = LogisticRegression(**params)
                else:
                    model = None
                if model is not None:
                    model.fit(X, y)
                    st.session_state["trained_model"] = model
                    st.session_state["feature_names"] = feature_names
                    st.session_state["problem_type"] = task
                    st.success("HPO 최적 파라미터로 모델을 학습하여 세션에 저장했습니다.")
            except Exception as e:
                st.warning(f"HPO 모델 학습/저장 실패: {e}")
    except Exception as e:
        st.error(f"HPO 실행 실패: {e}")

# Results
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
st.info("다음 단계: Validation 페이지에서 모델 검증을 진행하세요.")
