# -*- coding: utf-8 -*-
# pages/07_hpo.py
"""
7단계 - Hyperparameter Tuning

- 특징공학 반영본(df_features)만 사용
- 모델 셀렉션 결과(베이스라인)에서 선택하거나 지원 모델 목록 중 선택
- Optuna 또는 GridSearchCV를 사용하며, 불균형 분류 시 샘플링 옵션을 제공합니다.
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    _HAS_IMB = True
except Exception:
    ImbPipeline, SMOTE, RandomUnderSampler = None, None, None
    _HAS_IMB = False

st.set_page_config(layout="wide")
st.title("7 - Hyperparameter Tuning (HPO)")
st.info("train 기준 데이터를 사용해 Optuna 또는 GridSearchCV로 튜닝합니다. 기본값으로 바로 실행한 뒤, 필요하면 고급 옵션(샘플링·조기중단)을 펼쳐서 조정하세요.")

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


def _safe_cv(df_in: pd.DataFrame, target: str, requested_cv: int, task: str) -> Tuple[Optional[int], Optional[str]]:
    """작은 데이터나 불균형에서 stratified CV가 실패하는 것을 방지하기 위해 cv 값을 조정합니다."""
    n_samples = len(df_in)
    if n_samples < 2:
        return None, "샘플 수가 2개 미만입니다."
    cv_adj = max(2, min(requested_cv, n_samples))
    if task == "classification":
        counts = df_in[target].value_counts(dropna=True)
        if counts.empty:
            return None, "타깃 값이 비어 있습니다."
        min_count = int(counts.min())
        if min_count < 2:
            return None, "한 클래스에 샘플이 1개뿐이라 교차검증을 할 수 없습니다."
        cv_adj = min(cv_adj, min_count)
    return cv_adj, None


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

col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.metric("train 행 수", df_train.shape[0])
with col_meta2:
    st.metric("전체 열 수", df_train.shape[1])
with col_meta3:
    if isinstance(train_idx, list) and len(train_idx) > 0:
        st.caption(f"train 인덱스 {len(train_idx)}개 사용 (전체 {df.shape[0]} 중)")
    else:
        st.caption("전체 데이터를 train으로 사용합니다.")

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
imbalance_ratio = None
if unique_vals > 1:
    try:
        counts = target_ser.value_counts(dropna=True)
        if not counts.empty:
            imbalance_ratio = counts.max() / max(1, counts.sum())
    except Exception:
        imbalance_ratio = None
st.markdown(
    f"- 타깃 고유값: **{unique_vals}** | 타깃 결측: **{target_nulls}** | 수치형 특징 수: **{len(numeric_features)}**"
)
if imbalance_ratio is not None:
    st.caption(f"클래스 불균형 비율(최대/전체): {imbalance_ratio:.3f}")
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
st.caption("추천: Optuna + 기본값. GridSearchCV는 시간 예산/조기중단을 지원하지 않으며, coarse→fine 2단계 그리드를 사용합니다.")
col1, col2, col3 = st.columns([1.4, 1, 1])
with col1:
    search_method = st.selectbox("탐색 방법", options=["Optuna", "GridSearchCV"], index=0)
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
    if search_method == "Optuna":
        time_budget = st.number_input("시간 제한 (초, 0=기본 60초)", min_value=0, value=60, step=10, help="0이면 60초로 동작")
    else:
        time_budget = 0
        st.caption("GridSearchCV는 시간 예산/조기중단을 사용하지 않습니다.")
    n_jobs = st.number_input("교차검증 n_jobs", min_value=1, value=1, step=1, help="병렬 처리 개수")
with col3:
    cv = st.number_input("CV 분할 수", min_value=2, max_value=10, value=3, step=1, help="교차검증 폴드")
    random_state = st.number_input("random_state", min_value=0, value=0, step=1, help="재현성 설정")

# Sampling options (classification 전용)
with st.expander("불균형 대응/고급 옵션", expanded=False):
    sampling_options = ["없음 (기본)"]
    if _HAS_IMB:
        sampling_options.extend(["SMOTE (오버샘플링)", "언더샘플링"])
    if imbalance_ratio and imbalance_ratio >= 0.7:
        st.warning("클래스 불균형이 높습니다. 샘플링 기법을 고려하세요.")
    sampling_method = st.selectbox("샘플링 기법", options=sampling_options, index=0, help="교차검증에서 train fold에만 적용됩니다.")
    def _make_sampler():
        if not _HAS_IMB:
            return None
        if sampling_method.startswith("SMOTE"):
            return SMOTE(random_state=int(random_state))
        if sampling_method.startswith("언더샘플링"):
            return RandomUnderSampler(random_state=int(random_state))
        return None

    # Early stopping & search controls (Optuna 전용)
    if search_method == "Optuna":
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            n_trials = st.number_input("n_trials (0=시간 기반)", min_value=0, value=0, step=10)
        with col_e2:
            patience = st.number_input("조기중단 patience(연속 미개선 trial 수)", min_value=0, value=0, step=1)
        with col_e3:
            tolerance = st.number_input("개선 허용 오차(tolerance)", min_value=0.0, value=0.0, step=0.001, format="%.3f")
        if patience > 0:
            st.caption(f"연속 {int(patience)}회 개선 없으면 탐색 중단 (최소 trial 확보 후). tolerance={tolerance:.3f}")
    else:
        n_trials, patience, tolerance = None, None, 0.0

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
        cv_adj, cv_warn = _safe_cv(df_hpo, target_col, int(cv), task)
        if cv_adj is None:
            st.error(cv_warn or "교차검증 설정이 유효하지 않습니다.")
            st.stop()
        if cv_warn:
            st.warning(cv_warn)
        if cv_adj != cv:
            st.info(f"CV를 {cv}→{cv_adj}로 조정했습니다.")
        if search_method == "Optuna":
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
                cv=int(cv_adj),
                random_state=int(random_state),
                n_jobs=int(n_jobs),
                task_override=problem_type_override,
                callbacks=[_on_trial],
                patience=int(patience) if patience and patience > 0 else None,
                tolerance=float(tolerance),
                sampler=_make_sampler() if task == "classification" else None,
            )
        else:
            # GridSearchCV 경로
            if task == "classification":
                scoring = "accuracy"
                base_model = {
                    "RandomForestClassifier": RandomForestClassifier(random_state=int(random_state)),
                    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=int(random_state)),
                    "SVC": SVC(probability=True, random_state=int(random_state)),
                    "LogisticRegression": LogisticRegression(max_iter=500, random_state=int(random_state)),
                    "LogisticRegressionCV": LogisticRegression(max_iter=500, random_state=int(random_state)),
                }.get(model_option)
            else:
                scoring = "r2"
                base_model = {
                    "RandomForestRegressor": RandomForestRegressor(random_state=int(random_state)),
                    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=int(random_state)),
                    # 회귀용 SVC는 구현하지 않음
                }.get(model_option)

            if base_model is None:
                st.error("선택한 모델은 GridSearchCV 설정이 없습니다.")
                st.stop()

            # 1단계(Coarse) 그리드
            param_grids_coarse = {
                "RandomForestClassifier": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 8, 16, 24],
                    "max_features": ["sqrt", "log2"],
                },
                "RandomForestRegressor": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 8, 16, 24],
                    "max_features": ["sqrt", "log2"],
                },
                "GradientBoostingClassifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [2, 3],
                    "subsample": [0.7, 1.0],
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [2, 3],
                    "subsample": [0.7, 1.0],
                },
                "SVC": {
                    "C": [0.1, 1, 10],
                    "gamma": [0.01, 0.1, 1],
                    "kernel": ["rbf"],
                },
                "LogisticRegression": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                },
                "LogisticRegressionCV": {
                    "Cs": [1, 5, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                },
            }
            grid = param_grids_coarse.get(model_option)
            progress_bar.progress(0.2, text="GridSearchCV 1단계(Coarse) 실행 중...")
            sampler_obj = _make_sampler() if task == "classification" else None
            estimator_to_use = base_model
            if sampler_obj is not None and _HAS_IMB:
                estimator_to_use = ImbPipeline([("sampler", sampler_obj), ("model", base_model)])
            gs = GridSearchCV(
                estimator=estimator_to_use,
                param_grid=grid,
                scoring=scoring,
                cv=int(cv_adj),
                n_jobs=int(n_jobs),
                verbose=0,
            )
            try:
                gs.fit(X, y)
            except Exception as e:
                if int(n_jobs) > 1:
                    st.warning(f"병렬 GridSearchCV 실패로 n_jobs=1로 재시도합니다. 원인: {e}")
                    gs = GridSearchCV(
                        estimator=estimator_to_use,
                        param_grid=grid,
                        scoring=scoring,
                        cv=int(cv_adj),
                        n_jobs=1,
                        verbose=0,
                    )
                    gs.fit(X, y)
                else:
                    raise
            best_params = gs.best_params_
            best_score = float(gs.best_score_) if gs.best_score_ is not None else None
            total_trials = len(gs.cv_results_["params"])
            chart_scores = list(gs.cv_results_["mean_test_score"])

            # 2단계(Fine) 그리드: 1단계 최고치 주변에서 좁혀 탐색
            def _refine_grid(model_name: str, params: Dict[str, Any]) -> Optional[Dict[str, list]]:
                if params is None:
                    return None
                def around(val, factors=(0.5, 1.0, 2.0)):
                    vals = set()
                    for f in factors:
                        try:
                            vals.add(type(val)(val * f))
                        except Exception:
                            pass
                    return sorted(vals)

                if model_name.startswith("RandomForest"):
                    ne = params.get("n_estimators", 100)
                    md = params.get("max_depth", None)
                    grid_ref = {
                        "n_estimators": sorted(set([ne, max(50, int(ne * 0.8)), int(ne * 1.5)])),
                        "max_features": [params.get("max_features", "sqrt")],
                    }
                    if md is None:
                        grid_ref["max_depth"] = [None, 12, 20]
                    else:
                        grid_ref["max_depth"] = sorted(set([md, max(2, int(md * 0.7)), int(md * 1.3)]))
                    return grid_ref
                if model_name.startswith("GradientBoosting"):
                    lr = params.get("learning_rate", 0.1)
                    ne = params.get("n_estimators", 100)
                    md = params.get("max_depth", 3)
                    return {
                        "learning_rate": around(lr, factors=(0.5, 1.0, 1.5)),
                        "n_estimators": sorted(set([ne, max(50, int(ne * 0.8)), int(ne * 1.5)])),
                        "max_depth": sorted(set([md, max(2, int(md * 0.8)), md + 1])),
                        "subsample": [params.get("subsample", 1.0)],
                    }
                if model_name == "SVC":
                    c = params.get("C", 1.0)
                    g = params.get("gamma", 0.1)
                    return {
                        "C": around(c, factors=(0.5, 1.0, 2.0)),
                        "gamma": around(g, factors=(0.5, 1.0, 2.0)),
                        "kernel": ["rbf"],
                    }
                if model_name.startswith("LogisticRegression"):
                    c = params.get("C", 1.0)
                    return {
                        "C": around(c, factors=(0.5, 1.0, 2.0)),
                        "penalty": [params.get("penalty", "l2")],
                        "solver": [params.get("solver", "lbfgs")],
                    }
                return None

            fine_grid = _refine_grid(model_option, best_params)
            if fine_grid:
                progress_bar.progress(0.6, text="GridSearchCV 2단계(Fine) 실행 중...")
                estimator_fine = ImbPipeline([("sampler", sampler_obj), ("model", base_model.__class__(**base_model.get_params()))]) if sampler_obj is not None and _HAS_IMB else base_model.__class__(**base_model.get_params())
                gs_fine = GridSearchCV(
                    estimator=estimator_fine,
                    param_grid=fine_grid,
                    scoring=scoring,
                    cv=int(cv_adj),
                    n_jobs=int(n_jobs),
                    verbose=0,
                )
                try:
                    gs_fine.fit(X, y)
                except Exception as e:
                    if int(n_jobs) > 1:
                        st.warning(f"병렬 GridSearchCV(2단계) 실패로 n_jobs=1로 재시도합니다. 원인: {e}")
                        gs_fine = GridSearchCV(
                            estimator=estimator_fine,
                            param_grid=fine_grid,
                            scoring=scoring,
                            cv=int(cv_adj),
                            n_jobs=1,
                            verbose=0,
                        )
                        gs_fine.fit(X, y)
                    else:
                        raise
                chart_scores.extend(list(gs_fine.cv_results_["mean_test_score"]))
                if gs_fine.best_score_ is not None and (best_score is None or gs_fine.best_score_ > best_score):
                    best_score = float(gs_fine.best_score_)
                    best_params = gs_fine.best_params_
                total_trials += len(gs_fine.cv_results_["params"])

            # 그리드 탐색 결과 차트 시각화
            try:
                history_chart.line_chart(pd.DataFrame(chart_scores, columns=["score"]), height=200)
            except Exception:
                pass

            hpo_result = {
                "best_params": best_params,
                "best_value": best_score,
                "study_summary": {"best_trial": {"value": best_score, "params": best_params}, "n_trials": total_trials},
                "method": "grid",
            }
            progress_bar.progress(1.0, text="GridSearchCV 완료")

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
    best_params = hpo_res.get("best_params") or {}
    best_value = hpo_res.get("best_value")
    with st.expander("베스트 파라미터 / 스코어", expanded=True):
        st.write(f"best_value: {best_value}")
        st.json(best_params)
    with st.expander("Study / Trials", expanded=False):
        if hpo_res.get("study_summary"):
            st.markdown("**Study Summary**")
            st.json(hpo_res.get("study_summary"))
        if isinstance(hpo_res.get("trials"), list):
            st.markdown("**Trials (상위 20개)**")
            st.dataframe(pd.DataFrame(hpo_res.get("trials")).head(20))

    st.markdown("---")
    st.subheader("결과 다운로드")
    params_bytes = json.dumps(best_params, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        "best_params JSON 다운로드",
        data=params_bytes,
        file_name="hpo_best_params.json",
        mime="application/json",
    )

st.markdown("---")
st.info("다음 단계: Validation 페이지에서 모델 검증을 진행하세요.")
