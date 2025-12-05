# -*- coding: utf-8 -*-
# pages/06_model_selection.py
"""
6단계 - 모델 선택

- df_features 사용
- 타깃 컬럼 지정 후 베이스라인 모델들을 교차검증으로 비교
- 결과/시각화/스냅샷/간단 예측 제공
"""
import json
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("6 - Model Selection")
st.info("train 기준 데이터를 사용해 빠르게 베이스라인을 비교합니다. 기본값 그대로 실행해도 되며, 결과 표와 그래프를 확인한 뒤 최적 모델을 저장하세요.")

# Session defaults
for key, default in [
    ("df_original", None),
    ("df_dropped", None),
    ("df_preprocessed", None),
    ("df_features", None),
    ("df_features_train", None),
    ("df_features_test", None),
    ("df", None),
    ("preprocessing_pipeline", None),
    ("target_col", None),
    ("baselines_df", None),
    ("baseline_models", None),
    ("best_model_name", None),
]:
    st.session_state.setdefault(key, default)

# Imports
try:
    from modules import model_search as ms
except Exception:
    import modules.model_search as ms  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore

try:
    import plotly.express as px
except Exception:
    px = None


def get_active_df():
    return st.session_state.get("df_features")

def _safe_cv_for_data(df: pd.DataFrame, target: str, requested_cv: int) -> Tuple[Optional[int], Optional[str]]:
    n_samples = len(df)
    if n_samples < 2:
        return None, "샘플 수가 2개 미만입니다."
    cv_adj = max(2, min(requested_cv, n_samples))
    s = df[target]
    if pd.api.types.is_numeric_dtype(s):
        nunique = int(s.nunique(dropna=True))
    else:
        nunique = int(s.astype(str).nunique(dropna=True))
    if nunique <= 20:
        counts = s.value_counts(dropna=True)
        min_count = int(counts.min()) if not counts.empty else 0
        if min_count < 2:
            return None, "클래스별 샘플 수가 부족합니다."
        cv_adj = min(cv_adj, min_count)
    return max(cv_adj, 2), None


# Pick data
st.subheader("데이터 선택")
active_df = get_active_df()
if active_df is None:
    st.warning("Upload 이후 데이터를 준비한 뒤 다시 시도하세요.")
    st.stop()
train_idx = st.session_state.get("train_idx")
if isinstance(train_idx, list) and len(train_idx) > 0:
    active_df_train = active_df.iloc[train_idx].copy()
else:
    active_df_train = active_df
problem_type_override = st.session_state.get("problem_type")

col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.metric("train 행 수", active_df_train.shape[0])
with col_meta2:
    st.metric("전체 열 수", active_df_train.shape[1])
with col_meta3:
    st.caption("tip: 전처리/특징공학 단계에서 만든 컬럼을 그대로 사용합니다.")

# Target selection
action_target = st.session_state.get("target_col")
cols: List[str] = active_df.columns.tolist()
target_override = st.selectbox(
    "타깃 컬럼 (필수)",
    options=["(세션 값 사용)"] + cols,
    index=0 if action_target is None else cols.index(action_target) + 1 if action_target in cols else 0,
)
target_col = action_target if target_override == "(세션 값 사용)" else target_override
if target_col is None or target_col not in active_df.columns:
    st.error("타깃 컬럼을 선택하세요.")
    st.stop()

# Diagnostics
work_df = active_df_train.dropna(subset=[target_col]).copy()
target_series = work_df[target_col]
target_nulls = int(active_df[target_col].isnull().sum())
unique_classes = target_series.nunique()
numeric_features = [c for c in work_df.select_dtypes(include=["number"]).columns if c != target_col]
st.markdown(
    f"- 타깃 고유값: **{unique_classes}** | 타깃 결측: **{target_nulls}** | 수치형 특징 수: **{len(numeric_features)}**"
)
if unique_classes <= 1:
    st.error("타깃이 단일 값입니다.")
    st.stop()
if len(numeric_features) == 0:
    st.error("사용 가능한 수치형 특징이 없습니다.")
    st.stop()
if target_nulls > 0:
    st.warning("타깃 결측 행을 제거하고 실행합니다.")
if problem_type_override:
    st.info(f"문제 유형 설정: {problem_type_override} (Overview에서 지정)")

# CV/random_state
st.subheader("베이스라인 실행 설정")
st.caption("기본값을 권장합니다. CV는 클래스 최소 빈도에 맞춰 자동 조정됩니다.")
col_a, col_b = st.columns(2)
with col_a:
    cv = st.number_input("CV 분할 수", min_value=2, max_value=10, value=3, step=1, help="교차검증 폴드 수")
with col_b:
    random_state = st.number_input("random_state", min_value=0, value=0, step=1, help="재현성을 위한 시드")

# Run baselines
if st.button("빠른 베이스라인 실행 (Quick Baselines)"):
    progress_bar = st.progress(0, text="베이스라인 실행 준비 중...")
    status_placeholder = st.empty()
    try:
        cv_adj, cv_warn = _safe_cv_for_data(work_df, target_col, int(cv))
        if cv_adj is None:
            st.error(cv_warn)
        else:
            if cv_warn:
                st.warning(cv_warn)
            progress_bar.progress(30, text=f"베이스라인 실행 중... (cv={cv_adj})")
            status_placeholder.write("교차검증 수행 및 모델 학습 중...")
            with st.spinner(f"베이스라인 실행 중... (cv={cv_adj})"):
                results_df, trained_models = ms.quick_baselines(
                    work_df,
                    target_col=target_col,
                    cv=int(cv_adj),
                    random_state=int(random_state),
                    forced_task=problem_type_override,
                )
            progress_bar.progress(100, text="베이스라인 실행 완료")
            st.session_state["baselines_df"] = results_df
            st.session_state["baseline_models"] = trained_models
            best_name = None
            if not results_df.empty:
                if "mean_accuracy" in results_df.columns:
                    best_name = results_df.sort_values(by="mean_accuracy", ascending=False).iloc[0]["model"]
                elif "mean_r2" in results_df.columns:
                    best_name = results_df.sort_values(by="mean_r2", ascending=False).iloc[0]["model"]
                else:
                    best_name = results_df.iloc[0]["model"]
            st.session_state["best_model_name"] = best_name
            st.session_state["target_col"] = target_col
            if results_df.replace({np.nan: None}).drop(columns=["model", "task"], errors="ignore").isna().all().all():
                st.warning("모든 스코어가 NaN입니다. 데이터/타깃/특징을 확인하세요.")
            else:
                st.success("베이스라인 실행 완료.")
    except Exception as e:
        st.error(f"베이스라인 실행 실패: {e}")

# Show baseline results
st.markdown("---")
st.subheader("베이스라인 결과")
baselines_df: Optional[pd.DataFrame] = st.session_state.get("baselines_df")
baseline_models = st.session_state.get("baseline_models") or {}

if baselines_df is None:
    st.info("아직 베이스라인을 실행하지 않았습니다.")
else:
    with st.expander("결과 표 보기", expanded=True):
        st.dataframe(baselines_df.fillna("N/A").round(4), use_container_width=True)

    if px is not None:
        try:
            if "mean_accuracy" in baselines_df.columns:
                fig = px.bar(baselines_df, x="model", y="mean_accuracy", error_y="std_accuracy", title="Model comparison: mean_accuracy")
                st.plotly_chart(fig, use_container_width=True)
            elif "mean_r2" in baselines_df.columns:
                fig = px.bar(baselines_df, x="model", y="mean_r2", error_y="std_r2", title="Model comparison: mean_r2")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"시각화 실패: {e}")

    st.markdown("---")
    st.subheader("모델 선택 및 스냅샷")
    model_options = baselines_df["model"].tolist()
    selected_model_name = st.selectbox("확인/저장할 모델 선택", options=model_options, index=0)

    selected_entry = baseline_models.get(selected_model_name)
    if selected_entry is None or selected_entry.get("model") is None:
        st.warning("선택한 모델 객체를 찾을 수 없습니다.")
    else:
        mdl = selected_entry.get("model")
        feat_names = selected_entry.get("feature_names", [])
        with st.expander("모델 정보 보기", expanded=False):
            st.write("모델 타입:", type(mdl))
            try:
                params = mdl.get_params()
                short_params = {k: params[k] for k in list(params.keys())[:10]}
                st.write("모델 파라미터(일부):", short_params)
            except Exception:
                st.write("모델 파라미터를 불러올 수 없습니다.")

        if st.button("세션에 이 모델 저장 (trained_model)"):
            st.session_state["trained_model"] = mdl
            st.session_state["feature_names"] = feat_names
            st.session_state["problem_type"] = selected_entry.get("task")
            st.success("선택 모델을 세션에 저장했습니다. Validation/Inference에서 사용 가능합니다.")

        if st.button("선택 모델 아티팩트 저장 (artifacts/)"):
            try:
                preproc = st.session_state.get("preprocessing_pipeline")
                try:
                    params_obj = mdl.get_params()
                except Exception:
                    params_obj = None
                paths = io_utils.snapshot_artifacts(
                    model_obj=mdl,
                    preprocessor_obj=preproc,
                    params=params_obj,
                    base_dir="artifacts",
                    prefix="baseline",
                )
                st.success(f"스냅샷 저장 완료: {paths}")
                if paths.get("model_path"):
                    try:
                        with open(paths["model_path"], "rb") as f:
                            data = f.read()
                        st.download_button(
                            "모델 파일 다운로드 (.pkl)",
                            data=data,
                            file_name=paths["model_path"].split("/")[-1],
                            mime="application/octet-stream",
                        )
                    except Exception as e:
                        st.info(f"모델 다운로드 준비 실패: {e}")
            except Exception as e:
                st.error(f"스냅샷 저장 실패: {e}")

        with st.expander("간단 예측 데모 (상위 1개 샘플)", expanded=False):
            try:
                if feat_names:
                    sample_df = active_df[feat_names].head(1).fillna(0)
                else:
                    num_cols = active_df.select_dtypes(include=["number"]).columns.tolist()
                    sample_df = active_df[num_cols].head(1).fillna(0) if num_cols else None
                if sample_df is not None and not sample_df.empty:
                    pred = mdl.predict(sample_df.values)
                    st.write("입력 샘플:")
                    st.dataframe(sample_df)
                    st.write("모델 예측 결과:", pred)
                else:
                    st.info("예측 데모에 사용할 수치형 컬럼이 없습니다.")
            except Exception as e:
                st.warning(f"예측 데모 실패: {e}")

# Accept best model quick action
st.markdown("---")
st.subheader("최적 모델 저장")
best_name = st.session_state.get("best_model_name")
if best_name:
    st.write(f"현재 추천 최적 모델: `{best_name}`")
else:
    st.info("베이스라인 결과에서 최적 모델을 아직 산출하지 못했습니다.")

if st.button("추천 최적 모델을 최종 스냅샷으로 저장"):
    if not best_name:
        st.error("추천 최적 모델이 없습니다.")
    else:
        best_entry = (st.session_state.get("baseline_models") or {}).get(best_name)
        if best_entry is None or best_entry.get("model") is None:
            st.error("추천 모델 객체를 찾을 수 없습니다.")
        else:
            try:
                mdl = best_entry.get("model")
                preproc = st.session_state.get("preprocessing_pipeline")
                try:
                    params_obj = mdl.get_params()
                except Exception:
                    params_obj = None
                paths = io_utils.snapshot_artifacts(
                    model_obj=mdl,
                    preprocessor_obj=preproc,
                    params=params_obj,
                    base_dir="artifacts",
                    prefix="final",
                )
                st.session_state["trained_model"] = mdl
                st.session_state["feature_names"] = best_entry.get("feature_names")
                st.session_state["problem_type"] = best_entry.get("task")
                st.success(f"최종 모델 스냅샷 저장 완료: {paths}")
            except Exception as e:
                st.error(f"최종 모델 저장 실패: {e}")

st.markdown("---")
st.write("다음 단계: Hyperparameter Tuning 페이지로 이동하세요.")
