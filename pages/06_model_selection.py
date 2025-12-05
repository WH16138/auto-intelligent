# pages/06_model_selection.py
"""
Step 6 - Model Selection

Flow:
- Choose the active dataframe (prefer preprocessed if available).
- Confirm/override target column.
- Run quick baselines (cross-validated) via modules.model_search.quick_baselines.
- Inspect metrics, visualize comparisons, and snapshot the chosen model.
"""
import io
import json
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("6 - Model Selection")

# Session defaults
st.session_state.setdefault("df", None)
st.session_state.setdefault("df_preprocessed", None)
st.session_state.setdefault("preprocessing_pipeline", None)
st.session_state.setdefault("target_col", None)
st.session_state.setdefault("baselines_df", None)
st.session_state.setdefault("baseline_models", None)
st.session_state.setdefault("best_model_name", None)

# Imports (fallback)
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


def get_active_df() -> Tuple[Optional[pd.DataFrame], str]:
    """Return (df, source_label)."""
    if st.session_state.get("df_features") is not None:
        return st.session_state["df_features"], "df_features (특징공학/전처리 반영)"
    if st.session_state.get("df_preprocessed") is not None:
        return st.session_state["df_preprocessed"], "df_preprocessed (전처리)"
    if st.session_state.get("df") is not None:
        return st.session_state["df"], "df (원본)"
    return None, "없음"

def _safe_cv_for_data(df: pd.DataFrame, target: str, requested_cv: int) -> Tuple[Optional[int], Optional[str]]:
    """Adjust CV folds based on sample/class counts; return (cv, warning)."""
    n_samples = len(df)
    if n_samples < 2:
        return None, "샘플 수가 2개 미만이라 교차검증을 수행할 수 없습니다."
    cv_adj = max(2, min(requested_cv, n_samples))
    s = df[target]
    if pd.api.types.is_numeric_dtype(s):
        nunique = int(s.nunique(dropna=True))
    else:
        nunique = int(s.astype(str).nunique(dropna=True))
    # classification heuristic: if nunique is small treat as classification
    if nunique <= 20:
        # need at least 2 samples per class for StratifiedKFold
        counts = s.value_counts(dropna=True)
        min_count = int(counts.min()) if not counts.empty else 0
        if min_count < 2:
            return None, "타깃 클래스별 샘플 수가 부족해 교차검증을 수행할 수 없습니다. 데이터를 늘리거나 타깃을 확인하세요."
        cv_adj = min(cv_adj, min_count)
    return max(cv_adj, 2), None

# Pick data
st.subheader("데이터 선택")
active_df, df_source = get_active_df()
if active_df is None:
    st.warning("Upload / Preprocessing 페이지에서 데이터를 준비한 뒤 다시 시도하세요.")
    st.stop()

st.write(
    f"사용 데이터: {active_df.shape[0]} 행 × {active_df.shape[1]} 열 "
    f"(preprocessed 사용: {'예' if used_pre else '아니요'})"
)

# Target selection (prefer session target, allow override)
session_target = st.session_state.get("target_col")
columns_list: List[str] = active_df.columns.tolist()
target_override = st.selectbox(
    "타깃 컬럼 선택 (필수)",
    options=["(세션 값 사용)"] + columns_list,
    index=0 if session_target is None else columns_list.index(session_target) + 1 if session_target in columns_list else 0,
)
if target_override == "(세션 값 사용)":
    target_col = session_target
else:
    target_col = target_override

if target_col is None:
    st.error("타깃 컬럼을 선택하세요. Overview 페이지에서 설정했거나 여기서 선택할 수 있습니다.")
    st.stop()
elif target_col not in active_df.columns:
    st.error(f"타깃 컬럼 `{target_col}` 이(가) 데이터프레임에 없습니다.")
    st.stop()

# Diagnostics: target health & usable feature count
target_series = active_df[target_col]
target_nulls = int(target_series.isnull().sum())
unique_classes = target_series.dropna().nunique()
numeric_features = [c for c in active_df.select_dtypes(include=["number"]).columns if c != target_col]
st.markdown(
    f"- 타깃 고유값 개수: **{unique_classes}**  |  결측치: **{target_nulls}**\n"
    f"- 사용 가능한 수치형 특징 수: **{len(numeric_features)}**"
)
if target_nulls > 0:
    st.warning("타깃에 결측치가 있습니다. 결측 행은 베이스라인 실행 시 자동 제거됩니다.")
if unique_classes <= 1:
    st.error("타깃이 단일 클래스입니다. 최소 두 개 이상의 클래스/값이 필요합니다.")
    st.stop()
if len(numeric_features) == 0:
    st.error("사용 가능한 수치형 특징이 없습니다. 전처리/인코딩 단계를 확인하세요.")
    st.stop()

# CV and random state
st.subheader("베이스라인 실행 설정")
col_a, col_b = st.columns(2)
with col_a:
    cv = st.number_input("CV 분할 수 (n_splits)", min_value=2, max_value=10, value=3, step=1)
with col_b:
    random_state = st.number_input("random_state", min_value=0, value=0, step=1)

# Run baselines
if st.button("빠른 베이스라인 실행 (Quick Baselines)"):
    try:
        # drop rows with missing target to avoid estimator errors
        work_df = active_df.dropna(subset=[target_col]).copy()
        cv_adj, cv_warn = _safe_cv_for_data(work_df, target_col, int(cv))
        if cv_adj is None:
            st.error(cv_warn)
        else:
            if cv_warn:
                st.warning(cv_warn)
            with st.spinner(f"베이스라인 실행 중... (cv={cv_adj})"):
                results_df, trained_models = ms.quick_baselines(
                    work_df,
                    target_col=target_col,
                    cv=int(cv_adj),
                    random_state=int(random_state),
                )
            st.session_state["baselines_df"] = results_df
            st.session_state["baseline_models"] = trained_models
            # choose best model name
            best_name = None
            if not results_df.empty:
                if "mean_accuracy" in results_df.columns:
                    best_name = results_df.sort_values(by="mean_accuracy", ascending=False).iloc[0]["model"]
                elif "mean_r2" in results_df.columns:
                    best_name = results_df.sort_values(by="mean_r2", ascending=False).iloc[0]["model"]
                else:
                    best_name = results_df.iloc[0]["model"]
            st.session_state["best_model_name"] = best_name
            st.session_state["target_col"] = target_col  # persist latest choice
            # warn if everything is nan
            if results_df.replace({np.nan: None}).drop(columns=["model", "task"], errors="ignore").isna().all().all():
                st.warning("모든 스코어가 NaN입니다. 데이터 크기/타깃/특징을 확인하세요.")
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
    st.dataframe(baselines_df.fillna("N/A").round(4), use_container_width=True)

    # Visualization
    if px is not None:
        try:
            if "mean_accuracy" in baselines_df.columns:
                fig = px.bar(
                    baselines_df, x="model", y="mean_accuracy", error_y="std_accuracy", title="Model comparison: mean_accuracy"
                )
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
        st.write("모델 타입:", type(mdl))

        try:
            params = mdl.get_params()
            short_params = {k: params[k] for k in list(params.keys())[:10]}
            st.write("모델 파라미터(일부):", short_params)
        except Exception:
            st.write("모델 파라미터를 불러올 수 없습니다.")

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

        st.markdown("간단 예측 데모 (상위 1개 샘플)")
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
                st.info("예측 데모에 사용할 수 있는 수치형 컬럼이 없습니다.")
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
                st.success(f"최종 모델 스냅샷 저장 완료: {paths}")
            except Exception as e:
                st.error(f"최종 모델 저장 실패: {e}")

st.markdown("---")
st.write("다음 단계: Hyperparameter Tuning 페이지에서 추가 개선을 진행하세요.")
