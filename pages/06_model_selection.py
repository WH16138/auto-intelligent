# pages/06_model_selection.py
"""
06 - Model Selection (모델 추천 및 베이스라인 비교)

기능:
- 학습에 사용할 데이터 선택(전처리된 데이터 우선)
- CV 분할 수 및 랜덤시드 지정
- quick_baselines 실행 (modules.model_search.quick_baselines)
- 결과 테이블 표시 및 주요 지표(accuracy / r2 등)로 정렬/시각화
- 특정 모델 선택 → 세션에 저장, 아티팩트로 snapshot 저장(모델 + 파이프라인 + params)
- 선택 모델 다운로드(artifacts로 저장 후 파일 제공)
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import io
import json

st.set_page_config(layout="wide")
st.title("6 - Model Selection (모델 추천 및 비교)")

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_preprocessed" not in st.session_state:
    st.session_state["df_preprocessed"] = None
if "preprocessing_pipeline" not in st.session_state:
    st.session_state["preprocessing_pipeline"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None
if "baselines_df" not in st.session_state:
    st.session_state["baselines_df"] = None
if "baseline_models" not in st.session_state:
    st.session_state["baseline_models"] = None
if "best_model_name" not in st.session_state:
    st.session_state["best_model_name"] = None

# imports (fallback)
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

# pick data
st.subheader("학습 데이터 선택")
preferred_df = st.session_state.get("df_preprocessed") or st.session_state.get("df")
if preferred_df is None:
    st.warning("먼저 Upload / Preprocessing 단계를 완료하세요.")
    st.stop()

st.write(f"사용 가능한 데이터: {preferred_df.shape[0]} 행 × {preferred_df.shape[1]} 열")
target_col = st.session_state.get("target_col")
st.write(f"설정된 타깃 컬럼: `{target_col}`" if target_col else "타깃 컬럼이 설정되어 있지 않습니다. Overview에서 설정하세요.")
# allow override of target for this page
target_override = st.selectbox("이 페이지에서 사용할 타깃 컬럼 (오버라이드 가능)", options=["(세션값 사용)"] + list(preferred_df.columns), index=0)
if target_override == "(세션값 사용)":
    page_target = target_col
else:
    page_target = target_override

# CV and random state
st.subheader("베이스라인 실행 옵션")
col_a, col_b = st.columns(2)
with col_a:
    cv = st.number_input("CV 분할 수 (n_splits)", min_value=2, max_value=10, value=3, step=1)
with col_b:
    random_state = st.number_input("랜덤 시드", min_value=0, value=0, step=1)

# Run baselines
if st.button("빠른 베이스라인 실행 (Quick Baselines)"):
    if page_target is None:
        st.error("타깃 컬럼이 지정되어 있지 않습니다. 먼저 타깃을 지정해주세요.")
    else:
        try:
            with st.spinner("베이스라인 실행 중... 잠시 기다려주세요."):
                results_df, trained_models = ms.quick_baselines(preferred_df.copy(), target_col=page_target, cv=int(cv), random_state=int(random_state))
            st.session_state["baselines_df"] = results_df
            st.session_state["baseline_models"] = trained_models
            # infer best model name and store
            if not results_df.empty:
                # choose primary metric column
                if "mean_accuracy" in results_df.columns:
                    best = results_df.sort_values(by="mean_accuracy", ascending=False).iloc[0]["model"]
                elif "mean_r2" in results_df.columns:
                    best = results_df.sort_values(by="mean_r2", ascending=False).iloc[0]["model"]
                else:
                    best = results_df.iloc[0]["model"]
                st.session_state["best_model_name"] = best
            st.success("베이스라인 실행 완료")
        except Exception as e:
            st.error(f"베이스라인 실행 중 오류가 발생했습니다: {e}")

# show baseline results
st.markdown("---")
st.subheader("베이스라인 결과")
baselines_df: Optional[pd.DataFrame] = st.session_state.get("baselines_df")
if baselines_df is None:
    st.info("베이스라인이 아직 실행되지 않았습니다. 상단의 버튼으로 실행하세요.")
else:
    # display table with relevant columns
    st.dataframe(baselines_df.fillna("N/A").round(4), use_container_width=True)

    # visualization
    if px is not None:
        try:
            if "mean_accuracy" in baselines_df.columns:
                fig = px.bar(baselines_df, x="model", y="mean_accuracy", error_y="std_accuracy", title="Model comparison: mean_accuracy")
                st.plotly_chart(fig, use_container_width=True)
            elif "mean_r2" in baselines_df.columns:
                fig = px.bar(baselines_df, x="model", y="mean_r2", error_y="std_r2", title="Model comparison: mean_r2")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("시각화 생성 실패:", e)

    # let user pick a model to inspect / save
    st.markdown("---")
    st.subheader("모델 선택 및 저장")
    model_options = baselines_df["model"].tolist()
    selected_model_name = st.selectbox("저장할/검사할 모델 선택", options=model_options, index=0)
    if selected_model_name:
        model_entry = st.session_state["baseline_models"].get(selected_model_name)
        if model_entry is None or model_entry.get("model") is None:
            st.warning("선택한 모델은 학습되지 않았거나 오류가 발생했습니다.")
        else:
            mdl = model_entry.get("model")
            feat_names = model_entry.get("feature_names", [])
            st.write("모델 클래스:", type(mdl))
            # show top-level params (short)
            try:
                params = mdl.get_params()
                # pretty print subset
                short = {k: params[k] for k in list(params.keys())[:10]}
                st.write("모델 파라미터(일부):", short)
            except Exception:
                st.write("모델 파라미터 조회 불가")

            # option: save snapshot (model + preprocessor + params)
            if st.button("선택 모델을 artifacts에 snapshot으로 저장"):
                try:
                    # prefer best model object; preprocessor from session
                    preproc = st.session_state.get("preprocessing_pipeline")
                    # try to get params from model.get_params()
                    try:
                        params_obj = mdl.get_params()
                    except Exception:
                        params_obj = None
                    paths = io_utils.snapshot_artifacts(model_obj=mdl, preprocessor_obj=preproc, params=params_obj, base_dir="artifacts", prefix="baseline")
                    st.success(f"아티팩트 저장 완료: {paths}")
                    st.write(paths)
                    # offer download for model file if present
                    if paths.get("model_path"):
                        try:
                            with open(paths["model_path"], "rb") as f:
                                data = f.read()
                            st.download_button("모델 파일 다운로드 (.pkl)", data=data, file_name=paths["model_path"].split("/")[-1], mime="application/octet-stream")
                        except Exception as e:
                            st.info("모델 다운로드 준비 실패: " + str(e))
                except Exception as e:
                    st.error(f"아티팩트 저장 실패: {e}")

            # quick predict demo (single row)
            st.markdown("간단 예측 데모 (상위 1개 샘플 사용)")
            try:
                # prepare a single sample X using feature names (if feature names exist)
                if len(feat_names) > 0:
                    sample_df = preferred_df[feat_names].head(1).fillna(0)
                    pred = mdl.predict(sample_df.values)
                    st.write("입력(샘플):")
                    st.dataframe(sample_df)
                    st.write("모델 예측 결과:", pred)
                else:
                    # fallback numeric columns
                    num_cols = preferred_df.select_dtypes(include=["number"]).columns.tolist()
                    if num_cols:
                        sample_df = preferred_df[num_cols].head(1).fillna(0)
                        pred = mdl.predict(sample_df.values)
                        st.write("입력(샘플):")
                        st.dataframe(sample_df)
                        st.write("모델 예측 결과:", pred)
                    else:
                        st.info("예측 데모를 위해 사용 가능한 숫자형 피처가 없습니다.")
            except Exception as e:
                st.warning("간단 예측 데모 실패: " + str(e))

# quick action: accept best model as final
st.markdown("---")
st.subheader("최종 모델로 채택")
if st.session_state.get("best_model_name"):
    st.write(f"현재 추천된 최적 모델: `{st.session_state.get('best_model_name')}`")
else:
    st.info("베이스라인 결과로부터 추천된 최적 모델이 아직 없습니다.")

if st.button("추천된 최적 모델을 최종 모델로 채택 및 snapshot 저장"):
    best_name = st.session_state.get("best_model_name")
    if not best_name:
        st.error("추천된 최적 모델이 없습니다.")
    else:
        best_entry = st.session_state.get("baseline_models", {}).get(best_name)
        if best_entry is None or best_entry.get("model") is None:
            st.error("추천된 모델이 준비되지 않았습니다.")
        else:
            try:
                mdl = best_entry.get("model")
                preproc = st.session_state.get("preprocessing_pipeline")
                try:
                    params_obj = mdl.get_params()
                except Exception:
                    params_obj = None
                p = io_utils.snapshot_artifacts(model_obj=mdl, preprocessor_obj=preproc, params=params_obj, base_dir="artifacts", prefix="final")
                st.success(f"최종 모델 snapshot 저장 완료: {p}")
                st.write(p)
                # generate reproducible train script if params available
                if params_obj:
                    script_path = f"scripts/train_best_final.py"
                    io_utils.generate_train_script(params=params_obj, model_filename=p.get("model_path", "artifacts/final_best_model.pkl"), pipeline_filename=p.get("pipeline_path"), script_path=script_path, target_col=page_target or "target")
                    st.success(f"재현 스크립트 생성: {script_path}")
                    try:
                        with open(script_path, "r", encoding="utf-8") as f:
                            st.code(f.read()[:1000] + "\n...\n", language="python")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"최종 모델 채택 실패: {e}")

st.markdown("---")
st.write("다음 단계: Hyperparameter Tuning 페이지로 이동하거나 Evaluation 페이지에서 최종 모델 성능을 확인하세요.")
