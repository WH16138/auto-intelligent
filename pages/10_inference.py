"""
10 - Model Inference & Download

기능:
- 세션에 저장된 모델(훈련된 모델 또는 베이스라인) 선택
- 전처리 파이프라인과 특징공학 메타를 사용해 업로드된 CSV를 변환 후 예측
- 예측 결과(target 컬럼 포함)를 CSV로 다운로드
- 모델 파일 다운로드
"""
import io
import pickle
from typing import Optional, List

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("10 - 모델 추론 및 다운로드")
st.info(
    "이미 학습된 모델로 새 CSV를 예측하고, Kaggle/Dacon 제출용 파일까지 바로 만들 수 있습니다."
)
with st.expander("빠른 사용 순서 보기", expanded=True):
    st.markdown(
        "1) **모델 선택** (세션에 저장된 학습/베이스라인)\n"
        "2) **예측 옵션**: 예측 컬럼명·제출 파일명·ID 컬럼 지정\n"
        "3) **CSV 업로드** → 자동 전처리/특징공학/정렬 후 예측\n"
        "4) **결과 다운로드**: 전체 예측 CSV + 제출용(ID+예측)\n"
        "5) **모델 다운로드**: 하단에서 피클로 내보내 재사용/배포 가능\n"
    )

# Session defaults
for key in [
    "df_features",
    "df_features_train",
    "df_features_test",
    "preprocessing_pipeline",
    "feature_engineering_meta",
    "target_col",
    "trained_model",
    "baseline_models",
    "best_model_name",
    "inference_uploaded_model",
    "inference_uploaded_feature_names",
]:
    st.session_state.setdefault(key, None)

try:
    from modules import ingestion
except Exception:
    import modules.ingestion as ingestion

try:
    from modules import io_utils
except Exception:
    io_utils = None

try:
    from modules import feature_engineering as fe_mod
except Exception:
    fe_mod = None


def _apply_fe_meta(df_full: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Apply feature_engineering_meta rules (created on train) to full dataframe."""
    if fe_mod is None or not meta:
        return df_full
    df_out = df_full.copy()
    methods = meta.get("method", {})
    new_feats = meta.get("new_features", [])
    for f in new_feats:
        method = methods.get(f)
        try:
            if method == "ratio" and "__div__" in f:
                a, b = f.split("__div__")
                df_out[f] = fe_mod._safe_divide(df_out[a], df_out[b])  # type: ignore
            elif method == "diff" and "__minus__" in f:
                a, b = f.split("__minus__")
                df_out[f] = pd.to_numeric(df_out[a], errors="coerce").fillna(0) - pd.to_numeric(df_out[b], errors="coerce").fillna(0)
            elif method == "sum" and "__plus__" in f:
                a, b = f.split("__plus__")
                df_out[f] = pd.to_numeric(df_out[a], errors="coerce").fillna(0) + pd.to_numeric(df_out[b], errors="coerce").fillna(0)
            elif method == "log1p" and f.endswith("__log1p"):
                base = f[: -len("__log1p")]
                df_out[f] = fe_mod._safe_log1p(df_out[base])  # type: ignore
            elif method == "datetime_extract" and "__" in f:
                base, attr = f.split("__", 1)
                s = pd.to_datetime(df_out[base], errors="coerce")
                if attr == "weekday":
                    df_out[f] = s.dt.weekday.fillna(-1).astype(int)
                elif attr == "hour":
                    df_out[f] = s.dt.hour.fillna(-1).astype(int)
                else:
                    df_out[f] = getattr(s.dt, attr).fillna(-1).astype(int)
        except Exception:
            continue
    return df_out


def _resolve_model() -> Optional[dict]:
    """Return selected model entry and name. 옵션을 두 가지로 단순화: 세션 추천 베이스라인 vs HPO 튜닝."""
    trained_model = st.session_state.get("trained_model")
    baselines = st.session_state.get("baseline_models") or {}
    best_baseline_name = st.session_state.get("best_model_name")
    hpo_res = st.session_state.get("hpo_result")

    options = []
    baseline_label = None
    hpo_label = None

    if best_baseline_name and baselines.get(best_baseline_name):
        baseline_label = f"세션 추천 모델 (베이스라인: {best_baseline_name})"
    elif baselines:
        first_name = list(baselines.keys())[0]
        baseline_label = f"세션 추천 모델 (베이스라인: {first_name})"

    if trained_model is not None and hpo_res:
        hpo_label = "HPO 튜닝 모델 (세션 저장본)"

    if baseline_label:
        options.append(baseline_label)
    if hpo_label:
        options.append(hpo_label)

    if not options:
        st.error("사용 가능한 모델이 없습니다. 이전 단계에서 모델을 학습하세요.")
        return None

    choice = st.selectbox("사용할 모델 선택", options=options, index=0)
    if baseline_label and choice == baseline_label:
        entry = baselines.get(best_baseline_name) if best_baseline_name and baselines.get(best_baseline_name) else list(baselines.values())[0]
        return {"name": best_baseline_name or "baseline", "model": entry.get("model"), "feature_names": entry.get("feature_names")}
    if hpo_label and choice == hpo_label and trained_model is not None:
        return {"name": "hpo_trained", "model": trained_model, "feature_names": st.session_state.get("feature_names")}
    return None


def _align_features(df_feat: pd.DataFrame, feature_names: Optional[List[str]]) -> pd.DataFrame:
    """Align dataframe columns to model's feature names."""
    if not feature_names:
        return df_feat.select_dtypes(include=["number"])
    aligned = pd.DataFrame()
    for col in feature_names:
        if col in df_feat.columns:
            aligned[col] = df_feat[col]
        else:
            aligned[col] = 0  # fill missing with 0
    return aligned


# ---- UI ----
target_col = st.session_state.get("target_col")
preproc_obj = st.session_state.get("preprocessing_pipeline")
fe_meta = st.session_state.get("feature_engineering_meta") or {}
fe_top_features = st.session_state.get("fe_top_features")

st.info(
    "이 페이지는 **저장된 전처리 파이프라인 + 특징공학 메타 + 모델**을 그대로 사용해 "
    "새로운 CSV에 대한 예측을 수행하고, 예측이 추가된 CSV 또는 제출용 컬럼만 포함한 CSV를 다운로드할 수 있게 합니다."
)

model_entry = _resolve_model()
if model_entry is None or model_entry.get("model") is None:
    st.stop()

st.write(f"선택된 모델: `{model_entry.get('name')}` | feature_names: {len(model_entry.get('feature_names') or [])}")

if preproc_obj is None:
    st.error("전처리 파이프라인이 세션에 없습니다. 04 전처리 단계에서 파이프라인을 적용한 뒤 다시 시도하세요.")
    st.stop()

st.markdown("### 예측 옵션")
pred_col_name = st.text_input("예측 컬럼 이름", value=target_col or "prediction")
submit_filename = st.text_input("제출용 파일 이름", value="submission.csv")
id_column = st.text_input("ID 컬럼 이름 (없으면 입력 생략)", value="ID")
st.caption("ID 컬럼을 지정하면 제출용 파일에 함께 포함됩니다. 없으면 예측 컬럼만 생성합니다.")

uploaded_pred = st.file_uploader("예측용 CSV 업로드 (타깃 컬럼 제외)", type=["csv"], accept_multiple_files=False, key="inference_uploader")

if uploaded_pred is not None:
    try:
        try:
            df_pred_orig = ingestion.safe_read_csv(uploaded_pred)
        except Exception:
            uploaded_pred.seek(0)
            df_pred_orig = pd.read_csv(uploaded_pred)
        st.write(f"입력 데이터: {df_pred_orig.shape[0]} 행 × {df_pred_orig.shape[1]} 열")
        df_work = df_pred_orig.copy()
        if target_col and target_col in df_work.columns:
            st.warning(f"입력에 타깃 컬럼 `{target_col}`이 포함되어 제거합니다.")
            df_work = df_work.drop(columns=[target_col])

        # 1) 전처리
        if preproc_obj is not None:
            try:
                from modules import preprocessing as prep
                df_work = prep.apply_preprocessor(preproc_obj, df_work)
            except Exception as e:
                st.error(f"전처리 적용 실패: {e}")
                st.stop()
        else:
            df_work = df_work.select_dtypes(include=["number"])

        # 2) 특징공학 메타 적용
        df_work = _apply_fe_meta(df_work, fe_meta)

        # 2.5) 추천 특징만 남기기 (있다면)
        if fe_top_features:
            keep_cols = [c for c in fe_top_features if c in df_work.columns]
            if keep_cols:
                df_work = df_work[keep_cols].copy()

        # 3) 모델 feature alignment
        feature_names = model_entry.get("feature_names")
        df_aligned = _align_features(df_work, feature_names)

        # 4) 예측
        model_obj = model_entry.get("model")
        preds = model_obj.predict(df_aligned.values)
        df_result = df_pred_orig.copy()
        target_name = pred_col_name or target_col or "prediction"
        df_result[target_name] = preds

        st.subheader("예측 결과 미리보기")
        st.dataframe(df_result.head(50))
        csv_bytes = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "예측 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="prediction_result.csv",
            mime="text/csv",
        )
        # Submission-only file (id + prediction) (콘테스트 날먹용)
        sub_df = pd.DataFrame({target_name: preds})
        if id_column and id_column in df_pred_orig.columns:
            sub_df.insert(0, id_column, df_pred_orig[id_column].values)
        else:
            st.warning("ID 컬럼이 없어 단일 예측 컬럼만 포함된 제출 파일을 생성합니다.")
        sub_csv = sub_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "제출용 CSV 다운로드 (예측 컬럼만)",
            data=sub_csv,
            file_name=submit_filename or "submission.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"예측 파이프라인 실행 실패: {e}")

st.markdown("---")
st.subheader("모델 다운로드")
if st.button("모델 파일 다운로드 준비"):
    try:
        data = pickle.dumps(model_entry.get("model"))
        st.download_button(
            "모델 피클 다운로드",
            data=data,
            file_name=f"{model_entry.get('name') or 'model'}.pkl",
            mime="application/octet-stream",
        )
        st.success("모델 직렬화 및 다운로드 준비 완료")
    except Exception as e:
        st.error(f"모델 직렬화 실패: {e}")
st.info("모델 파일은 피클 형식(.pkl)으로 다운로드됩니다. 재사용 또는 배포 시 활용하세요.")
