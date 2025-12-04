# pages/05_feature_engineering.py
"""
05 - Feature Engineering 페이지

기능:
- 현재(전처리된) 데이터 선택 (st.session_state['df_preprocessed'] 우선, 없으면 st.session_state['df'])
- 규칙 기반 자동 피처 생성(auto_generate_features) 및 생성 메타 표시
- 생성된 피처 미리보기(상위 N행) 및 새 피처 선택 적용/취소
- 중요도 기반 피처 선택(select_features_by_importance) 지원
- 생성된 피처를 세션에 영구 반영(데이터 교체)하거나 별도 다운로드
- 아티팩트(생성 메타) 저장 지원 (modules.io_utils)
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any

st.set_page_config(layout="wide")
st.title("5 - Feature Engineering (특징공학)")

# session defaults
if "df" not in st.session_state:
    st.session_state["df"] = None
if "df_preprocessed" not in st.session_state:
    st.session_state["df_preprocessed"] = None
if "feature_engineering_meta" not in st.session_state:
    st.session_state["feature_engineering_meta"] = None

# imports (fallbacks)
try:
    from modules import feature_engineering as fe_mod
except Exception:
    import modules.feature_engineering as fe_mod  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore

# choose base dataframe: prefer preprocessed version if exists
base_df = st.session_state.get("df_preprocessed") or st.session_state.get("df")
if base_df is None:
    st.warning("먼저 Upload(또는 Preprocessing) 페이지에서 데이터를 준비하세요.")
    st.stop()

st.write(f"분석용 데이터: {base_df.shape[0]} 행 × {base_df.shape[1]} 열 (preprocessed 사용 여부: {'예' if st.session_state.get('df_preprocessed') is not None else '아니요'})")

# Controls
with st.sidebar:
    st.header("자동 피처 생성 옵션")
    max_new = st.number_input("최대 생성 피처 수 (max_new)", min_value=1, max_value=200, value=10, step=1)
    methods = st.multiselect("사용할 변환 방법 (methods)", options=["ratio", "diff", "sum", "log", "datetime"], default=["ratio", "diff", "log", "datetime"])
    datetime_extract = st.checkbox("datetime 컬럼에서 연/월/일/요일/시간 추출", value=True)
    preview_rows = st.number_input("미리보기 행 수", min_value=5, max_value=500, value=50, step=5)
    save_meta_name = st.text_input("생성 메타 저장 경로", value="artifacts/fe_meta.json")
    st.markdown("---")
    st.header("피처 선택 (중요도 기반)")
    use_importance = st.checkbox("중요도 기반 피처 선택 실행", value=False)
    top_k = st.number_input("중요도 기반 선택: 상위 k개", min_value=1, max_value=500, value=20, step=1)
    st.markdown("---")
    st.write("설명:")
    st.write("- 자동 생성은 안전한 규칙(비율/차/로그 등)을 사용합니다. 생성 수를 적절히 제한하세요.")
    st.write("- 중요도 기반 선택은 RandomForest를 사용합니다 (숫자형 우선).")

st.markdown("---")

# 1) 자동 피처 생성 실행
st.subheader("자동 피처 생성 (규칙 기반)")
col_gen_a, col_gen_b = st.columns([2,1])
with col_gen_a:
    if st.button("자동 피처 생성 실행"):
        try:
            df_extended, meta = fe_mod.auto_generate_features(base_df, max_new=int(max_new), methods=methods, datetime_extract=bool(datetime_extract))
            st.session_state["fe_generated_df"] = df_extended
            st.session_state["feature_engineering_meta"] = meta
            st.success(f"자동 피처 생성 완료 — {len(meta.get('new_features', []))}개 피처 생성")
        except Exception as e:
            st.error(f"자동 피처 생성 실패: {e}")

with col_gen_b:
    if st.session_state.get("feature_engineering_meta"):
        meta = st.session_state["feature_engineering_meta"]
        st.markdown("**최근 생성 메타**")
        st.json(meta)
    else:
        st.info("아직 생성된 피처가 없습니다. '자동 피처 생성 실행'을 눌러보세요.")

st.markdown("---")

# 2) 미리보기 및 선택 적용
st.subheader("생성 피처 미리보기 및 적용")
if st.session_state.get("fe_generated_df") is None:
    st.info("생성된 피처가 없습니다. 먼저 자동 생성 또는 외부 작업을 통해 피처를 추가하세요.")
else:
    df_gen = st.session_state["fe_generated_df"]
    new_feats = st.session_state["feature_engineering_meta"].get("new_features", [])
    st.markdown(f"생성된 전체 피처 수: **{len(new_feats)}**")
    # show head
    st.caption("데이터 미리보기 (원본 + 생성 피처)")
    st.dataframe(df_gen.head(int(preview_rows)))

    # allow user to select which new features to keep
    keep = st.multiselect("세션에 반영할 생성 피처 선택 (여러개)", options=new_feats, default=new_feats)
    if st.button("선택한 생성 피처만 원래 데이터에 합치기 (세션 반영)"):
        try:
            df_base_copy = base_df.copy()
            for f in keep:
                if f in df_gen.columns:
                    df_base_copy[f] = df_gen[f].values
            # update session: if preprocessed existed, update df_preprocessed; else update df
            if st.session_state.get("df_preprocessed") is not None:
                st.session_state["df_preprocessed"] = df_base_copy
            else:
                st.session_state["df"] = df_base_copy
            # store meta
            st.session_state["feature_engineering_meta"] = st.session_state.get("feature_engineering_meta") or {}
            st.success(f"선택 피처 {len(keep)}개가 세션 데이터에 반영되었습니다.")
        except Exception as e:
            st.error(f"생성 피처 반영 실패: {e}")

    if st.button("생성된 전체 데이터로 세션 대체 (원본 교체)"):
        try:
            # replace base with full generated df
            if st.session_state.get("df_preprocessed") is not None:
                st.session_state["df_preprocessed"] = df_gen
            else:
                st.session_state["df"] = df_gen
            st.success("세션 데이터가 생성된 전체 데이터로 교체되었습니다.")
        except Exception as e:
            st.error(f"세션 교체 실패: {e}")

    if st.button("생성 메타 저장 (artifacts)"):
        try:
            meta = st.session_state.get("feature_engineering_meta") or {}
            io_utils.save_json(meta, save_meta_name)
            st.success(f"생성 메타를 저장했습니다: {save_meta_name}")
        except Exception as e:
            st.error(f"생성 메타 저장 실패: {e}")

    # allow download of generated dataset
    csv_bytes = df_gen.head(1000).to_csv(index=False).encode("utf-8")
    st.download_button("생성된 데이터 샘플 다운로드 (상위 1000행)", data=csv_bytes, file_name="generated_features_preview.csv", mime="text/csv")

st.markdown("---")

# 3) 중요도 기반 선택 (선택 실행)
st.subheader("중요도 기반 피처 선택 (선택 사항)")
if use_importance:
    if st.button("중요도 기반으로 상위 k개 피처 추천"):
        try:
            # need target column
            target_col = st.session_state.get("target_col")
            if target_col is None:
                st.warning("타깃 컬럼이 설정되어 있지 않습니다. Overview 페이지에서 타깃을 지정하세요.")
            else:
                # build a DF for importance calc: use current session df_preprocessed or df (with generated features if applied)
                df_for_imp = st.session_state.get("df_preprocessed") or st.session_state.get("df")
                if df_for_imp is None or target_col not in df_for_imp.columns:
                    st.error("중요도 계산을 위한 데이터가 준비되지 않았거나 타깃이 데이터에 없습니다.")
                else:
                    top_feats = fe_mod.select_features_by_importance(df_for_imp, target_col=target_col, model="RandomForest", top_k=int(top_k))
                    st.session_state["fe_top_features"] = top_feats
                    st.success(f"중요도 기반 추천 상위 {len(top_feats)}개 피처를 생성했습니다.")
                    st.write(top_feats)
                    if st.button("추천 피처만 세션 데이터로 유지 (원래 데이터에 적용)"):
                        try:
                            # keep only recommended features + target
                            df_keep = df_for_imp[top_feats].copy()
                            df_keep[target_col] = df_for_imp[target_col].values
                            if st.session_state.get("df_preprocessed") is not None:
                                st.session_state["df_preprocessed"] = df_keep
                            else:
                                st.session_state["df"] = df_keep
                            st.success("추천 피처로 데이터 교체 완료.")
                        except Exception as e:
                            st.error(f"추천 피처 적용 실패: {e}")
        except Exception as e:
            st.error(f"중요도 기반 추천 실패: {e}")
else:
    st.info("중요도 기반 선택을 사용하려면 사이드바에서 '중요도 기반 피처 선택 실행'을 체크하세요.")

st.markdown("---")
st.write("다음 단계: 모델 선택 페이지로 이동하여 베이스라인 모델 성능을 확인하세요.")
