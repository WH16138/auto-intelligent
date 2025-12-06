"""
5단계 - 특징 공학

안내:
- 타깃 컬럼은 자동 생성 입력에서 제외(누수 방지).
- 생성 입력은 항상 전처리본(df_preprocessed) 선택합니다.
- 생성 결과는 df_features에 저장하며, 필요 시 생략/확정 버튼으로 이전 단계 데이터를 그대로 사용할 수 있습니다. (그냥 페이지 스킵도 가능)
"""
import streamlit as st
from typing import List
import pandas as pd

st.set_page_config(layout="wide")
st.title("5 - 특징 공학")
st.info("기본값으로 빠르게 실행해보고, 필요한 경우에만 생성 규칙/적용 범위를 조정하세요. 타깃은 자동으로 생성 대상에서 제외됩니다.")

# Session defaults
for key in [
    "df_original",
    "df_dropped",
    "df_preprocessed",
    "df_features",
    "df_features_train",
    "df_features_test",
    "feature_engineering_meta",
    "target_col",
]:
    st.session_state.setdefault(key, None)

try:
    from modules import feature_engineering as fe_mod
except Exception:
    import modules.feature_engineering as fe_mod

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils

def _apply_meta_to_full(df_full: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Apply generated feature rules (from train fit) to full dataset."""
    df_out = df_full.copy()
    methods = meta.get("method", {})
    new_feats = meta.get("new_features", [])
    for f in new_feats:
        method = methods.get(f)
        try:
            if method == "ratio" and "__div__" in f:
                a, b = f.split("__div__")
                df_out[f] = fe_mod._safe_divide(df_out[a], df_out[b])
            elif method == "diff" and "__minus__" in f:
                a, b = f.split("__minus__")
                df_out[f] = pd.to_numeric(df_out[a], errors="coerce").fillna(0) - pd.to_numeric(df_out[b], errors="coerce").fillna(0)
            elif method == "sum" and "__plus__" in f:
                a, b = f.split("__plus__")
                df_out[f] = pd.to_numeric(df_out[a], errors="coerce").fillna(0) + pd.to_numeric(df_out[b], errors="coerce").fillna(0)
            elif method == "log1p" and f.endswith("__log1p"):
                base = f[: -len("__log1p")]
                df_out[f] = fe_mod._safe_log1p(df_out[base])
            elif method == "datetime_extract" and "__" in f:
                base, attr = f.split("__", 1)
                s = pd.to_datetime(df_out[base], errors="coerce")
                if attr == "weekday":
                    df_out[f] = s.dt.weekday.fillna(-1).astype(int)
                elif attr == "hour":
                    df_out[f] = s.dt.hour.fillna(-1).astype(int)
                else:
                    df_out[f] = getattr(s.dt, attr).fillna(-1).astype(int)
            # if method is unknown, skip.
        except Exception:
            continue
    return df_out


def get_source_df():
    return st.session_state.get("df_preprocessed")


def get_current_features_df():
    if st.session_state.get("df_features") is not None:
        return st.session_state["df_features"], "df_features"
    return get_source_df()

source_df = st.session_state["df_preprocessed"]
base_df = get_current_features_df()
target_col = st.session_state.get("target_col")
train_idx = st.session_state.get("train_idx")
test_idx = st.session_state.get("test_idx")

if source_df is None:
    st.warning("Upload/Preprocessing 이후 데이터를 준비해 주세요.")
    st.stop()

st.write(
    f"생성 입력 데이터: {source_df.shape[0]} 행 × {source_df.shape[1]} 열"
)
if target_col and target_col in source_df.columns:
    st.caption(f"타깃 컬럼은 생성 대상에서 제외합니다: `{target_col}`")

# 본문 세팅
st.markdown("### 자동 생성 설정")
st.caption("기본 규칙(비율/차이/로그/날짜 추출)으로 최대 10개를 생성합니다. 먼저 실행해보고 결과를 골라서 적용하세요.")
col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    max_new = st.number_input("최대 생성 개수", min_value=1, max_value=500, value=10, step=1)
    preview_rows = st.number_input("미리보기 행수", min_value=5, max_value=200, value=50, step=5)
    st.caption("과도한 생성은 과적합·지연을 유발할 수 있어 500까지 제한합니다.")
with col_cfg2:
    methods: List[str] = st.multiselect(
        "사용할 변환",
        options=["ratio", "diff", "sum", "log", "datetime"],
        default=["ratio", "diff", "log", "datetime"],
    )
    datetime_extract = st.checkbox("datetime에서 연/월/일/시간 추출", value=True)
with col_cfg3:
    st.markdown("생성·추천 안내")
    st.caption("- 생성 규칙은 train 기준으로 학습 후 전체에 반영됩니다.\n- 중요도 추천은 아래 섹션에서 바로 실행할 수 있습니다.")

st.info("타깃 컬럼은 자동 생성 입력에서 제외됩니다. 필요 시 상단에서 타깃을 먼저 지정하세요.")

st.markdown("---")

# 1) 자동 특징 생성 (타깃 제외)
st.subheader("자동 특징 생성 (규칙 기반)")
col_gen_a, col_gen_b = st.columns([2, 1])
with col_gen_a:
    if st.button("자동 생성 실행"):
        try:
            feature_input_full = source_df.drop(columns=[target_col], errors="ignore") if target_col else source_df
            # fit on train subset if available
            if isinstance(train_idx, list) and len(train_idx) > 0:
                feature_input_train = feature_input_full.iloc[train_idx].copy()
            else:
                feature_input_train = feature_input_full

            df_extended_train, meta = fe_mod.auto_generate_features(
                feature_input_train,
                max_new=int(max_new),
                methods=methods,
                datetime_extract=bool(datetime_extract),
            )
            # apply learned feature rules to full data
            df_extended_full = _apply_meta_to_full(feature_input_full, meta)
            st.session_state["fe_generated_df"] = df_extended_full
            st.session_state["feature_engineering_meta"] = meta
            st.success(f"{len(meta.get('new_features', []))}개 특징을 생성했습니다.")
        except Exception as e:
            st.error(f"생성 실패: {e}")

with col_gen_b:
    if st.session_state.get("feature_engineering_meta"):
        st.markdown("**최근 생성 메타**")
        with st.expander("메타 보기 (길면 접어서 확인)", expanded=False):
            st.json(st.session_state["feature_engineering_meta"])
    else:
        st.info("아직 생성된 특징이 없습니다. '자동 생성 실행'을 눌러주세요.")

st.markdown("---")

# 2) 미리보기 및 적용
st.subheader("생성 특징 미리보기 및 적용")
if st.session_state.get("fe_generated_df") is None:
    st.info("먼저 자동 생성 기능을 실행하세요.")
else:
    df_gen = st.session_state["fe_generated_df"]
    meta = st.session_state.get("feature_engineering_meta") or {}
    new_feats = meta.get("new_features", [])

    st.markdown(f"생성된 신규 특징 수: **{len(new_feats)}**")
    preview_df = source_df.copy()
    for f in new_feats:
        if f in df_gen.columns:
            preview_df[f] = df_gen[f].values
    st.caption("미리보기 (입력 + 생성 특징)")
    st.dataframe(preview_df.head(int(preview_rows)))

    keep = st.multiselect("적용할 생성 특징 선택", options=new_feats, default=new_feats, help="과하게 생성된 특징을 줄일 때 선택하세요.")
    if st.button("선택 특징만 반영"):
        try:
            df_base_copy = source_df.copy()
            for f in keep:
                if f in df_gen.columns:
                    df_base_copy[f] = df_gen[f].values
            st.session_state["df_features"] = df_base_copy
            if isinstance(train_idx, list) and len(train_idx) > 0:
                st.session_state["df_features_train"] = df_base_copy.iloc[train_idx].copy()
            if isinstance(test_idx, list) and len(test_idx) > 0:
                st.session_state["df_features_test"] = df_base_copy.iloc[test_idx].copy()
            st.success(f"선택한 {len(keep)}개 특징을 반영했습니다.")
        except Exception as e:
            st.error(f"적용 실패: {e}")

    if st.button("전체 생성본으로 교체 (타깃 유지)"):
        try:
            df_with_target = source_df.copy()
            for f in new_feats:
                if f in df_gen.columns:
                    df_with_target[f] = df_gen[f].values
            st.session_state["df_features"] = df_with_target
            if isinstance(train_idx, list) and len(train_idx) > 0:
                st.session_state["df_features_train"] = df_with_target.iloc[train_idx].copy()
            if isinstance(test_idx, list) and len(test_idx) > 0:
                st.session_state["df_features_test"] = df_with_target.iloc[test_idx].copy()
            st.success("현재 데이터프레임을 생성본으로 교체했습니다.")
        except Exception as e:
            st.error(f"교체 실패: {e}")

    csv_bytes = preview_df.head(200).to_csv(index=False).encode("utf-8")
    st.download_button(
        "미리보기 CSV 다운로드 (상위 200행)",
        data=csv_bytes,
        file_name="generated_features_preview.csv",
        mime="text/csv",
    )

st.markdown("---")

# 3) 중요도 기반 선택
st.subheader("중요도 기반 선택")
st.caption("타깃이 있을 때 RandomForest 기반 상위 k개를 추천합니다. 실행과 적용 버튼을 한곳에 모았습니다.")
col_imp_main, col_imp_side = st.columns([2, 1])
with col_imp_main:
    top_k = st.number_input("상위 k개 추천", min_value=1, max_value=500, value=20, step=1)
    if st.button("중요도 기준 추천 실행"):
        try:
            target_col = st.session_state.get("target_col")
            if target_col is None:
                st.warning("타깃 컬럼을 먼저 지정하세요. Overview 페이지에서 설정할 수 있습니다.")
            else:
                df_for_imp_full = next(
                    (
                        x
                        for x in [
                            st.session_state.get("df_features"),
                            st.session_state.get("df_preprocessed"),
                            st.session_state.get("df_dropped"),
                            st.session_state.get("df_original"),
                        ]
                        if x is not None
                    ),
                    None,
                )
                if df_for_imp_full is None or target_col not in df_for_imp_full.columns:
                    st.error("타깃 컬럼이 포함된 데이터가 없습니다.")
                else:
                    if isinstance(train_idx, list) and len(train_idx) > 0:
                        df_for_imp = df_for_imp_full.iloc[train_idx].copy()
                    else:
                        df_for_imp = df_for_imp_full
                    top_feats = fe_mod.select_features_by_importance(
                        df_for_imp,
                        target_col=target_col,
                        model="RandomForest",
                        top_k=int(top_k),
                    )
                    st.session_state["fe_top_features"] = top_feats
                    st.success(f"중요도 기준 {len(top_feats)}개를 추천했습니다.")
                    st.write(top_feats)
        except Exception as e:
            st.error(f"중요도 계산 실패: {e}")

with col_imp_side:
    top_feats_saved: list = st.session_state.get("fe_top_features") or []
    if top_feats_saved:
        st.caption(f"최근 추천 {len(top_feats_saved)}개 적용")
        if st.button("추천 특징만 남기기 (타깃 포함)"):
            try:
                target_col = st.session_state.get("target_col")
                df_for_imp_full = next(
                    (
                        x
                        for x in [
                            st.session_state.get("df_features"),
                            st.session_state.get("df_preprocessed"),
                            st.session_state.get("df_dropped"),
                            st.session_state.get("df_original"),
                        ]
                        if x is not None
                    ),
                    None,
                )
                if target_col is None or df_for_imp_full is None or target_col not in df_for_imp_full.columns:
                    st.error("타깃 컬럼이 포함된 데이터가 없습니다.")
                else:
                    available = [c for c in top_feats_saved if c in df_for_imp_full.columns]
                    missing = [c for c in top_feats_saved if c not in df_for_imp_full.columns]
                    if not available:
                        st.error("추천된 특징이 현재 데이터에 없습니다.")
                    else:
                        df_keep_full = df_for_imp_full[available].copy()
                        df_keep_full[target_col] = df_for_imp_full[target_col].values
                        st.session_state["df_features"] = df_keep_full
                        if isinstance(train_idx, list) and len(train_idx) > 0:
                            st.session_state["df_features_train"] = df_keep_full.iloc[train_idx].copy()
                        if isinstance(test_idx, list) and len(test_idx) > 0:
                            st.session_state["df_features_test"] = df_keep_full.iloc[test_idx].copy()
                        if missing:
                            st.warning(f"일부 추천 특징은 현재 데이터에 없어 제외했습니다: {missing}")
                        st.success(f"추천 특징 {len(available)}개만 유지했습니다.")
            except Exception as e:
                st.error(f"추천 적용 실패: {e}")
    else:
        st.info("추천을 실행하면 여기에서 바로 적용할 수 있습니다.")

st.markdown("---")
if st.button("특징공학 생략/현재 데이터로 확정"):
    st.session_state["df_features"] = source_df.copy()
    if isinstance(train_idx, list) and len(train_idx) > 0:
        st.session_state["df_features_train"] = source_df.iloc[train_idx].copy()
    if isinstance(test_idx, list) and len(test_idx) > 0:
        st.session_state["df_features_test"] = source_df.iloc[test_idx].copy()
    st.success("현재 단계 데이터를 그대로 다음 단계에 사용합니다.")

st.markdown("---")
st.info("다음 단계: Model Selection 페이지로 이동하여 베이스라인 모델을 비교해보세요.")
