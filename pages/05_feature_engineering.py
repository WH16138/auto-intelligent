# pages/05_feature_engineering.py
"""
5단계 - 특징 공학

안내:
- 타깃 컬럼은 자동 생성 입력에서 제외합니다(데이터 누수 방지).
- 생성된 특징을 미리보고 선택 적용/교체할 수 있습니다.
- 옵션: 중요도 기반 특징 선택(RandomForest).
"""
import streamlit as st
from typing import List, Tuple

st.set_page_config(layout="wide")
st.title("5 - 특징 공학")

# Session defaults
st.session_state.setdefault("df", None)
st.session_state.setdefault("df_preprocessed", None)
st.session_state.setdefault("feature_engineering_meta", None)
st.session_state.setdefault("df_features", None)

# Imports (fallbacks)
try:
    from modules import feature_engineering as fe_mod
except Exception:
    import modules.feature_engineering as fe_mod  # type: ignore

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore


def get_base_df() -> Tuple[object, bool]:
    """Prefer preprocessed dataframe when available."""
    df_feat = st.session_state.get("df_features")
    if df_feat is not None:
        return df_feat, st.session_state.get("df_preprocessed") is not None
    df_pre = st.session_state.get("df_preprocessed")
    if df_pre is not None:
        return df_pre, True
    return st.session_state.get("df"), False


base_df, used_pre = get_base_df()
target_col = st.session_state.get("target_col")

if base_df is None:
    st.warning("먼저 Upload 또는 Preprocessing 페이지에서 데이터를 준비해 주세요.")
    st.stop()

st.write(
    f"활성 데이터프레임: {base_df.shape[0]} 행 × {base_df.shape[1]} 열 "
    f"(전처리본 사용: {'예' if used_pre else '아니요'})"
)
if target_col and target_col in base_df.columns:
    st.caption(f"타깃 컬럼은 생성 대상에서 제외합니다: `{target_col}`")

# 주요 설정 (화면 내)
st.markdown("### 자동 생성 설정")
col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    max_new = st.number_input("최대 생성 개수", min_value=1, max_value=200, value=10, step=1)
    preview_rows = st.number_input("미리보기 행수", min_value=5, max_value=500, value=50, step=5)
with col_cfg2:
    methods: List[str] = st.multiselect(
        "사용할 변환",
        options=["ratio", "diff", "sum", "log", "datetime"],
        default=["ratio", "diff", "log", "datetime"],
    )
    datetime_extract = st.checkbox("datetime에서 연/월/일/시간 추출", value=True)
with col_cfg3:
    save_meta_name = st.text_input("메타 저장 경로", value="artifacts/fe_meta.json")
    st.markdown("옵션: 중요도 기반 선택")
    use_importance = st.checkbox("중요도 기반 선택 실행", value=False)
    top_k = st.number_input("상위 k개", min_value=1, max_value=500, value=20, step=1)

st.info("타깃 컬럼은 자동 생성 입력에서 제외됩니다. 필요 시 상단에서 타깃을 먼저 지정하세요.")

st.markdown("---")

# 1) 자동 특징 생성 (타깃 제외)
st.subheader("자동 특징 생성 (규칙 기반)")
col_gen_a, col_gen_b = st.columns([2, 1])
with col_gen_a:
    if st.button("자동 생성 실행"):
        try:
            feature_input = (
                base_df.drop(columns=[target_col], errors="ignore") if target_col else base_df
            )
            df_extended, meta = fe_mod.auto_generate_features(
                feature_input,
                max_new=int(max_new),
                methods=methods,
                datetime_extract=bool(datetime_extract),
            )
            st.session_state["fe_generated_df"] = df_extended
            st.session_state["feature_engineering_meta"] = meta
            st.success(f"{len(meta.get('new_features', []))}개 특징을 생성했습니다.")
        except Exception as e:
            st.error(f"생성 실패: {e}")

with col_gen_b:
    if st.session_state.get("feature_engineering_meta"):
        st.markdown("**최근 생성 메타**")
        st.json(st.session_state["feature_engineering_meta"])
    else:
        st.info("아직 생성된 특징이 없습니다. '자동 생성 실행'을 눌러주세요.")

st.markdown("---")

# 2) 생성 특징 미리보기 및 적용 (타깃 유지)
st.subheader("생성 특징 미리보기 및 적용")
if st.session_state.get("fe_generated_df") is None:
    st.info("먼저 자동 생성 기능을 실행하세요.")
else:
    df_gen = st.session_state["fe_generated_df"]
    meta = st.session_state.get("feature_engineering_meta") or {}
    new_feats = meta.get("new_features", [])

    st.markdown(f"생성된 신규 특징 수: **{len(new_feats)}**")
    preview_df = base_df.copy()
    for f in new_feats:
        if f in df_gen.columns:
            preview_df[f] = df_gen[f].values
    st.caption("미리보기 (원본 + 생성 특징)")
    st.dataframe(preview_df.head(int(preview_rows)))

    keep = st.multiselect("적용할 생성 특징 선택", options=new_feats, default=new_feats)
    if st.button("선택 특징만 원본에 추가"):
        try:
            df_base_copy = base_df.copy()
            for f in keep:
                if f in df_gen.columns:
                    df_base_copy[f] = df_gen[f].values
            if used_pre:
                st.session_state["df_preprocessed"] = df_base_copy
            else:
                st.session_state["df"] = df_base_copy
            st.session_state["df_features"] = df_base_copy
            st.success(f"선택한 {len(keep)}개 특징을 반영했습니다.")
        except Exception as e:
            st.error(f"적용 실패: {e}")

    if st.button("전체 생성본으로 교체 (타깃 유지)"):
        try:
            df_with_target = base_df.copy()
            for f in new_feats:
                if f in df_gen.columns:
                    df_with_target[f] = df_gen[f].values
            if used_pre:
                st.session_state["df_preprocessed"] = df_with_target
            else:
                st.session_state["df"] = df_with_target
            st.session_state["df_features"] = df_with_target
            st.success("현재 데이터프레임을 생성본으로 교체했습니다.")
        except Exception as e:
            st.error(f"교체 실패: {e}")

    if st.button("생성 메타 저장 (artifacts)"):
        try:
            meta = st.session_state.get("feature_engineering_meta") or {}
            io_utils.save_json(meta, save_meta_name)
            st.success(f"메타 저장 완료: {save_meta_name}")
        except Exception as e:
            st.error(f"저장 실패: {e}")

    csv_bytes = preview_df.head(1000).to_csv(index=False).encode("utf-8")
    st.download_button(
        "미리보기 CSV 다운로드 (상위 1000행)",
        data=csv_bytes,
        file_name="generated_features_preview.csv",
        mime="text/csv",
    )

st.markdown("---")

# 3) 중요도 기반 선택 (옵션, 타깃 필요)
st.subheader("중요도 기반 선택 (옵션)")
if use_importance:
    if st.button("중요도 기준 상위 k개 추천"):
        try:
            target_col = st.session_state.get("target_col")
            if target_col is None:
                st.warning("타깃 컬럼을 먼저 지정하세요. Overview 페이지에서 설정할 수 있습니다.")
            else:
                df_for_imp, _ = get_base_df()
                if df_for_imp is None or target_col not in df_for_imp.columns:
                    st.error("타깃 컬럼이 포함된 데이터가 없습니다.")
                else:
                    top_feats = fe_mod.select_features_by_importance(
                        df_for_imp,
                        target_col=target_col,
                        model="RandomForest",
                        top_k=int(top_k),
                    )
                    st.session_state["fe_top_features"] = top_feats
                    st.success(f"중요도 기준 {len(top_feats)}개를 추천했습니다.")
                    st.write(top_feats)
                    if st.button("추천 특징만 남기기 (타깃 포함)"):
                        try:
                            df_keep = df_for_imp[top_feats].copy()
                            df_keep[target_col] = df_for_imp[target_col].values
                            if used_pre:
                                st.session_state["df_preprocessed"] = df_keep
                            else:
                                st.session_state["df"] = df_keep
                            st.success("추천 특징만 유지했습니다.")
                        except Exception as e:
                            st.error(f"추천 적용 실패: {e}")
        except Exception as e:
            st.error(f"중요도 계산 실패: {e}")
else:
    st.info("위 설정에서 '중요도 기반 선택 실행'을 켜면 동작합니다.")

st.markdown("---")
if st.button("특징공학 생략/현재 데이터로 확정"):
    st.session_state["df_features"] = base_df.copy()
    st.success("현재 데이터 상태를 그대로 다음 단계에 사용합니다.")

st.write("다음 단계: 모델 선택(베이스라인/HPO)으로 이동하세요.")
