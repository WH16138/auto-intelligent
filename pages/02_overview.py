# pages/02_overview.py
"""
Overview page for AutoML-Edu Streamlit app.

기능:
- 현재 업로드된 DataFrame의 컬럼 메타(타입, 유일값 수, 결측) 표시
- 타깃(target) 컬럼 선택/확정
- 컬럼 타입 자동 추정 결과를 보고 수동으로 override 가능 (numeric/categorical/datetime/drop)
- 선택한 변경사항 적용 → st.session_state['df'], ['fingerprint'] 업데이트
- 변경 전/후 간단 미리보기 제공
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

st.set_page_config(layout="wide")
st.title("2 - Overview (데이터 개요 및 컬럼 관리)")

# Ensure session keys
if "df" not in st.session_state:
    st.session_state["df"] = None
if "fingerprint" not in st.session_state:
    st.session_state["fingerprint"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = None

# import ingestion for fingerprint regeneration
try:
    from modules import ingestion
except Exception:
    import modules.ingestion as ingestion  # fallback

# helper: infer simple column categories (reuse fingerprint if present)
def _infer_column_roles(df: pd.DataFrame) -> Dict[str, str]:
    roles = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            roles[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[c]):
            # numeric but low cardinality -> maybe categorical
            nunique = int(df[c].nunique(dropna=True))
            if nunique <= 20:
                roles[c] = "categorical"
            else:
                roles[c] = "numeric"
        else:
            roles[c] = "categorical"
    return roles

# UI: if no data, show message
if st.session_state["df"] is None:
    st.warning("먼저 1) Upload 페이지에서 CSV를 업로드하거나 샘플 데이터를 로드하세요.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
st.write(f"데이터: {df.shape[0]} 행 × {df.shape[1]} 열")

# show current fingerprint if exists
with st.expander("현재 데이터 지문 (Fingerprint) 보기"):
    if st.session_state.get("fingerprint") is not None:
        st.json(st.session_state["fingerprint"])
    else:
        st.info("지문이 아직 생성되지 않았습니다. '지문 생성' 버튼을 눌러 생성하세요.")
        if st.button("지문 생성 (이 페이지에서)"):
            try:
                fp = ingestion.generate_fingerprint(df)
                st.session_state["fingerprint"] = fp
                st.success("지문 생성 완료")
            except Exception as e:
                st.error(f"지문 생성 실패: {e}")

# build column metadata table
cols = list(df.columns)
inferred = _infer_column_roles(df)
meta_rows = []
for c in cols:
    dtype = str(df[c].dtype)
    n_missing = int(df[c].isnull().sum())
    n_unique = int(df[c].nunique(dropna=True))
    role_guess = inferred.get(c, "unknown")
    meta_rows.append({"column": c, "dtype": dtype, "n_unique": n_unique, "n_missing": n_missing, "inferred_role": role_guess})

meta_df = pd.DataFrame(meta_rows)

st.subheader("컬럼 메타 정보")
st.dataframe(meta_df, use_container_width=True)

# Controls: target selection, manual role override, drop columns
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("타깃(target) 설정")
    current_target = st.session_state.get("target_col")
    options = ["(없음)"] + cols
    # determine default index safely
    if current_target in cols:
        default_index = cols.index(current_target) + 1
    else:
        # try fingerprint suggestion
        fp_suggest = None
        if st.session_state.get("fingerprint"):
            fp_suggest = st.session_state["fingerprint"].get("target_column")
        if fp_suggest in cols:
            default_index = cols.index(fp_suggest) + 1
        else:
            default_index = 0
    sel = st.selectbox("타깃 컬럼 선택", options=options, index=default_index, key="overview_target_select")
    if sel == "(없음)":
        st.session_state["target_col"] = None
    else:
        st.session_state["target_col"] = sel
    if st.session_state.get("target_col"):
        st.success(f"선택된 타깃: `{st.session_state['target_col']}`")

    st.markdown("### 컬럼 타입 수동 보정")
    st.markdown("각 컬럼에 대해 자동 추정된 역할을 검토하고 필요시 수동으로 변경하세요. 변경사항은 아래 '변경 적용' 버튼으로 반영됩니다.")
    # create a form for batch edits
    with st.form("col_override_form"):
        overrides: Dict[str, str] = {}
        role_options = ["numeric", "categorical", "datetime", "drop"]
        for r in meta_rows:
            c = r["column"]
            inferred_role = r["inferred_role"]
            default = inferred_role if inferred_role in role_options else "categorical"
            sel_role = st.selectbox(f"{c} (dtype: {r['dtype']}, unique: {r['n_unique']})", options=role_options, index=role_options.index(default), key=f"role_{c}")
            overrides[c] = sel_role
        submitted = st.form_submit_button("변경 적용")
        if submitted:
            # apply overrides: drop columns, cast types where feasible
            df_new = df.copy()
            dropped = []
            casted = []
            for col, role in overrides.items():
                try:
                    if role == "drop":
                        if col in df_new.columns:
                            df_new.drop(columns=[col], inplace=True)
                            dropped.append(col)
                    elif role == "datetime":
                        # try to parse as datetime
                        try:
                            df_new[col] = pd.to_datetime(df_new[col], errors="coerce")
                            casted.append((col, "datetime"))
                        except Exception:
                            # leave as is
                            pass
                    elif role == "numeric":
                        try:
                            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
                            casted.append((col, "numeric"))
                        except Exception:
                            pass
                    elif role == "categorical":
                        # leave as object / category (no automatic encoding here)
                        try:
                            df_new[col] = df_new[col].astype("object")
                            casted.append((col, "categorical(object)"))
                        except Exception:
                            pass
                except Exception as e:
                    st.warning(f"컬럼 처리 중 문제가 발생했습니다: {col} — {e}")
            # save changes to session and regenerate fingerprint
            st.session_state["df"] = df_new
            try:
                fp_new = ingestion.generate_fingerprint(df_new)
                st.session_state["fingerprint"] = fp_new
            except Exception:
                st.session_state["fingerprint"] = None
            st.success(f"적용 완료. 드롭된 컬럼: {dropped} / 타입 변환 시도된 컬럼: {casted}")

with col_right:
    st.subheader("컬럼 제거(빠른 선택)")
    drop_sel = st.multiselect("모델에서 사용하지 않을 컬럼 선택(여러개 선택 가능)", options=cols, default=[])
    if st.button("선택한 컬럼 제거"):
        if drop_sel:
            df2 = df.copy()
            df2.drop(columns=drop_sel, inplace=True)
            st.session_state["df"] = df2
            try:
                st.session_state["fingerprint"] = ingestion.generate_fingerprint(df2)
            except Exception:
                st.session_state["fingerprint"] = None
            st.success(f"컬럼 {drop_sel} 제거 완료.")
        else:
            st.info("먼저 제거할 컬럼을 선택하세요.")

st.markdown("---")
st.subheader("변경 전/후 미리보기")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("현재(변경 전) 데이터 미리보기")
    st.dataframe(df.head(50))
with col_b:
    st.caption("세션에 반영된 최신 데이터 (변경사항이 적용되었을 경우)")
    st.dataframe(st.session_state["df"].head(50))

st.markdown("---")
st.write("### 메타/설정 저장 및 다운로드")
if st.button("현재 지문 다운로드 (JSON)"):
    import json, io
    fp = st.session_state.get("fingerprint") or {}
    b = json.dumps(fp, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("지문 JSON 저장", data=b, file_name="fingerprint.json", mime="application/json")

st.info("다음 단계: EDA(시각화) 페이지로 이동하여 분포·상관관계·결측 패턴을 확인하세요.")
