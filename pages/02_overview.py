# -*- coding: utf-8 -*-
# pages/02_overview.py
"""
2단계 - 데이터 개요 & 컬럼 설정

- 로드된 데이터(df) 확인, 지문(fingerprint) 생성
- 타깃 컬럼 지정
- 컬럼 타입/드랍 수동 조정 후 df에 반영
- 데이터 분할 방식 추천 및 수동 선택 (random/stratified/time/group)
"""
import json
from typing import Dict, List

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GroupShuffleSplit

st.set_page_config(layout="wide")
st.title("2 - Overview (데이터 개요)")

# 세션 기본 키
for key in [
    "df_original",
    "df_dropped",
    "df_preprocessed",
    "df_features",
    "df",
    "fingerprint",
    "target_col",
    "train_idx",
    "test_idx",
    "split_meta",
    "split_strategy",
]:
    st.session_state.setdefault(key, None)

# ingestion 불러오기
try:
    from modules import ingestion
except Exception:
    import modules.ingestion as ingestion  # type: ignore
try:
    from modules import model_search as ms
except Exception:
    import modules.model_search as ms  # type: ignore


def _suggest_split_strategy(df: pd.DataFrame, target_col: str, problem_type: str) -> dict:
    """간단 통계 기반 분할 추천."""
    suggestion = {"strategy": "random", "reason": "기본값(무작위 분할)", "datetime_cols": [], "group_candidates": []}
    # datetime 컬럼 확인
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    suggestion["datetime_cols"] = dt_cols
    # 그룹 후보: 전체 행 대비 2~50% 사이 고유값을 가지는 컬럼
    group_candidates = []
    for c in df.columns:
        if c == target_col:
            continue
        try:
            nunique = int(df[c].nunique(dropna=True))
        except Exception:
            continue
        ratio = nunique / max(1, len(df))
        if 2 <= nunique <= max(2, int(len(df) * 0.5)) and 0.02 <= ratio <= 0.5:
            group_candidates.append(c)
    suggestion["group_candidates"] = group_candidates

    # 문제 유형 판단
    task = None
    try:
        task = ms.detect_task_type(df, target_col=target_col, forced_task=problem_type)
    except Exception:
        task = problem_type

    # 1) 시계열 우선
    if dt_cols:
        suggestion["strategy"] = "time"
        suggestion["reason"] = f"datetime 컬럼 발견: {dt_cols[0]}"
        return suggestion
    # 2) 그룹 우선
    if group_candidates:
        suggestion["strategy"] = "group"
        suggestion["reason"] = f"그룹 후보 컬럼 발견: {group_candidates[0]}"
        return suggestion
    # 3) 분류 + 불균형이면 stratified
    if task == "classification" and target_col in df.columns:
        try:
            counts = df[target_col].value_counts(dropna=True)
            if not counts.empty:
                maj_ratio = counts.max() / counts.sum()
                if maj_ratio >= 0.8 and counts.size > 1:
                    suggestion["strategy"] = "stratified"
                    suggestion["reason"] = f"불균형 감지(최대 클래스 비율 {maj_ratio:.2f})"
                    return suggestion
        except Exception:
            pass
    # 4) 기본 무작위
    return suggestion


def _infer_column_roles(df: pd.DataFrame) -> Dict[str, str]:
    roles = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            roles[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[c]):
            roles[c] = "numeric" if df[c].nunique(dropna=True) > 20 else "categorical"
        else:
            roles[c] = "categorical"
    return roles


# 데이터 확인
if st.session_state["df_original"] is None:
    st.warning("1) Upload 페이지에서 CSV를 로드하거나 샘플을 로드해 주세요.")
    st.stop()

# 작업 대상: dropped 단계 (없으면 원본으로 초기화)
if st.session_state["df_dropped"] is None:
    st.session_state["df_dropped"] = st.session_state["df_original"].copy()

df: pd.DataFrame = st.session_state["df_dropped"]
st.write(f"현재 데이터(dropped 단계): {df.shape[0]} 행 × {df.shape[1]} 열")

# Fingerprint
with st.expander("Fingerprint 보기/생성"):
    if st.session_state.get("fingerprint") is not None:
        st.json(st.session_state["fingerprint"])
    if st.button("Fingerprint 생성"):
        try:
            fp = ingestion.generate_fingerprint(df)
            st.session_state["fingerprint"] = fp
            st.success("Fingerprint 생성 완료")
        except Exception as e:
            st.error(f"Fingerprint 생성 실패: {e}")

# 컬럼 메타 테이블
inferred = _infer_column_roles(df)
meta_rows: List[Dict[str, object]] = []
for c in df.columns:
    meta_rows.append(
        {
            "column": c,
            "dtype": str(df[c].dtype),
            "n_unique": int(df[c].nunique(dropna=True)),
            "n_missing": int(df[c].isnull().sum()),
            "inferred_role": inferred.get(c, "unknown"),
        }
    )
meta_df = pd.DataFrame(meta_rows)
st.subheader("컬럼 메타")
st.dataframe(meta_df, use_container_width=True)

st.markdown("---")
col_left, col_right = st.columns([2, 1])

# 타깃 선택
with col_left:
    st.subheader("타깃 컬럼 지정")
    cols = list(df.columns)
    current_target = st.session_state.get("target_col")
    default_index = cols.index(current_target) + 1 if current_target in cols else 0
    sel_target = st.selectbox("타깃 컬럼", options=["(미선택)"] + cols, index=default_index)
    st.session_state["target_col"] = None if sel_target == "(미선택)" else sel_target
    if st.session_state.get("target_col"):
        st.success(f"타깃 컬럼: `{st.session_state['target_col']}`")

    st.markdown("### 데이터 분할 설정")
    with st.expander("train/test 분할 생성·재생성", expanded=False):
        split_meta_default = {"test_size": 0.2, "random_state": 42, "stratify": False}
        split_meta_saved = st.session_state.get("split_meta") or split_meta_default
        suggestion = _suggest_split_strategy(df, st.session_state.get("target_col"), st.session_state.get("problem_type"))
        st.caption(f"추천 분할: {suggestion['strategy']} ({suggestion['reason']})")

        strategy_options = ["auto", "random", "stratified", "time", "group"]
        strategy_choice = st.selectbox(
            "분할 전략",
            options=strategy_options,
            index=0,
        )

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            test_size = st.number_input(
                "test 비율",
                min_value=0.05,
                max_value=0.5,
                value=float(split_meta_saved.get("test_size", 0.2)),
                step=0.05,
            )
        with col_s2:
            random_state = st.number_input(
                "random_state",
                min_value=0,
                max_value=10000,
                value=int(split_meta_saved.get("random_state", 42)),
                step=1,
            )
        with col_s3:
            can_stratify = bool(st.session_state.get("target_col"))
            stratify_flag = st.checkbox(
                "stratify target (classification)",
                value=bool(split_meta_saved.get("stratify", False) and can_stratify),
                disabled=not can_stratify,
            )

        time_col = None
        group_col = None
        if strategy_choice in ("time", "auto") and suggestion.get("datetime_cols"):
            time_col = st.selectbox("시계열 정렬 컬럼", options=suggestion["datetime_cols"], index=0)
        elif strategy_choice == "time" and not suggestion.get("datetime_cols"):
            st.warning("시계열 분할을 위해 datetime 컬럼이 필요합니다.")

        if strategy_choice in ("group", "auto") and suggestion.get("group_candidates"):
            group_col = st.selectbox("그룹 분리 기준 컬럼", options=suggestion["group_candidates"], index=0)
        elif strategy_choice == "group" and not suggestion.get("group_candidates"):
            st.warning("그룹 분할을 위해 반복되는 그룹 컬럼이 필요합니다.")

        if st.button("분할 생성/재생성"):
            try:
                chosen = suggestion["strategy"] if strategy_choice == "auto" else strategy_choice
                idx = list(range(len(df)))
                train_idx, test_idx = None, None

                if chosen == "time":
                    if not time_col:
                        raise ValueError("시계열 컬럼을 선택하세요.")
                    s = pd.to_datetime(df[time_col], errors="coerce")
                    df_sorted = df.copy()
                    df_sorted["__order__"] = s
                    df_sorted = df_sorted.sort_values("__order__")
                    cutoff = int(len(df_sorted) * (1 - float(test_size)))
                    train_idx = df_sorted.index[:cutoff].tolist()
                    test_idx = df_sorted.index[cutoff:].tolist()
                elif chosen == "group":
                    if not group_col:
                        raise ValueError("그룹 컬럼을 선택하세요.")
                    splitter = GroupShuffleSplit(
                        n_splits=1,
                        test_size=float(test_size),
                        random_state=int(random_state),
                    )
                    groups = df[group_col].values
                    split = next(splitter.split(idx, groups=groups))
                    train_idx = [idx[i] for i in split[0]]
                    test_idx = [idx[i] for i in split[1]]
                else:
                    stratify_vals = df[st.session_state["target_col"]] if (stratify_flag and st.session_state.get("target_col")) else None
                    train_idx, test_idx = train_test_split(
                        idx,
                        test_size=float(test_size),
                        random_state=int(random_state),
                        stratify=stratify_vals if chosen == "stratified" else None,
                    )

                st.session_state["train_idx"] = train_idx
                st.session_state["test_idx"] = test_idx
                st.session_state["split_meta"] = {
                    "test_size": float(test_size),
                    "random_state": int(random_state),
                    "stratify": bool(stratify_flag if chosen != "time" else False),
                }
                st.session_state["split_strategy"] = {
                    "type": chosen,
                    "time_col": time_col,
                    "group_col": group_col,
                    "reason": suggestion.get("reason"),
                }
                st.success(
                    f"분할 완료 | 전략: {chosen} | train: {len(train_idx)} | test: {len(test_idx)} | 비율 {test_size:.2f} | stratify: {st.session_state['split_meta']['stratify']}"
                )
            except Exception as e:
                st.error(f"분할 생성 실패: {e}")

    if isinstance(st.session_state.get("train_idx"), list) and isinstance(st.session_state.get("test_idx"), list):
        meta = st.session_state.get("split_meta") or {}
        strategy = (st.session_state.get("split_strategy") or {}).get("type", "random")
        st.caption(
            f"현재 분할 — train: {len(st.session_state['train_idx'])}, "
            f"test: {len(st.session_state['test_idx'])}, "
            f"test 비율: {meta.get('test_size', 0.2):.2f}, "
            f"stratify: {meta.get('stratify', False)}, "
            f"random_state: {meta.get('random_state', 42)}, "
            f"전략: {strategy}"
        )

    # 문제 유형 자동 추천 + 수동 선택
    if st.session_state.get("target_col"):
        try:
            suggested_task = ms.detect_task_type(df, target_col=st.session_state["target_col"])
        except Exception:
            suggested_task = None
        current_task = st.session_state.get("problem_type")
        options_task = [
            f"자동 감지 (추천: {suggested_task or '없음'})",
            "분류 (classification)",
            "회귀 (regression)",
        ]
        if current_task == "classification":
            default_task_idx = 1
        elif current_task == "regression":
            default_task_idx = 2
        else:
            default_task_idx = 0
        chosen = st.radio("문제 유형 설정", options=options_task, index=default_task_idx, horizontal=True)
        if chosen.startswith("자동"):
            st.session_state["problem_type"] = suggested_task
        elif "분류" in chosen:
            st.session_state["problem_type"] = "classification"
        else:
            st.session_state["problem_type"] = "regression"
        st.caption(f"현재 문제 유형: {st.session_state.get('problem_type') or '자동 감지 결과를 사용합니다.'}")
    else:
        st.session_state["problem_type"] = None
        st.info("타깃 컬럼을 먼저 선택하면 문제 유형을 자동 추천합니다.")

    st.markdown("### 컬럼 타입/드랍 수동 조정")
    role_options = ["numeric", "categorical", "datetime", "drop"]
    with st.form("col_override_form"):
        overrides: Dict[str, str] = {}
        for r in meta_rows:
            colname = r["column"]
            default_role = r["inferred_role"] if r["inferred_role"] in role_options else "categorical"
            overrides[colname] = st.selectbox(
                f"{colname} (dtype: {r['dtype']}, unique: {r['n_unique']})",
                options=role_options,
                index=role_options.index(default_role),
                key=f"role_{colname}",
            )
        submitted = st.form_submit_button("변경 적용")
        if submitted:
            df_new = df.copy()
            dropped = []
            casted = []
            for colname, role in overrides.items():
                try:
                    if role == "drop":
                        if colname in df_new.columns:
                            df_new.drop(columns=[colname], inplace=True)
                            dropped.append(colname)
                    elif role == "datetime":
                        df_new[colname] = pd.to_datetime(df_new[colname], errors="coerce")
                        casted.append((colname, "datetime"))
                    elif role == "numeric":
                        df_new[colname] = pd.to_numeric(df_new[colname], errors="coerce")
                        casted.append((colname, "numeric"))
                    elif role == "categorical":
                        df_new[colname] = df_new[colname].astype("object")
                        casted.append((colname, "categorical"))
                except Exception as e:
                    st.warning(f"{colname} 처리 중 오류: {e}")
            st.session_state["df"] = df_new
            try:
                st.session_state["fingerprint"] = ingestion.generate_fingerprint(df_new)
            except Exception:
                st.session_state["fingerprint"] = None
            st.session_state["df_dropped"] = df_new.copy()
            st.session_state["df_preprocessed"] = df_new.copy()
            st.session_state["df_features"] = df_new.copy()
            st.session_state["df"] = st.session_state.get("df_original")  # 유지용
            st.success(f"적용 완료. 드랍: {dropped}, 타입 변경: {casted}")
            df = df_new

with col_right:
    st.subheader("간단 컬럼 삭제")
    drop_sel = st.multiselect("삭제할 컬럼 선택", options=list(df.columns), default=[])
    if st.button("선택 컬럼 삭제"):
        if drop_sel:
            df_new = df.copy()
            df_new.drop(columns=drop_sel, inplace=True)
            st.session_state["df_dropped"] = df_new.copy()
            st.session_state["df_preprocessed"] = df_new.copy()
            st.session_state["df_features"] = df_new.copy()
            st.session_state["df"] = st.session_state.get("df_original")
            try:
                st.session_state["fingerprint"] = ingestion.generate_fingerprint(df_new)
            except Exception:
                st.session_state["fingerprint"] = None
            st.success(f"컬럼 삭제 완료: {drop_sel}")
            df = df_new
        else:
            st.info("삭제할 컬럼을 선택하세요.")

st.markdown("---")
st.subheader("미리보기")
st.dataframe(df.head(50))

st.markdown("---")
st.write("### Fingerprint 다운로드")
if st.button("Fingerprint 다운로드 (JSON)"):
    fp = st.session_state.get("fingerprint") or {}
    b = json.dumps(fp, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("fingerprint.json 다운로드", data=b, file_name="fingerprint.json", mime="application/json")

st.info("다음 단계: EDA/전처리/특징공학으로 이동하세요.")
