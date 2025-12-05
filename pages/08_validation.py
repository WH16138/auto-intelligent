# pages/08_validation.py
"""
Step 8 - Validation

Supports evaluating:
- Trained model stored in session (trained_model + X_test/y_test if provided)
- Best baseline model from Model Selection
- A RandomForest retrained with HPO best_params (if available)

Target column is never used as a feature; rows with missing target are dropped before splitting.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import clone

try:
    from modules import explain as explain_mod
    _HAS_EXPLAIN = True
except Exception:
    explain_mod = None
    _HAS_EXPLAIN = False

try:
    from modules import model_search as ms
except Exception:
    import modules.model_search as ms  # type: ignore
try:
    from modules import preprocessing as prep
except Exception:
    prep = None  # type: ignore

st.set_page_config(layout="wide")
st.title("8 - Model Validation")
st.info("이전 단계에서 준비한 모델/데이터로 검증을 수행합니다. 기본 설정으로 바로 실행되며, 결과 표·그래프는 필요할 때만 펼쳐 볼 수 있도록 구성했습니다.")


# ------------ helpers ------------
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_regression_residuals(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Plot")
    fig.tight_layout()
    return fig


def get_active_df():
    return st.session_state.get("df_features")

def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    feature_names_hint: Optional[list] = None,
    preprocessor=None,
    forced_task: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, str, list]:
    """Return X_df, y, task, feature_names; applies preprocessor if provided."""
    df_clean = df.dropna(subset=[target_col]).copy()
    if df_clean.empty:
        raise ValueError("타깃 결측 제거 후 데이터가 비어 있습니다.")
    y = df_clean[target_col].values
    # drop target
    X_df = df_clean.drop(columns=[target_col])
    # apply preprocessor if available
    if preprocessor is not None and prep is not None:
        expected_cols = set(getattr(preprocessor, "feature_names_in_", []))
        missing_cols = expected_cols - set(X_df.columns) if expected_cols else set()
        if missing_cols:
            st.warning(f"전처리기 입력 컬럼이 없어 원본 수치형만 사용합니다. 누락: {sorted(missing_cols)}")
            X_df = X_df.select_dtypes(include=["number"])
        else:
            try:
                X_df = prep.apply_preprocessor(preprocessor, X_df)
            except Exception as e:
                st.warning(f"전처리기 적용 실패, 원본 수치형만 사용합니다: {e}")
                X_df = X_df.select_dtypes(include=["number"])
    else:
        X_df = X_df.select_dtypes(include=["number"])

    if X_df.shape[1] == 0:
        raise ValueError("사용 가능한 수치형 특징이 없습니다.")

    # align to hint if provided
    if feature_names_hint:
        common = [c for c in feature_names_hint if c in X_df.columns]
        missing = [c for c in feature_names_hint if c not in X_df.columns]
        if missing:
            st.warning(f"평가 데이터에 없는 특징이 있어 제외합니다: {missing}")
        if common:
            X_df = X_df[common]
    feature_names = list(X_df.columns)
    task = ms.detect_task_type(pd.DataFrame({target_col: y}), target_col=target_col, forced_task=forced_task)
    return X_df, y, task, feature_names


def prepare_split(
    df: pd.DataFrame,
    target_col: str,
    feature_names_hint: Optional[list] = None,
    preprocessor=None,
    test_size: float = 0.2,
    random_state: int = 42,
    forced_task: Optional[str] = None,
    split_indices: Optional[Tuple[list, list]] = None,
):
    X_df, y, task, feature_names = prepare_features(
        df,
        target_col,
        feature_names_hint=feature_names_hint,
        preprocessor=preprocessor,
        forced_task=forced_task,
    )
    if split_indices:
        train_idx, test_idx = split_indices
        X_train = X_df.iloc[train_idx].values
        X_test = X_df.iloc[test_idx].values
        y_train = y[train_idx]
        y_test = y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if task == "classification" else None,
        )
    return X_train, X_test, y_train, y_test, task, feature_names


def get_or_create_split_indices(
    df: pd.DataFrame,
    target_col: str,
    session_key: str = "val_split_indices",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_flag: bool = False,
    force_new: bool = False,
) -> Tuple[list, list]:
    """Persist split indices in session to ensure consistent train/test separation."""
    saved: Optional[Dict[str, Any]] = st.session_state.get(session_key)
    if (
        not force_new
        and saved
        and saved.get("target_col") == target_col
        and saved.get("n_rows") == len(df)
        and abs(saved.get("test_size", test_size) - test_size) < 1e-6
        and saved.get("random_state", random_state) == random_state
        and saved.get("stratify_flag", stratify_flag) == stratify_flag
    ):
        return saved.get("train_idx", []), saved.get("test_idx", [])

    idx = np.arange(len(df))
    stratify = None
    if stratify_flag:
        try:
            y_vals = df[target_col].values
            if len(pd.unique(y_vals)) > 1:
                stratify = y_vals
        except Exception:
            stratify = None

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    st.session_state[session_key] = {
        "train_idx": train_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "target_col": target_col,
        "n_rows": len(df),
        "test_size": test_size,
        "random_state": random_state,
        "stratify_flag": stratify_flag,
    }
    return train_idx, test_idx


def _build_hpo_model(model_name: str, params: dict, task: str, random_state: int = 0):
    """Instantiate model from HPO results with safety defaults."""
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=random_state, **params)
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(random_state=random_state, **params)
    if model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(random_state=random_state, **params)
    if model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(random_state=random_state, **params)
    if model_name == "SVC":
        # ensure probability flag for downstream metrics
        params = {**params}
        params.setdefault("probability", True)
        params.setdefault("random_state", random_state)
        return SVC(**params)
    if model_name == "LogisticRegression":
        params = {**params}
        params.setdefault("max_iter", 500)
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)
    # fallback to tree models by task
    if task == "classification":
        return RandomForestClassifier(random_state=random_state, **params)
    return RandomForestRegressor(random_state=random_state, **params)


def train_hpo_model(
    df: pd.DataFrame,
    target_col: str,
    params: dict,
    model_name: Optional[str],
    random_state: int = 0,
    preprocessor=None,
    forced_task: Optional[str] = None,
    split_indices: Optional[Tuple[list, list]] = None,
):
    X_train, X_test, y_train, y_test, task, feature_names = prepare_split(
        df,
        target_col,
        feature_names_hint=None,
        preprocessor=preprocessor,
        test_size=0.2,
        random_state=random_state,
        forced_task=forced_task,
        split_indices=split_indices,
    )
    model = _build_hpo_model(model_name or "", params, task, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test, task, feature_names


# ------------ data selection ------------
active_df = get_active_df()
if active_df is None:
    st.error("Upload 단계에서 데이터를 준비한 뒤 다시 시도하세요.")
    st.stop()

target_col = st.session_state.get("target_col")
if target_col is None or target_col not in active_df.columns:
    st.error("타깃 컬럼이 설정되지 않았습니다. Overview에서 설정하거나 이전 단계에서 지정하세요.")
    st.stop()

preproc_obj = st.session_state.get("preprocessing_pipeline")

df_feat = st.session_state.get("df_features")
# 고정: 특징공학/전처리 반영 데이터 사용
df_eval = df_feat
if df_eval is None:
    st.error("평가할 데이터가 없습니다. 이전 단계에서 데이터를 준비하세요.")
    st.stop()
split_state_key = "val_split_indices"
if split_state_key not in st.session_state:
    st.session_state[split_state_key] = None

# If 별도 train/test 특징 데이터가 있으면 사용
df_feat_train = st.session_state.get("df_features_train")
df_feat_test = st.session_state.get("df_features_test")
has_precomputed_split = df_feat_train is not None and df_feat_test is not None
if has_precomputed_split:
    df_eval = pd.concat([df_feat_train, df_feat_test], axis=0).sort_index()
    preproc_obj = None  # 이미 전처리/특징공학 적용됨

# 추천 특징(중요도 기반)만 존재하면 적용
fe_top_features = st.session_state.get("fe_top_features")
if fe_top_features:
    keep_cols = [c for c in fe_top_features if c in df_eval.columns]
    if target_col and target_col in df_eval.columns:
        keep_cols.append(target_col)
    if keep_cols:
        df_eval = df_eval[keep_cols].copy()

# Split info (read-only here; config is set upstream)
split_meta_default = {"test_size": 0.2, "random_state": 42, "stratify": False}
split_meta_saved = st.session_state.get("split_meta") or split_meta_default
train_idx_saved = st.session_state.get("train_idx")
test_idx_saved = st.session_state.get("test_idx")
if not (isinstance(train_idx_saved, list) and isinstance(test_idx_saved, list) and len(train_idx_saved) > 0 and len(test_idx_saved) > 0):
    # fallback: create once based on saved meta
    train_idx_saved, test_idx_saved = get_or_create_split_indices(
        df_eval,
        target_col,
        session_key=split_state_key,
        test_size=float(split_meta_saved.get("test_size", 0.2)),
        random_state=int(split_meta_saved.get("random_state", 42)),
        stratify_flag=bool(split_meta_saved.get("stratify", False)),
        force_new=False,
    )
    st.session_state["train_idx"] = train_idx_saved
    st.session_state["test_idx"] = test_idx_saved

split_indices = (train_idx_saved, test_idx_saved)
col_meta1, col_meta2, col_meta3 = st.columns(3)
with col_meta1:
    st.metric("전체 행 수", df_eval.shape[0])
    st.caption("타깃 결측은 자동 제거 후 사용합니다.")
with col_meta2:
    st.metric("특징 수", df_eval.shape[1])
with col_meta3:
    st.metric("Train/Test", f"{len(split_indices[0])}/{len(split_indices[1])}")
st.caption(
    f"분할 정보(Upload/Overview에서 설정): test 비율 {split_meta_saved.get('test_size', 0.2):.2f}, "
    f"stratify: {bool(split_meta_saved.get('stratify', False))}, random_state: {split_meta_saved.get('random_state', 42)}"
)

# model sources
problem_type_override = st.session_state.get("problem_type")
trained_model = st.session_state.get("trained_model")
baselines = st.session_state.get("baseline_models") or {}
best_baseline_name = st.session_state.get("best_model_name")
hpo_res = st.session_state.get("hpo_result") or {}
best_params = hpo_res.get("best_params") if isinstance(hpo_res, dict) else None
best_params_model = None
if isinstance(hpo_res, dict):
    best_params_model = hpo_res.get("model_name")
if best_params_model is None:
    best_params_model = st.session_state.get("hpo_model_name")

options = []
if trained_model is not None:
    options.append("훈련된 모델 (trained_model)")
if best_baseline_name and baselines.get(best_baseline_name):
    options.append(f"베이스라인 추천 모델 ({best_baseline_name})")
elif baselines:
    options.append("베이스라인 첫 모델")
if best_params:
    options.append("HPO best_params로 새로 학습")

if not options:
    st.error("평가할 모델이 없습니다. Model Selection/HPO 단계 수행 후 다시 시도하세요.")
    st.stop()

st.subheader("검증용 모델 선택")
st.caption("세션에 저장된 모델 우선 → 베이스라인 → HPO 파라미터 재학습 순으로 선택할 수 있습니다.")
model_choice = st.selectbox("평가할 모델 선택", options=options, index=0)

# ------------ prepare model and data ------------
model = None
X_test = None
y_test = None
task = None
feature_names = None

try:
    if model_choice.startswith("훈련된 모델") and trained_model is not None:
        # If X_test/y_test exist in session, use them; else split from current df_eval (apply preprocessor on raw if available)
        if "X_test" in st.session_state and "y_test" in st.session_state:
            model = trained_model
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            feature_names = st.session_state.get("feature_names")
            task = st.session_state.get("problem_type")
        else:
            X_train, X_test, y_train, y_test, task, feature_names = prepare_split(
                df_eval,
                target_col,
                feature_names_hint=None,
                preprocessor=preproc_obj,
                forced_task=problem_type_override,
                split_indices=split_indices,
            )
            model = trained_model
    elif model_choice.startswith("베이스라인") and baselines:
        entry = baselines.get(best_baseline_name) if best_baseline_name and baselines.get(best_baseline_name) else list(baselines.values())[0]
        base_model = entry.get("model")
        task = entry.get("task")
        feature_names = entry.get("feature_names")
        X_train, X_test, y_train, y_test, task, feature_names = prepare_split(
            df_eval,
            target_col,
            feature_names_hint=feature_names,
            preprocessor=preproc_obj,
            forced_task=problem_type_override,
            split_indices=split_indices,
        )
        try:
            model = clone(base_model)
        except Exception:
            model = base_model
        if model is not None:
            model.fit(X_train, y_train)
    elif model_choice.startswith("HPO") and best_params:
        # retrain RF with best_params; use preprocessor on raw if available
        model, X_test, y_test, task, feature_names = train_hpo_model(
            df_eval,
            target_col,
            best_params,
            best_params_model,
            random_state=0,
            preprocessor=preproc_obj,
            forced_task=problem_type_override,
            split_indices=split_indices,
        )
    else:
        st.error("선택한 모델을 준비할 수 없습니다.")
        st.stop()
except Exception as e:
    st.error(f"검증 데이터/모델 준비 실패: {e}")
    st.stop()

# ------------ evaluation ------------
st.write(f"검증 샘플 수: {len(y_test)}, 특징 수: {X_test.shape[1] if hasattr(X_test, 'shape') else 'unknown'}")

try:
    y_pred = model.predict(X_test)
except Exception as e:
    st.error(f"모델 예측 실패: {e}")
    st.stop()

if task is None:
    task = ms.detect_task_type(pd.DataFrame({target_col: y_test}), target_col=target_col)

val_result: Dict[str, Any] = {
    "task": task,
    "n_test": int(len(y_test)),
    "n_features": int(X_test.shape[1]) if hasattr(X_test, "shape") else None,
}

if task == "classification":
    st.header("Classification metrics")
    try:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Accuracy", f"{acc:.4f}")
        mcol2.metric("F1 (weighted)", f"{f1:.4f}")
        mcol3.metric("Precision", f"{prec:.4f}")
        mcol4.metric("Recall", f"{rec:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        with st.expander("Confusion Matrix 보기", expanded=True):
            st.pyplot(plot_confusion_matrix(cm, [str(c) for c in np.unique(y_test)]))
        val_result.update(
            {
                "accuracy": float(acc),
                "f1_weighted": float(f1),
                "precision_weighted": float(prec),
                "recall_weighted": float(rec),
                "classes": [str(c) for c in np.unique(y_test)],
            }
        )

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            with st.expander("ROC / PR Curve (이진)", expanded=False):
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                fig_roc, axr = plt.subplots(figsize=(6, 5))
                axr.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                axr.plot([0, 1], [0, 1], linestyle="--", color="gray")
                axr.set_xlabel("FPR")
                axr.set_ylabel("TPR")
                axr.legend()
                st.pyplot(fig_roc)

                precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba[:, 1])
                fig_pr, axpr = plt.subplots(figsize=(6, 5))
                axpr.plot(recall_vals, precision_vals)
                axpr.set_xlabel("Recall")
                axpr.set_ylabel("Precision")
                st.pyplot(fig_pr)
                val_result.update({"auc": float(roc_auc)})
    except Exception as e:
        st.error(f"분류 지표 계산 실패: {e}")

else:
    st.header("Regression metrics")
    try:
        y_true = np.array(y_test, dtype=float)
        y_pred_f = np.array(y_pred, dtype=float)
        mae = mean_absolute_error(y_true, y_pred_f)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_f))
        r2 = r2_score(y_true, y_pred_f)
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("MAE", f"{mae:.4f}")
        mcol2.metric("RMSE", f"{rmse:.4f}")
        mcol3.metric("R²", f"{r2:.4f}")
        with st.expander("Residual plot", expanded=True):
            st.pyplot(plot_regression_residuals(y_true, y_pred_f))
        val_result.update({"mae": float(mae), "rmse": float(rmse), "r2": float(r2)})
    except Exception as e:
        st.error(f"회귀 지표 계산 실패: {e}")

# Save concise validation summary for report page
st.session_state["validation_result"] = val_result

# ------------ importance ------------
st.header("Feature importance")
try:
    if isinstance(X_test, pd.DataFrame):
        feature_names = list(X_test.columns)
        X_for_imp = X_test.values
    else:
        feature_names = feature_names or [f"f{i}" for i in range(X_test.shape[1])]
        X_for_imp = X_test

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).reset_index(drop=True)
        with st.expander("내장 feature_importances_ 확인", expanded=False):
            st.dataframe(imp_df.head(30))
    else:
        st.caption("내장 feature_importances_가 없어 permutation importance로 계산합니다.")
        with st.spinner("Permutation importance 계산 중..."):
            r = permutation_importance(model, X_for_imp, y_test, n_repeats=10, random_state=0, n_jobs=1)
            imp_df = pd.DataFrame(
                {"feature": feature_names, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
            ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        with st.expander("Permutation importance 결과 보기", expanded=False):
            st.dataframe(imp_df.head(30))
except Exception as e:
    st.error(f"특징 중요도 계산 실패: {e}")

# ------------ SHAP ------------
if _HAS_EXPLAIN and explain_mod is not None:
    st.header("SHAP 설명 (옵션)")
    st.caption("무겁게 동작할 수 있으니 필요할 때만 실행하세요.")
    if st.button("SHAP 요약/워터폴 생성"):
        try:
            X_for_shap = X_test if isinstance(X_test, (pd.DataFrame, np.ndarray)) else np.asarray(X_test)
            expl = explain_mod.explain_model_shap(
                model,
                X_for_shap,
                feature_names=feature_names,
                max_display=20,
                sample_for_background=100,
                single_sample_index=0,
            )
            if expl.get("summary_png_b64"):
                st.image(expl["summary_png_b64"])
            if expl.get("waterfall_png_b64"):
                st.image(expl["waterfall_png_b64"])
            if expl.get("error"):
                st.error(f"SHAP 오류: {expl.get('error')}")
        except Exception as e:
            st.error(f"SHAP 실행 실패: {e}")
else:
    st.info("SHAP 모듈을 사용할 수 없습니다. 필요 시 requirements.txt의 shap을 설치하고 modules/explain을 확인하세요.")

st.success("Validation 완료.")
