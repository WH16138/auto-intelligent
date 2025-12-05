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
from typing import Optional, Tuple
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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


def get_active_df() -> Tuple[Optional[pd.DataFrame], bool]:
    df_feat = st.session_state.get("df_features")
    if df_feat is not None:
        return df_feat, True
    df_pre = st.session_state.get("df_preprocessed")
    if df_pre is not None:
        return df_pre, True
    return st.session_state.get("df"), False


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    feature_names_hint: Optional[list] = None,
    preprocessor=None,
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
    task = ms.detect_task_type(pd.DataFrame({target_col: y}), target_col=target_col)
    return X_df, y, task, feature_names


def prepare_split(
    df: pd.DataFrame,
    target_col: str,
    feature_names_hint: Optional[list] = None,
    preprocessor=None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_df, y, task, feature_names = prepare_features(df, target_col, feature_names_hint=feature_names_hint, preprocessor=preprocessor)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == "classification" else None,
    )
    return X_train, X_test, y_train, y_test, task, feature_names


def train_hpo_model(df: pd.DataFrame, target_col: str, params: dict, random_state: int = 0, preprocessor=None):
    X_train, X_test, y_train, y_test, task, feature_names = prepare_split(
        df,
        target_col,
        feature_names_hint=None,
        preprocessor=preprocessor,
        test_size=0.2,
        random_state=random_state,
    )
    if task == "classification":
        model = RandomForestClassifier(random_state=random_state, **params)
    else:
        model = RandomForestRegressor(random_state=random_state, **params)
    model.fit(X_train, y_train)
    return model, X_test, y_test, task, feature_names


# ------------ data selection ------------
active_df, used_pre = get_active_df()
if active_df is None:
    st.error("Upload/Preprocessing 단계에서 데이터를 준비한 뒤 다시 시도하세요.")
    st.stop()

target_col = st.session_state.get("target_col")
if target_col is None or target_col not in active_df.columns:
    st.error("타깃 컬럼이 설정되지 않았습니다. Overview에서 설정하거나 이전 단계에서 지정하세요.")
    st.stop()

preproc_obj = st.session_state.get("preprocessing_pipeline")

df_feat = st.session_state.get("df_features")
df_pre = st.session_state.get("df_preprocessed")
df_raw = st.session_state.get("df")
# 고정: 특징공학/전처리 반영 데이터 우선, 없으면 원본
df_eval = df_feat or df_pre or df_raw
if df_eval is None:
    st.error("평가할 데이터가 없습니다. 이전 단계에서 데이터를 준비하세요.")
    st.stop()

# model sources
trained_model = st.session_state.get("trained_model")
trained_problem_type = st.session_state.get("problem_type")
baselines = st.session_state.get("baseline_models") or {}
best_baseline_name = st.session_state.get("best_model_name")
hpo_res = st.session_state.get("hpo_result") or {}
best_params = hpo_res.get("best_params") if isinstance(hpo_res, dict) else None

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
                preprocessor=preproc_obj if df_eval is df_raw else None,
            )
            model = trained_model
    elif model_choice.startswith("베이스라인") and baselines:
        entry = baselines.get(best_baseline_name) if best_baseline_name and baselines.get(best_baseline_name) else list(baselines.values())[0]
        model = entry.get("model")
        task = entry.get("task")
        feature_names = entry.get("feature_names")
        _, X_test, _, y_test, _, _ = prepare_split(
            df_eval,
            target_col,
            feature_names_hint=feature_names,
            preprocessor=preproc_obj if df_eval is df_raw else None,
        )
    elif model_choice.startswith("HPO") and best_params:
        # retrain RF with best_params; use preprocessor on raw if available
        model, X_test, y_test, task, feature_names = train_hpo_model(
            df_eval if df_eval is not None else df_raw,
            target_col,
            best_params,
            random_state=0,
            preprocessor=preproc_obj if df_eval is df_raw else None,
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

if task == "classification":
    st.header("Classification metrics")
    try:
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("F1 (weighted)", f"{f1:.4f}")
        st.metric("Precision (weighted)", f"{prec:.4f}")
        st.metric("Recall (weighted)", f"{rec:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(cm, [str(c) for c in np.unique(y_test)]))

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception:
                y_proba = None
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            st.subheader("ROC Curve (binary)")
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            fig_roc, axr = plt.subplots(figsize=(6, 5))
            axr.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            axr.plot([0, 1], [0, 1], linestyle="--", color="gray")
            axr.set_xlabel("FPR")
            axr.set_ylabel("TPR")
            axr.legend()
            st.pyplot(fig_roc)

            st.subheader("Precision-Recall Curve")
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba[:, 1])
            fig_pr, axpr = plt.subplots(figsize=(6, 5))
            axpr.plot(recall_vals, precision_vals)
            axpr.set_xlabel("Recall")
            axpr.set_ylabel("Precision")
            st.pyplot(fig_pr)
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
        st.metric("MAE", f"{mae:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("R²", f"{r2:.4f}")
        st.subheader("Residual plot")
        st.pyplot(plot_regression_residuals(y_true, y_pred_f))
    except Exception as e:
        st.error(f"회귀 지표 계산 실패: {e}")

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
        st.subheader("Built-in feature_importances_")
        st.dataframe(imp_df.head(50))
    else:
        st.info("내장 feature_importances_가 없어 permutation importance로 계산합니다.")
        with st.spinner("Permutation importance 계산 중..."):
            r = permutation_importance(model, X_for_imp, y_test, n_repeats=10, random_state=0, n_jobs=1)
            imp_df = pd.DataFrame(
                {"feature": feature_names, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
            ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
            st.dataframe(imp_df.head(50))
except Exception as e:
    st.error(f"특징 중요도 계산 실패: {e}")

# ------------ SHAP ------------
if _HAS_EXPLAIN and explain_mod is not None:
    st.header("SHAP 설명 (옵션)")
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
