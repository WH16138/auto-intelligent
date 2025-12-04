# pages/08_validation.py
"""
08 - Validation (평가 전용)

- 세션에서 준비된 모델 및 X_test/y_test를 사용하여 평가합니다.
- 분류/회귀 자동 판별, 주요 지표 및 시각화 제공.
- feature importance: model.feature_importances_ 우선, 없으면 permutation_importance 대체 시도.
- SHAP: modules.explain.explain_model_shap 가 있으면 이미지로 표시.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

# permutation importance
from sklearn.inspection import permutation_importance

# optional explain module
try:
    from modules import explain as explain_mod
    _HAS_EXPLAIN = True
except Exception:
    explain_mod = None
    _HAS_EXPLAIN = False


def plot_confusion_matrix(cm, class_names):
    """confusion matrix 시각화 (matplotlib Figure 반환)"""
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
            ax.text(j, i, format(int(cm[i, j]), "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
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


def page_validation():
    st.title("8 - Model Validation (평가)")

    # 세션 체크
    if "trained_model" not in st.session_state:
        st.warning("세션에 'trained_model'이 없습니다. 이전 단계에서 모델을 학습/저장하세요.")
        return

    for key in ["X_test", "y_test", "problem_type"]:
        if key not in st.session_state:
            st.error(f"세션에 '{key}'가 없습니다. 이전 페이지(혹은 데이터 준비 단계)를 확인하세요.")
            return

    model = st.session_state["trained_model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    problem_type = st.session_state["problem_type"]  # "classification" or "regression"

    st.write(f"테스트 샘플 수: {len(y_test)}, 피처 수: {X_test.shape[1] if hasattr(X_test, 'shape') else 'unknown'}")
    # 예측
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        st.error(f"모델 예측 실패: {e}")
        return

    # 분류
    if problem_type == "classification":
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

            # confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            class_names = [str(c) for c in np.unique(y_test)]
            fig_cm = plot_confusion_matrix(cm, class_names)
            st.pyplot(fig_cm)

            # ROC/PR if probabilities available and binary or multiclass
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
            else:
                st.info("predict_proba 또는 적절한 확률 출력이 없어 ROC/PR를 계산할 수 없습니다 (또는 multiclass 확률 처리 필요).")
        except Exception as e:
            st.error(f"분류 지표 계산 중 오류: {e}")

    # 회귀
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
            fig_res = plot_regression_residuals(y_true, y_pred_f)
            st.pyplot(fig_res)
        except Exception as e:
            st.error(f"회귀 지표 계산 중 오류: {e}")

    # Feature importance: try built-in, else permutation importance
    st.header("Feature importance")
    try:
        feature_names = None
        # X_test could be DataFrame or numpy
        if isinstance(X_test, pd.DataFrame):
            feature_names = list(X_test.columns)
            X_for_imp = X_test.values
        else:
            # if numpy and session stores names
            feature_names = st.session_state.get("feature_names", [f"f{i}" for i in range(X_test.shape[1])])
            X_for_imp = X_test

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
            st.subheader("Built-in feature_importances_")
            st.dataframe(imp_df.head(50))
        else:
            st.info("내장 importance가 없어 permutation importance를 시도합니다 (시간 소요 가능).")
            with st.spinner("Permutation importance 계산 중..."):
                try:
                    r = permutation_importance(model, X_for_imp, y_test, n_repeats=10, random_state=0, n_jobs=1)
                    imp_df = pd.DataFrame({"feature": feature_names, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
                    imp_df = imp_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
                    st.dataframe(imp_df.head(50))
                except Exception as e:
                    st.error(f"permutation importance 계산 실패: {e}")
    except Exception as e:
        st.error(f"피처 중요도 계산 중 오류: {e}")

    # SHAP 설명 (optional)
    if _HAS_EXPLAIN and explain_mod is not None:
        st.header("SHAP 설명 (선택)")
        if st.button("SHAP 요약/샘플 설명 생성"):
            try:
                # use explain_mod.explain_model_shap which returns base64 images
                feat_names_for_shap = feature_names
                # call with DataFrame if available
                X_for_shap = X_test if isinstance(X_test, (pd.DataFrame, np.ndarray)) else np.asarray(X_test)
                expl = explain_mod.explain_model_shap(model, X_for_shap, feature_names=feat_names_for_shap, max_display=20, sample_for_background=100, single_sample_index=0)
                if expl.get("summary_png_b64"):
                    st.image(expl["summary_png_b64"])
                if expl.get("waterfall_png_b64"):
                    st.image(expl["waterfall_png_b64"])
                if expl.get("error"):
                    st.error(f"SHAP 오류: {expl.get('error')}")
            except Exception as e:
                st.error(f"SHAP 생성 실패: {e}")
    else:
        st.info("SHAP 모듈이 없거나 modules.explain가 준비되지 않았습니다. 설치 시 SHAP 기반 해석을 제공할 수 있습니다.")

    st.success("Validation 완료.")


if __name__ == "__main__":
    page_validation()
