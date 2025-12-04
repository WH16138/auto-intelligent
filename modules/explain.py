"""
modules/explain.py

SHAP 기반 모델 설명 유틸

주요 함수
- explain_model_shap(model, X, feature_names=None, max_display=20, sample_for_background=100)
    : 모델과 입력 X에 대해 SHAP 요약(전체) + 개별 샘플(첫 샘플) 설명 이미지를 생성하고 base64로 반환.
      반환 dict 예:
        {
          'summary_png_b64': "...",
          'waterfall_png_b64': "...",
          'meta': {'explainer_type': 'TreeExplainer', 'n_background': 100}
        }

- get_shap_explainer(model, X_background=None)
    : 적절한 SHAP explainer를 생성하여 반환 (TreeExplainer 우선, 없으면 KernelExplainer 사용).
      KernelExplainer는 배경 데이터가 필요하므로 X_background를 권장.

요구사항:
- shap 패키지 설치 필요 (requirements.txt에 추가 권장)
- matplotlib이 필요 (표준)
"""
from typing import Any, Dict, Optional, Sequence, Tuple
import io
import base64
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try import shap and matplotlib; provide helpful error if missing.
try:
    import shap
except Exception as e:
    shap = None
    logger.warning("shap 라이브러리를 찾을 수 없음: %s", e)

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    logger.warning("matplotlib을 찾을 수 없음: %s", e)


def _fig_to_base64_png(fig) -> str:
    """matplotlib Figure를 PNG base64 문자열로 반환"""
    buf = io.BytesIO()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _df_from_X(X: Any, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """X가 numpy나 DataFrame일 때 pandas DataFrame으로 변환하며 feature_names 적용"""
    if isinstance(X, pd.DataFrame):
        dfX = X.copy()
        if feature_names is not None:
            dfX.columns = feature_names
        return dfX
    else:
        arr = np.asarray(X)
        if feature_names is None:
            cols = [f"f{i}" for i in range(arr.shape[1])]
        else:
            cols = list(feature_names)
        return pd.DataFrame(arr, columns=cols)


def get_shap_explainer(model: Any, X_background: Optional[Any] = None, n_background_max: int = 100) -> Tuple[Any, Dict[str, Any]]:
    """
    모델에 적합한 SHAP explainer를 생성한다.
    반환: (explainer, meta)
    meta: {'explainer_type': 'TreeExplainer'|'KernelExplainer'|'GradientExplainer'|... , 'n_background': int}

    전략:
      1) Tree 계열 모델이면 TreeExplainer 사용
      2) 그 외에는 GradientExplainer(가능하면) 또는 KernelExplainer로 폴백
      3) KernelExplainer는 매우 느리므로 background는 최대 n_background_max 샘플로 제한
    """
    if shap is None:
        raise ImportError("shap가 설치되어 있지 않습니다. `pip install shap` 하세요.")

    meta = {"explainer_type": None, "n_background": 0}
    # Prefer TreeExplainer for tree models
    try:
        # detect tree models by module/class name
        model_cls_name = model.__class__.__module__ + "." + model.__class__.__name__
    except Exception:
        model_cls_name = str(type(model))

    # Try TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        meta["explainer_type"] = "TreeExplainer"
        meta["n_background"] = 0
        return explainer, meta
    except Exception:
        logger.debug("TreeExplainer 생성 실패, 다음 시도: GradientExplainer/KernelExplainer")

    # Try GradientExplainer (for some sklearn/pytorch/xgboost variants)
    try:
        if X_background is not None:
            Xb = _df_from_X(X_background)
            explainer = shap.GradientExplainer(model, Xb)
            meta["explainer_type"] = "GradientExplainer"
            meta["n_background"] = min(len(Xb), n_background_max)
            return explainer, meta
    except Exception as e:
        logger.debug("GradientExplainer 시도 실패: %s", e)

    # Fallback: KernelExplainer (very slow) — use small background
    try:
        if X_background is None:
            raise ValueError("KernelExplainer requires background data. X_background를 제공하세요.")
        Xb = _df_from_X(X_background)
        if Xb.shape[0] > n_background_max:
            Xb_sample = Xb.sample(n=n_background_max, random_state=1)
        else:
            Xb_sample = Xb
        explainer = shap.KernelExplainer(model.predict, Xb_sample)
        meta["explainer_type"] = "KernelExplainer"
        meta["n_background"] = Xb_sample.shape[0]
        return explainer, meta
    except Exception as e:
        logger.error("Explainer 생성 실패: %s", e)
        raise RuntimeError(f"SHAP explainer를 생성할 수 없습니다: {e}")


def explain_model_shap(
    model: Any,
    X: Any,
    feature_names: Optional[Sequence[str]] = None,
    max_display: int = 20,
    sample_for_background: int = 100,
    single_sample_index: int = 0,
) -> Dict[str, Any]:
    """
    모델과 데이터 X에 대해 SHAP 설명 이미지를 생성.

    Args:
        model: 학습된 모델 (sklearn-like, tree-based model 권장)
        X: numpy array or pandas DataFrame (n_samples, n_features)
        feature_names: optional list of feature names
        max_display: summary에서 보여줄 최대 피처 수
        sample_for_background: KernelExplainer 사용시 배경으로 쓸 최대 샘플 수
        single_sample_index: waterfall 등 개별 설명에 사용할 인덱스 (기본 첫 샘플)

    Returns:
        dict: {
            'summary_png_b64': data:image/png;base64,... or None,
            'waterfall_png_b64': data:image/png;base64,... or None,
            'meta': { 'explainer_type': ..., 'n_background': ... },
            'error': None or error string
        }

    Notes:
      - 이 함수는 가능한 TreeExplainer를 우선 사용하며, 그 외에는 Gradient/KernalExplainer를 시도합니다.
      - KernelExplainer는 매우 느릴 수 있음. background를 작게 유지하세요.
    """
    out = {"summary_png_b64": None, "waterfall_png_b64": None, "meta": None, "error": None}
    if shap is None:
        out["error"] = "shap 라이브러리가 설치되어 있지 않습니다."
        return out
    if plt is None:
        out["error"] = "matplotlib이 설치되어 있지 않습니다."
        return out

    try:
        # prepare DataFrame
        dfX = _df_from_X(X, feature_names=feature_names)
        # prepare background sample for non-tree explainers
        X_background = None
        if dfX.shape[0] > sample_for_background:
            X_background = dfX.sample(n=sample_for_background, random_state=1)
        else:
            X_background = dfX

        # create explainer
        explainer, meta = get_shap_explainer(model, X_background=X_background, n_background_max=sample_for_background)
        out["meta"] = meta

        # compute shap values
        # For TreeExplainer and GradientExplainer, use explainer(dfX) to get Explanation
        try:
            shap_values = explainer.shap_values(dfX)  # tree explainer may return list for multiclass
        except Exception:
            # fallback to explainer(dfX)
            try:
                shap_values = explainer(dfX)
            except Exception as e:
                logger.debug("explainer.shap_values/()(fallback) 실패: %s", e)
                shap_values = None

        # ---------- summary plot ----------
        try:
            # shap.summary_plot supports different inputs; create matplotlib figure
            fig = plt.figure(figsize=(8, 6))
            # handle multiclass: if shap_values is list, pick mean absolute across classes
            if isinstance(shap_values, list):
                # convert to array of mean abs across classes
                try:
                    # shap_values[i] shape: (n_samples, n_features)
                    sv_arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                    shap.summary_plot(sv_arr, dfX, show=False, max_display=max_display)
                except Exception:
                    shap.summary_plot(shap_values, dfX, show=False, max_display=max_display)
            else:
                # shap_values is array or Explanation
                shap.summary_plot(shap_values, dfX, show=False, max_display=max_display)
            summary_b64 = _fig_to_base64_png(fig)
            out["summary_png_b64"] = summary_b64
        except Exception as e:
            logger.warning("SHAP summary plot 생성 실패: %s", e)
            out["summary_png_b64"] = None

        # ---------- individual explanation (waterfall or force) ----------
        try:
            # choose single sample
            idx = int(single_sample_index) if single_sample_index < dfX.shape[0] else 0
            sample = dfX.iloc[idx: idx+1]
            fig2 = plt.figure(figsize=(8, 6))
            # If explainer returns Explanation object, shap.plots.waterfall can accept it
            try:
                if isinstance(shap_values, list):
                    # for multiclass pick first class's shap_values for sample
                    sv = [sv[idx] for sv in shap_values]
                    # create an Explanation? fallback to waterfall from shap
                    try:
                        shap.plots.waterfall(shap.Explanation(values=sv, base_values=None, data=sample, feature_names=list(sample.columns)), show=False)
                    except Exception:
                        # fallback: use summary_plot style for single sample
                        shap.summary_plot(sv, sample, show=False)
                else:
                    # shap_values is array-like
                    # shap.plots.waterfall expects an Explanation or shap_values for a single instance
                    try:
                        # If explainer provides base_values, try to use waterfall with Explanation
                        if hasattr(explainer, "expected_value"):
                            base = explainer.expected_value
                        else:
                            base = None
                        # create explanation if possible
                        try:
                            expl = shap.Explanation(values=shap_values[idx], base_values=base, data=sample.values, feature_names=list(sample.columns))
                            shap.plots.waterfall(expl, show=False)
                        except Exception:
                            # fallback: use force_plot or bar plot
                            shap.plots.bar(shap_values[idx], feature_names=list(sample.columns), show=False)
                    except Exception:
                        shap.plots.bar(shap_values[idx], feature_names=list(sample.columns), show=False)
            except Exception as e:
                logger.debug("개별 설명 생성 중 오류 (multiclass fallback): %s", e)
                try:
                    shap.plots.bar(shap_values[0], feature_names=list(sample.columns), show=False)
                except Exception:
                    raise

            waterfall_b64 = _fig_to_base64_png(fig2)
            out["waterfall_png_b64"] = waterfall_b64
        except Exception as e:
            logger.warning("개별 SHAP 설명 이미지 생성 실패: %s", e)
            out["waterfall_png_b64"] = None

        return out

    except Exception as e:
        logger.error("explain_model_shap 전체 실패: %s", e, exc_info=True)
        out["error"] = f"{e}"
        return out
