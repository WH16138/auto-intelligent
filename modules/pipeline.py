"""
modules/pipeline.py

자동화 워크플로 오케스트레이터 (MVP -> 확장용)

주요 기능
- 입력(CSV path 또는 DataFrame) 수용
- fingerprint 생성 (modules.ingestion.generate_fingerprint)
- 전처리 파이프라인 빌드/적용 (modules.preprocessing)
- 선택적 피처엔지니어링 후(가능하면) 베이스라인 모델 탐색 (modules.model_search)
- 선택적 HPO 실행(modules.hpo 가 존재하면 사용)
- 결과(모델, 전처리기, 성능, 아티팩트 경로) 저장 (modules.io_utils)
- 재현 가능한 train script 자동 생성 (modules.io_utils.generate_train_script)

사용 예:
    from modules.pipeline import run_full_pipeline
    res = run_full_pipeline(data="data/mydata.csv", target_col="target", run_hpo=False, hpo_time=120)
"""
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import traceback
import logging
import inspect

logger = logging.getLogger(__name__)

# try to import optional modules; if missing, handle gracefully
try:
    from modules import ingestion
except Exception:
    # fallback to one-off import path (in case user uses different structure)
    import modules.ingestion as ingestion  # type: ignore

try:
    from modules import preprocessing
except Exception:
    import modules.preprocessing as preprocessing  # type: ignore

try:
    from modules import feature_engineering
    _HAS_FE = True
except Exception:
    feature_engineering = None
    _HAS_FE = False

try:
    from modules import model_search
except Exception:
    import modules.model_search as model_search  # type: ignore

# hpo may be under modules.hpo if implemented by user; optional
try:
    from modules import hpo as hpo_module
    _HAS_HPO = True
except Exception:
    hpo_module = None
    _HAS_HPO = False

try:
    from modules import io_utils
except Exception:
    import modules.io_utils as io_utils  # type: ignore

# explain module optional (SHAP)
try:
    from modules import explain as explain_module
    _HAS_EXPLAIN = True
except Exception:
    explain_module = None
    _HAS_EXPLAIN = False


def _load_input(data: Any) -> pd.DataFrame:
    """data가 str(path)인지 pandas.DataFrame인지 판단하여 DataFrame으로 반환"""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str):
        # try safe_read_csv if available
        if hasattr(ingestion, "safe_read_csv"):
            return ingestion.safe_read_csv(data)
        else:
            return pd.read_csv(data)
    # try to interpret file-like with pandas
    try:
        return pd.read_csv(data)
    except Exception as e:
        raise ValueError(f"지원하지 않는 입력 형태입니다: {e}")


def _select_best_model_from_baselines(baselines_df: pd.DataFrame) -> Optional[str]:
    """baselines 결과 DataFrame에서 가장 좋은 모델명을 선택하는 단순 정책"""
    if baselines_df is None or baselines_df.empty:
        return None
    # classification 우선 mean_accuracy, regression mean_r2
    if "mean_accuracy" in baselines_df.columns:
        return baselines_df.sort_values(by="mean_accuracy", ascending=False).iloc[0]["model"]
    if "mean_r2" in baselines_df.columns:
        return baselines_df.sort_values(by="mean_r2", ascending=False).iloc[0]["model"]
    # fallback: first row
    return baselines_df.iloc[0]["model"]


def run_full_pipeline(
    data: Any,
    target_col: Optional[str] = None,
    run_hpo: bool = False,
    hpo_time: int = 120,
    snapshot_prefix: str = "run",
    artifacts_base: str = "artifacts",
    save_artifacts: bool = True,
    fe_max_new: int = 10,
) -> Dict[str, Any]:
    """
    전체 워크플로를 실행하는 주 함수.

    Args:
        data: pandas.DataFrame 또는 CSV 파일 경로 또는 file-like
        target_col: 타깃 컬럼명 (없으면 ingestion.generate_fingerprint의 권장값 사용)
        run_hpo: True이면 HPO를 시도 (modules.hpo가 존재해야 함)
        hpo_time: HPO 시간 예산(초)
        snapshot_prefix: snapshot 폴더 접두사
        artifacts_base: 아티팩트 저장 기본 폴더
        save_artifacts: True면 모델/파이프라인/params snapshot을 저장
        fe_max_new: 피처 엔지니어링 시 최대 생성 피처 수 (모듈이 존재할 때만 사용)

    Returns:
        dict with keys:
        - 'fingerprint', 'preprocessor', 'df_preprocessed', 'baselines_df', 'models',
          'best_model_name', 'best_model', 'hpo_result', 'artifact_paths', 'error' (if any)
    """
    result: Dict[str, Any] = {
        "fingerprint": None,
        "preprocessor": None,
        "df_preprocessed": None,
        "baselines_df": None,
        "models": None,
        "best_model_name": None,
        "best_model": None,
        "hpo_result": None,
        "artifact_paths": None,
        "error": None,
    }

    try:
        # --- 1) Load input ---
        df = _load_input(data)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("데이터를 DataFrame으로 로드하지 못했습니다.")
        logger.info("데이터 로드 완료: %s 행 x %s 열", df.shape[0], df.shape[1])

        # --- 2) Fingerprint ---
        try:
            fp = ingestion.generate_fingerprint(df, detect_target=True)
        except Exception:
            # fallback simple fingerprint
            fp = {"n_rows": df.shape[0], "n_cols": df.shape[1]}
        result["fingerprint"] = fp

        # target selection
        if target_col is None:
            target_col = fp.get("target_column")
        result["fingerprint"]["inferred_target"] = target_col

        # --- 3) Preprocessing build & apply ---
        preprocessor = preprocessing.build_preprocessor(df)
        df_transformed = preprocessing.apply_preprocessor(preprocessor, df)
        result["preprocessor"] = preprocessor
        result["df_preprocessed"] = df_transformed

        # If the preprocessor transformed to numpy-like with numeric names, ensure DataFrame
        if not isinstance(df_transformed, pd.DataFrame):
            # apply_preprocessor is implemented to return DataFrame but keep safe fallback
            try:
                df_transformed = pd.DataFrame(df_transformed)
            except Exception:
                pass
        result["df_preprocessed"] = df_transformed

        # --- 4) Optional Feature Engineering ---
        if _HAS_FE and feature_engineering is not None:
            try:
                # attempt auto feature generation; module should expose auto_generate_features
                if hasattr(feature_engineering, "auto_generate_features"):
                    df_fe = feature_engineering.auto_generate_features(df_transformed, max_new=fe_max_new)
                    # ensure DataFrame
                    if isinstance(df_fe, pd.DataFrame) and not df_fe.empty:
                        df_for_model = df_fe
                    else:
                        df_for_model = df_transformed
                else:
                    df_for_model = df_transformed
            except Exception as e:
                logger.warning("Feature engineering 실패: %s", e)
                df_for_model = df_transformed
        else:
            df_for_model = df_transformed

        # If the target column exists in original df but not in transformed (common), try to reattach it
        if target_col is not None and target_col in df.columns and target_col not in df_for_model.columns:
            try:
                df_for_model[target_col] = df[target_col].values
            except Exception:
                # if shapes mismatch, skip
                logger.warning("타깃 컬럼을 df_for_model에 재부착하지 못했습니다.")

        # --- 5) Quick baselines ---
        baselines_df, trained_models = model_search.quick_baselines(df_for_model, target_col=target_col)
        result["baselines_df"] = baselines_df
        result["models"] = trained_models

        best_name = _select_best_model_from_baselines(baselines_df)
        result["best_model_name"] = best_name
        if best_name and trained_models.get(best_name):
            result["best_model"] = trained_models[best_name].get("model")

        # --- 6) Optional HPO ---
        hpo_res = None
        if run_hpo and _HAS_HPO and hpo_module is not None:
            try:
                # prepare X,y using model_search helper
                try:
                    X, y, task, feature_names = model_search.get_X_y(df_for_model, target_col=target_col)
                except Exception:
                    X, y, task, feature_names = None, None, None, None

                if X is not None and y is not None:
                    # choose model_name default to best_name or RandomForest
                    model_to_tune = best_name or "RandomForest"
                    # call hpo module's main entry if available
                    if hasattr(hpo_module, "run_hpo"):
                        # run_hpo should accept (model_name, X, y, time_budget, direction) or similar
                        try:
                            hpo_res = hpo_module.run_hpo(model_name=model_to_tune, X=X, y=y, time_budget=hpo_time)
                        except TypeError:
                            # try alternative signature
                            hpo_res = hpo_module.run_hpo(X=X, y=y, time_budget=hpo_time, model_name=model_to_tune)
                    elif hasattr(hpo_module, "run_optuna_sample"):
                        # fallback to earlier simple interface
                        # hpo_module.run_optuna_sample(df, time_budget)
                        try:
                            hpo_res = hpo_module.run_optuna_sample(df_for_model, hpo_time)
                        except Exception:
                            hpo_res = None
                result["hpo_result"] = hpo_res
                # if hpo returns best_params, consider refitting best model with best_params
                if hpo_res and isinstance(hpo_res, dict) and hpo_res.get("best_params"):
                    best_params = hpo_res["best_params"]
                    if result["best_model"] is not None:
                        try:
                            # instantiate same class with best_params if class accessible
                            cls = result["best_model"].__class__
                            new_model = cls(**best_params)
                            # fit on full data
                            if X is not None and y is not None:
                                new_model.fit(X, y)
                                result["best_model"] = new_model
                        except Exception as e:
                            logger.warning("HPO 후 모델 재학습 실패: %s", e)
            except Exception as e:
                logger.warning("HPO 실행 중 오류: %s", e)
        else:
            result["hpo_result"] = None

        # --- 7) Save artifacts & generate train script ---
        artifact_paths = None
        try:
            if save_artifacts:
                # save model & preprocessor using io_utils.snapshot_artifacts
                model_obj = result.get("best_model")
                preprocessor_obj = result.get("preprocessor")
                params_obj = None
                # Prefer HPO params, else model.get_params()
                if result.get("hpo_result") and isinstance(result["hpo_result"], dict) and result["hpo_result"].get("best_params"):
                    params_obj = result["hpo_result"]["best_params"]
                elif model_obj is not None and hasattr(model_obj, "get_params"):
                    try:
                        params_obj = model_obj.get_params()
                    except Exception:
                        params_obj = None

                artifact_paths = io_utils.snapshot_artifacts(
                    model_obj=model_obj,
                    preprocessor_obj=preprocessor_obj,
                    params=params_obj,
                    base_dir=artifacts_base,
                    prefix=snapshot_prefix,
                )
                # generate train script if params available
                if params_obj is not None:
                    model_path = artifact_paths.get("model_path", f"{artifacts_base}/best_model.pkl")
                    pipeline_path = artifact_paths.get("pipeline_path")
                    script_path = f"scripts/train_best_{snapshot_prefix}.py"
                    io_utils.generate_train_script(
                        params=params_obj,
                        model_filename=model_path,
                        pipeline_filename=pipeline_path,
                        script_path=script_path,
                        target_col=target_col or "target",
                    )
                result["artifact_paths"] = artifact_paths
        except Exception as e:
            logger.warning("artifact 저장 실패: %s", e)

        # --- 8) Optional explain (SHAP) generation ---
        if _HAS_EXPLAIN and explain_module is not None and result.get("best_model") is not None:
            try:
                # try to generate summary explanation; expect explain_module.explain_model_shap
                if hasattr(explain_module, "explain_model_shap"):
                    X_for_explain, _, _, feat_names = None, None, None, None
                    try:
                        X_for_explain, _, _, feat_names = model_search.get_X_y(df_for_model, target_col=target_col)
                    except Exception:
                        pass
                    if X_for_explain is not None:
                        explain_res = explain_module.explain_model_shap(result["best_model"], X_for_explain, feature_names=feat_names)
                        result["explain"] = explain_res
            except Exception as e:
                logger.warning("설명(Explain) 생성 실패: %s", e)

        # success path
        return result

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("run_full_pipeline 실패: %s\n%s", e, tb)
        result["error"] = {"message": str(e), "traceback": tb}
        return result
