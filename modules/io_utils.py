"""
modules/io_utils.py

모델/파이프라인/파라미터/리포트 저장 및 재현 스크립트 생성 유틸.

주요 함수
- ensure_dirs(base_artifacts='artifacts', reports_dir='reports'): 필요한 디렉터리 생성
- save_model(obj, path): joblib으로 모델 저장
- load_model(path): joblib으로 모델 로드
- save_json(obj, path): json 파일 저장 (파라미터 등)
- load_json(path): json 파일 로드
- save_report_html(html_str, path): HTML 리포트 저장
- generate_train_script(params, model_filename, pipeline_filename, script_path, target_col='target'):
    HPO 결과(또는 하이퍼파라미터 dict)를 바탕으로 재현 가능한 train_best.py 스크립트 생성
- snapshot_artifacts(...): 모델 + 파이프라인 + params를 timestamped 폴더에 저장하고 경로 반환

설치 의존성: joblib (requirements.txt에 포함되어 있어야 함)
"""
from typing import Any, Dict, Optional
import os
import json
import joblib
import datetime
import textwrap

def ensure_dirs(base_artifacts: str = "artifacts", reports_dir: str = "reports") -> None:
    """필요한 폴더(artifacts, reports)를 생성합니다."""
    os.makedirs(base_artifacts, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)


def save_model(obj: Any, path: str) -> None:
    """joblib을 이용해 모델/객체 저장."""
    ensure_dirs()
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: str) -> Any:
    """joblib으로 저장된 모델/객체 로드."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {path}")
    return joblib.load(path)


def save_json(obj: Dict[str, Any], path: str, indent: int = 2) -> None:
    """파라미터·메타 정보를 JSON으로 저장."""
    ensure_dirs()
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 파일이 존재하지 않습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_report_html(html_str: str, path: str) -> None:
    """HTML 리포트를 파일로 저장."""
    ensure_dirs()
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)


def _params_to_assignments(params: Dict[str, Any]) -> str:
    """
    파라미터 dict -> Python 코드(사전 리터럴) 형태로 포맷.
    숫자/불리언/문자열 등을 적절히 직렬화.
    """
    # Use json.dumps then embed to preserve types (and ensure_ascii=False for readability)
    return json.dumps(params, indent=4, ensure_ascii=False)


def generate_train_script(
    params: Dict[str, Any],
    model_filename: str,
    pipeline_filename: Optional[str],
    script_path: str,
    target_col: str = "target",
    data_arg_name: str = "data_path",
) -> None:
    """
    HPO 결과(파라미터 dict)를 이용해 재현 가능한 파이썬 스크립트를 생성합니다.
    생성되는 스크립트는:
      - CSV(명령행 인자)를 읽고,
      - (선택적으로 저장된) 파이프라인을 로드하여 전처리 적용,
      - 하이퍼파라미터를 하드코딩한 모델을 생성/학습,
      - artifacts로 모델을 저장합니다.

    args:
      params: 하이퍼파라미터 dict (예: {'n_estimators': 100, 'max_depth': 8})
      model_filename: 생성될 모델 파일명(예: 'artifacts/best_model.pkl')
      pipeline_filename: 전처리 파이프라인이 있으면 그 파일 경로(또는 None)
      script_path: 생성될 스크립트 경로(예: 'scripts/train_best.py')
      target_col: 타깃 열 이름
      data_arg_name: 스크립트에서 받을 커맨드라인 인자 이름
    """
    # ensure directories for script
    dirpath = os.path.dirname(script_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    params_literal = _params_to_assignments(params)

    pipeline_block = ""
    pipeline_load_block = ""
    pipeline_apply_block = ""
    if pipeline_filename:
        pipeline_load_block = textwrap.dedent(f"""
            # Load preprocessor (if you saved one)
            import joblib
            preprocessor = joblib.load(r\"{pipeline_filename}\")
        """)
        pipeline_apply_block = textwrap.dedent(f"""
            # Apply preprocessor (ensure preprocessor expects pandas DataFrame and returns numpy/pandas)
            try:
                X = preprocessor.transform(df.drop(columns=[\"{target_col}\"]))
            except Exception:
                # if not fitted or transformer expects different interface, try fit_transform
                X = preprocessor.fit_transform(df.drop(columns=[\"{target_col}\"]))
        """)
    else:
        pipeline_apply_block = textwrap.dedent(f"""
            # No preprocessor provided — use numeric columns only as a fallback
            X = df.select_dtypes(include=[\"number\"]).drop(columns=[\"{target_col}\"], errors='ignore').fillna(0).values
        """)

    script = f"""\
#!/usr/bin/env python3
# Auto-generated train script (for reproducibility)
# Generated at {datetime.datetime.utcnow().isoformat()} UTC
import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import json
import os

def main({data_arg_name}):
    # Load data
    df = pd.read_csv({data_arg_name})
    if \"{target_col}\" not in df.columns:
        raise ValueError(\"Target column '{target_col}' not found in data\")

{pipeline_load_block}
{pipeline_apply_block}

    y = df[\"{target_col}\"].values

    # Hyperparameters (hardcoded)
    params = {params_literal}

    # Decide model class by target type heuristic (numeric with many unique -> regression)
    try:
        if df[\"{target_col}\"].dtype.kind in 'ifu' and df[\"{target_col}\"].nunique() > 20:
            # regression
            model = RandomForestRegressor(**params)
        else:
            # classification
            model = RandomForestClassifier(**params)
    except Exception:
        # fallback: classification
        model = RandomForestClassifier(**params)

    # Fit
    print(\"Training model with params:\", params)
    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(r\"{model_filename}\"), exist_ok=True)
    joblib.dump(model, r\"{model_filename}\")

    print(\"Saved model to: {model_filename}\")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to CSV file used for training')
    args = parser.parse_args()
    main('args.data_path' if False else args.data_path)
"""
    # write file
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


def snapshot_artifacts(
    model_obj: Optional[Any],
    preprocessor_obj: Optional[Any],
    params: Optional[Dict[str, Any]],
    base_dir: str = "artifacts",
    prefix: str = "snapshot"
) -> Dict[str, str]:
    """
    timestamped snapshot 폴더를 만들어 모델/파이프라인/파라미터를 저장합니다.
    반환 dict: {'model_path':..., 'pipeline_path':..., 'params_path':...}
    """
    ensure_dirs(base_dir, "reports")
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    folder = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(folder, exist_ok=True)

    paths = {}
    if model_obj is not None:
        model_path = os.path.join(folder, "best_model.pkl")
        save_model(model_obj, model_path)
        paths["model_path"] = model_path
    if preprocessor_obj is not None:
        pipeline_path = os.path.join(folder, "preprocessor.joblib")
        save_model(preprocessor_obj, pipeline_path)
        paths["pipeline_path"] = pipeline_path
    if params is not None:
        params_path = os.path.join(folder, "params.json")
        save_json(params, params_path)
        paths["params_path"] = params_path

    return paths
