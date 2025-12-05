"""
modules/ingestion.py

- 안전한 CSV 읽기(safe_read_csv): 파일 경로 또는 file-like object 지원, 인코딩 & 구분자 자동 추정(여러 전략 시도)
- 샘플 데이터 로드(load_sample): sklearn의 대표 dataset을 빠르게 불러오기 (breast_cancer, iris)
- 데이터 지문 생성(generate_fingerprint): 컬럼별 통계 + 전반적 메타 정보 반환

Usage (예):
    from modules.ingestion import safe_read_csv, generate_fingerprint, load_sample

    df = safe_read_csv("data/mydata.csv")
    fp = generate_fingerprint(df)
    sample = load_sample("breast_cancer")
"""
from typing import Optional, Union, Dict, Any, IO
import pandas as pd
import io
import csv
import logging

logger = logging.getLogger(__name__)


def _detect_separator(sample: str) -> str:
    """
    간단한 csv.Sniffer 기반 구분자 추정. 실패 시 ',' 반환.
    """
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def safe_read_csv(
    source: Union[str, IO[bytes], IO[str]],
    nrows_detect: int = 5000,
    encodings: Optional[list] = None,
    engine: Optional[str] = None,
) -> pd.DataFrame:
    """
    안전한 CSV 읽기. 파일 경로 또는 file-like 객체를 허용.
    과정:
      1. 바이너리/텍스트로 sample 읽음(최대 nrows_detect * bytes)
      2. csv.Sniffer로 구분자 추정
      3. 여러 인코딩(utf-8, cp949, latin1 등)으로 시도하여 pandas.read_csv 실행
      4. 실패 시 마지막 예외를 raise

    Parameters
    ----------
    source:
        파일 경로 또는 file-like object
    nrows_detect:
        탐지용으로 읽을 바이트 수(문제 있는 매우 큰 파일에도 안전하게)
    encodings:
        시도할 인코딩 리스트(우선순위)
    engine:
        pandas.read_csv에 넘길 engine (기본 None)

    Returns
    -------
    pd.DataFrame
    """
    if encodings is None:
        encodings = ["utf-8", "cp949", "euc-kr", "latin1", "utf-8-sig"]

    # read sample for delimiter detection
    sample_text = ""
    opened_here = False
    try:
        if isinstance(source, str):
            # path
            with open(source, "rb") as f:
                sample_bytes = f.read(min(8192, nrows_detect))
            try:
                sample_text = sample_bytes.decode("utf-8")
            except Exception:
                # fallback latin1 for sniffing
                sample_text = sample_bytes.decode("latin1", errors="ignore")
        else:
            # file-like: try to read from current position and then seek back if possible
            try:
                pos = source.tell()
            except Exception:
                pos = None
            sample_bytes = source.read(min(8192, nrows_detect))
            if isinstance(sample_bytes, bytes):
                try:
                    sample_text = sample_bytes.decode("utf-8")
                except Exception:
                    sample_text = sample_bytes.decode("latin1", errors="ignore")
            else:
                sample_text = str(sample_bytes)
            # reset file pointer if possible
            try:
                if pos is not None:
                    source.seek(pos)
            except Exception:
                pass

        sep = _detect_separator(sample_text)
    except Exception as e:
        logger.warning("구분자 감지 실패: %s — 기본 ',' 사용", e)
        sep = ","

    last_exc = None
    for enc in encodings:
        try:
            if isinstance(source, str):
                df = pd.read_csv(source, sep=sep, encoding=enc, engine=engine)
            else:
                # if file-like, ensure we get a fresh buffer for pandas
                # convert to TextIO if bytes
                if hasattr(source, "read"):
                    # try to read entire content to BytesIO / StringIO then pass to pandas
                    content = source.read()
                    # if bytes, decode with current encoding
                    if isinstance(content, bytes):
                        try:
                            text = content.decode(enc)
                        except Exception:
                            text = content.decode(enc, errors="replace")
                        buf = io.StringIO(text)
                        df = pd.read_csv(buf, sep=sep, engine=engine)
                    else:
                        # content is str
                        buf = io.StringIO(content)
                        df = pd.read_csv(buf, sep=sep, engine=engine)
                else:
                    raise ValueError("Unsupported source type")
            # successful read
            return df
        except Exception as e:
            last_exc = e
            logger.debug("encoding %s 실패: %s", enc, e)
            # try next encoding

    # if reached here, all encodings failed
    raise last_exc if last_exc is not None else ValueError("CSV 읽기 실패")


def load_sample(name: str = "breast_cancer") -> pd.DataFrame:
    """
    sklearn 제공 샘플 데이터 로드(데모용).
    지원: 'breast_cancer', 'iris'
    """
    if name == "breast_cancer":
        try:
            from sklearn.datasets import load_breast_cancer
            d = load_breast_cancer(as_frame=True)
            df = d.frame.copy()
            # older sklearn may not have .frame; fallback:
            if df is None:
                import pandas as pd
                X = d.data
                df = pd.DataFrame(X, columns=d.feature_names)
                df["target"] = d.target
            return df
        except Exception:
            # safe fallback small synthetic
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=200, n_features=10, random_state=0)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["target"] = y
            return df

    if name == "iris":
        try:
            from sklearn.datasets import load_iris
            d = load_iris(as_frame=True)
            df = d.frame.copy()
            # ensure target column exists
            if "target" not in df.columns:
                df["target"] = d.target
            return df
        except Exception:
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=150, n_features=4, random_state=0)
            df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
            df["target"] = y
            return df

    raise ValueError(f"지원하지 않는 샘플 이름: {name}")


def _summary_stats_for_series(s: pd.Series) -> Dict[str, Any]:
    """Series 단위의 간단 통계 반환(문자/숫자 구분)"""
    res: Dict[str, Any] = {}
    res["dtype"] = str(s.dtype)
    res["n_missing"] = int(s.isnull().sum())
    res["n_unique"] = int(s.nunique(dropna=True))
    res["n_total"] = int(s.shape[0])
    if pd.api.types.is_numeric_dtype(s):
        res["mean"] = None if s.dropna().empty else float(s.mean())
        res["std"] = None if s.dropna().empty else float(s.std())
        res["min"] = None if s.dropna().empty else float(s.min())
        res["max"] = None if s.dropna().empty else float(s.max())
    else:
        # top frequencies
        vc = s.value_counts(dropna=True).head(5)
        res["top_values"] = vc.to_dict()
    return res


def generate_fingerprint(df: pd.DataFrame, detect_target: bool = True) -> Dict[str, Any]:
    """
    데이터프레임으로부터 지문(fingerprint) 생성.

    반환 예시 구조:
    {
      'n_rows': 1000,
      'n_cols': 12,
      'missing_ratio': 0.02,
      'columns': {
         'age': {'dtype': 'int64', 'n_missing': 0, 'n_unique': 50, 'mean': ...},
         ...
      },
      'target_column': 'target' or None,
      'cardinality_summary': {'low': 5, 'medium': 3, 'high': 4}  # 예시
    }
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("generate_fingerprint expects a pandas.DataFrame")

    fp: Dict[str, Any] = {}
    fp["n_rows"], fp["n_cols"] = df.shape
    total_cells = df.shape[0] * max(1, df.shape[1])
    fp["n_missing_total"] = int(df.isnull().sum().sum())
    fp["missing_ratio"] = float(fp["n_missing_total"] / total_cells) if total_cells > 0 else 0.0

    cols: Dict[str, Any] = {}
    low_card, med_card, high_card = 0, 0, 0
    for c in df.columns:
        try:
            stats = _summary_stats_for_series(df[c])
        except Exception as e:
            logger.warning("컬럼 통계 생성 실패(%s): %s", c, e)
            stats = {"dtype": str(df[c].dtype)}
        cols[c] = stats
        nunique = stats.get("n_unique", 0)
        # cardinality bucket
        if nunique <= 10:
            low_card += 1
        elif nunique <= 100:
            med_card += 1
        else:
            high_card += 1

    fp["columns"] = cols
    fp["cardinality_summary"] = {"low": low_card, "medium": med_card, "high": high_card}

    # 간단한 타깃 자동 감지: 'target' 컬럼 우선, 그 다음은 명칭/카디널리티 기반 추정
    target_col = None
    if detect_target:
        if "target" in df.columns:
            target_col = "target"
        else:
            # 컬럼 이름 규칙 탐색
            for candidate in ["label", "y", "class", "target_label"]:
                if candidate in df.columns:
                    target_col = candidate
                    break
        if target_col is None:
            # 범주형이면서 고유값이 적은 컬럼을 찾음 (예: 2~20개)
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    nunique = int(df[c].nunique(dropna=True))
                    if 2 <= nunique <= min(20, max(2, int(df.shape[0] * 0.2))):
                        target_col = c
                        break

    fp["target_column"] = target_col

    return fp
