"""
modules/eda.py

EDA 유틸리티:
- plot_histogram(df, col)
- plot_box(df, col)
- plot_correlation_heatmap(df, numeric_only=True)
- plot_missing_matrix(df, max_cols=80)
- generate_eda_summary(df, fingerprint=None, target_col=None, top_k=5)
- create_eda_report_html(df, fingerprint=None, target_col=None)

모든 plot_* 함수는 plotly.graph_objs.Figure 를 반환합니다.
generate_eda_summary는 사람이 읽을 수 있는 문자열(요약) 및 구조화된 dict를 반환합니다.
create_eda_report_html는 간단한 HTML 문자열을 반환합니다.
"""
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import html
import io
import base64
import textwrap

# -------------------------
# Plot helpers (Plotly)
# -------------------------
def plot_histogram(df: pd.DataFrame, col: str, nbins: int = 50) -> go.Figure:
    """숫자/범주형에 따라 히스토그램 또는 막대그래프로 반환"""
    if col not in df.columns:
        raise ValueError(f"컬럼 없음: {col}")
    ser = df[col]
    if pd.api.types.is_numeric_dtype(ser):
        fig = px.histogram(df, x=col, nbins=nbins, title=f"{col} distribution")
    else:
        vc = ser.value_counts(dropna=False).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", title=f"{col} value counts")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_box(df: pd.DataFrame, col: str) -> go.Figure:
    """숫자형 컬럼의 박스플롯"""
    if col not in df.columns:
        raise ValueError(f"컬럼 없음: {col}")
    ser = df[col]
    if not pd.api.types.is_numeric_dtype(ser):
        raise ValueError("box plot은 숫자형 컬럼에만 적용됩니다.")
    fig = px.box(df, y=col, points="outliers", title=f"{col} boxplot")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_only: bool = True, top_n: Optional[int] = 30) -> go.Figure:
    """
    숫자형 컬럼 사이의 상관계수 heatmap을 반환.
    top_n 지정 시 절대값 기준으로 상위 top_n 피처만 포함(행렬 크기 제어).
    """
    if numeric_only:
        df_num = df.select_dtypes(include=["number"])
    else:
        df_num = df.copy()._get_numeric_data()
    if df_num.shape[1] == 0:
        raise ValueError("숫자형 컬럼이 없습니다.")
    corr = df_num.corr().fillna(0)
    if top_n is not None and corr.shape[0] > top_n:
        # select features with largest variance in correlation magnitude
        order = np.argsort((-np.abs(corr).sum(axis=0)).values)[:top_n]
        cols = corr.columns[order]
        corr = corr.loc[cols, cols]
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, title="Correlation heatmap")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_missing_matrix(df: pd.DataFrame, max_cols: int = 80) -> go.Figure:
    """
    결측 패턴을 시각화: 행은 샘플 일부(상위 1000), 열은 컬럼.
    너무 많은 열이면 상위 max_cols로 제한(결측 비율 기준).
    """
    df_mask = df.isnull()
    # order columns by missing ratio descending
    miss_ratio = df_mask.mean().sort_values(ascending=False)
    cols = miss_ratio.index.tolist()
    if len(cols) > max_cols:
        cols = miss_ratio.head(max_cols).index.tolist()
    # sample rows if too many
    if df_mask.shape[0] > 1000:
        df_mask_sample = df_mask.sample(n=1000, random_state=1)
    else:
        df_mask_sample = df_mask
    z = df_mask_sample[cols].astype(int).values  # 1 = missing
    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=[str(i) for i in df_mask_sample.index], colorscale=[[0, 'white'], [1, 'black']], showscale=False))
    fig.update_layout(title="Missing values matrix (black = missing)", xaxis_tickangle=45, height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# -------------------------
# Summary generator
# -------------------------
def _top_skewed_numeric(df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float]]:
    nums = df.select_dtypes(include=["number"])
    res = []
    for c in nums.columns:
        ser = nums[c].dropna()
        if ser.shape[0] < 3:
            continue
        skew = float(ser.skew())
        res.append((c, skew))
    res.sort(key=lambda x: -abs(x[1]))
    return res[:top_k]

def _top_correlated_pairs(df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, str, float]]:
    nums = df.select_dtypes(include=["number"])
    if nums.shape[1] < 2:
        return []
    corr = nums.corr().abs()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i,j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]

def generate_eda_summary(df: pd.DataFrame, fingerprint: Optional[Dict[str, Any]] = None, target_col: Optional[str] = None, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
    """
    자동 EDA 요약(자연어) + 구조화된 정보 반환.

    Returns:
        (text_summary, summary_dict)
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * max(1, n_cols)
    n_missing = int(df.isnull().sum().sum())
    missing_ratio = float(n_missing / total_cells) if total_cells > 0 else 0.0

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # target analysis
    target_info = None
    if target_col and target_col in df.columns:
        tv = df[target_col]
        try:
            vc = tv.value_counts(dropna=False)
            dist = vc.to_dict()
            n_classes = vc.shape[0]
            if pd.api.types.is_numeric_dtype(tv) and tv.nunique() > 20:
                problem_type = "regression"
            else:
                problem_type = "classification"
            target_info = {"n_classes": int(n_classes), "dist": dist, "problem_type": problem_type}
        except Exception:
            target_info = None

    top_skews = _top_skewed_numeric(df, top_k=top_k)
    top_corrs = _top_correlated_pairs(df, top_k=top_k)

    # top missing cols
    miss_by_col = df.isnull().sum().sort_values(ascending=False)
    top_missing = miss_by_col[miss_by_col > 0].head(top_k).to_dict()

    # generate text
    lines = []
    lines.append(f"Rows: {n_rows}, Columns: {n_cols}.")
    lines.append(f"Total missing cells: {n_missing} ({missing_ratio:.2%}).")
    if top_missing:
        lines.append(f"Top missing columns: {', '.join([f'{k}({v})' for k,v in top_missing.items()])}.")
    lines.append(f"Numeric columns: {len(numeric_cols)}, Categorical (object/bool): {len(cat_cols)}.")
    if top_skews:
        lines.append("Top skewed numeric features: " + ", ".join([f'{c} (skew={s:.2f})' for c,s in top_skews]) + ".")
    if top_corrs:
        lines.append("Top correlated pairs (abs corr): " + ", ".join([f'{a}-{b}({r:.2f})' for a,b,r in top_corrs]) + ".")

    if target_info:
        lines.append(f"Detected target column '{target_col}' as {target_info['problem_type']} with {target_info['n_classes']} unique values.")
        # show class imbalance summary if classification
        if target_info['problem_type'] == 'classification':
            # compute imbalance ratio
            vals = list(target_info['dist'].values())
            if len(vals) > 1:
                imbalance = max(vals)/max(1,min(vals))
                lines.append(f"Class imbalance observed (largest/smallest = {imbalance:.2f}).")
    else:
        # if fingerprint suggests a target, mention it
        if fingerprint and fingerprint.get("target_column"):
            lines.append(f"Fingerprint suggests candidate target column: {fingerprint.get('target_column')} (consider setting it).")

    text_summary = " ".join(lines)

    summary_dict = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_missing": n_missing,
        "missing_ratio": missing_ratio,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(cat_cols),
        "top_missing": top_missing,
        "top_skewed": [{ "feature": c, "skew": s } for c,s in top_skews],
        "top_correlated_pairs": [{ "f1": a, "f2": b, "corr": r } for a,b,r in top_corrs],
        "target_info": target_info,
    }

    return text_summary, summary_dict

# -------------------------
# Report HTML generator
# -------------------------
def _fig_to_html_div(fig: go.Figure, div_id: Optional[str] = None) -> str:
    """plotly figure를 standalone HTML div로 변환 (스트립트 포함)"""
    # Use plotly.io.to_html for reliable embedding
    try:
        import plotly.io as pio
        html_fragment = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)
    except Exception:
        # fallback: very simple representation
        html_fragment = f"<div><pre>{html.escape(repr(fig))}</pre></div>"
    return html_fragment

def create_eda_report_html(df: pd.DataFrame, fingerprint: Optional[Dict[str, Any]] = None, target_col: Optional[str] = None) -> str:
    """
    간단한 HTML 리포트 생성. 내부적으로 plotly html fragment를 포함한다.
    결과는 modules.io_utils.save_report_html 로 파일로 저장 가능.
    """
    text_summary, summary_struct = generate_eda_summary(df, fingerprint=fingerprint, target_col=target_col)
    # create a few figures
    figs = []
    # histogram for first numeric column if exists
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    cats = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if nums:
        try:
            figs.append(plot_histogram(df, nums[0]))
        except Exception:
            pass
    if cats:
        try:
            figs.append(plot_histogram(df, cats[0]))
        except Exception:
            pass
    try:
        figs.append(plot_correlation_heatmap(df))
    except Exception:
        pass
    try:
        figs.append(plot_missing_matrix(df))
    except Exception:
        pass

    # assemble html
    html_parts = []
    html_parts.append("<html><head>")
    # include plotly.js once
    html_parts.append("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")
    html_parts.append("<meta charset='utf-8'/>")
    html_parts.append(f"<title>EDA Report</title></head><body>")
    html_parts.append(f"<h1>EDA Report</h1>")
    html_parts.append(f"<h2>Summary</h2><p>{html.escape(text_summary)}</p>")

    if fingerprint:
        html_parts.append("<h3>Fingerprint</h3><pre>")
        html_parts.append(html.escape(str(fingerprint)))
        html_parts.append("</pre>")

    for fig in figs:
        html_parts.append(_fig_to_html_div(fig))

    html_parts.append("</body></html>")
    return "\n".join(html_parts)
