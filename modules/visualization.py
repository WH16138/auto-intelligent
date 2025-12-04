"""
modules/visualization.py

재사용 가능한 시각화 유틸:
- 안전한 Plotly/Figure 생성 함수들 (히스토그램, 박스, 상관, missing matrix, pair sample)
- small helper: fig_to_bytes (PNG) 및 dataframe preview 표 생성 지원
- Streamlit에서 바로 사용하기 쉬운 반환 타입을 목표로 함

Usage:
    from modules.visualization import hist_plotly, box_plotly, corr_heatmap_plotly, missing_matrix_plotly, fig_to_png_bytes
    fig = hist_plotly(df, "age")
    # in Streamlit: st.plotly_chart(fig, use_container_width=True)

Dependencies: plotly, matplotlib
"""
from typing import Optional, Tuple, List, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# -------------------------
# Plotly wrappers
# -------------------------
def hist_plotly(df: pd.DataFrame, col: str, nbins: int = 50, title: Optional[str] = None) -> go.Figure:
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")
    ser = df[col]
    if pd.api.types.is_numeric_dtype(ser):
        fig = px.histogram(df, x=col, nbins=nbins, title=title or f"{col} distribution")
    else:
        vc = ser.fillna("<<NA>>").value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", title=title or f"{col} value counts")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def box_plotly(df: pd.DataFrame, col: str, title: Optional[str] = None) -> go.Figure:
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")
    ser = df[col]
    if not pd.api.types.is_numeric_dtype(ser):
        raise ValueError("box_plotly applies to numeric columns only")
    fig = px.box(df, y=col, points="outliers", title=title or f"{col} boxplot")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def corr_heatmap_plotly(df: pd.DataFrame, numeric_only: bool = True, top_n: Optional[int] = 30, title: Optional[str] = None) -> go.Figure:
    if numeric_only:
        df_num = df.select_dtypes(include=["number"])
    else:
        df_num = df._get_numeric_data()
    if df_num.shape[1] == 0:
        raise ValueError("No numeric columns available for correlation.")
    corr = df_num.corr().fillna(0)
    if top_n is not None and corr.shape[0] > top_n:
        order = np.argsort((-np.abs(corr).sum(axis=0)).values)[:top_n]
        cols = corr.columns[order]
        corr = corr.loc[cols, cols]
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, title=title or "Correlation heatmap")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def missing_matrix_plotly(df: pd.DataFrame, max_cols: int = 80, sample_rows: int = 1000, title: Optional[str] = None) -> go.Figure:
    df_mask = df.isnull()
    miss_ratio = df_mask.mean().sort_values(ascending=False)
    cols = miss_ratio.index.tolist()
    if len(cols) > max_cols:
        cols = miss_ratio.head(max_cols).index.tolist()
    if df_mask.shape[0] > sample_rows:
        df_mask_sample = df_mask.sample(n=sample_rows, random_state=1)
    else:
        df_mask_sample = df_mask
    z = df_mask_sample[cols].astype(int).values
    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=[str(i) for i in df_mask_sample.index], colorscale=[[0, 'white'], [1, 'black']], showscale=False))
    fig.update_layout(title=title or "Missing values matrix (black = missing)", xaxis_tickangle=45, height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def pair_sample_plotly(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, sample_n: int = 500, title: Optional[str] = None) -> go.Figure:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns for pair sample plot.")
    sample = df[numeric_cols].dropna().sample(n=min(sample_n, max(10, len(df))), random_state=1)
    # use plotly scatter_matrix
    fig = px.scatter_matrix(sample, dimensions=numeric_cols, title=title or "Pairwise sample scatter matrix")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=800)
    return fig

# -------------------------
# Utility helpers
# -------------------------
def fig_to_png_bytes(fig: Any, dpi: int = 150) -> bytes:
    """
    plotly or matplotlib figure to PNG bytes.
    If input is plotly Figure, convert via to_image (requires kaleido); fallback to HTML snapshot.
    """
    # If plotly Figure
    try:
        if hasattr(fig, "to_image"):
            # prefer png via kaleido
            try:
                return fig.to_image(format="png", scale=2)
            except Exception as e:
                logger.debug("plotly to_image failed: %s", e)
    except Exception:
        pass

    # If matplotlib figure
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.debug("matplotlib save failed: %s", e)
        # fallback: return empty bytes
        return b""

def fig_to_base64_png(fig: Any, dpi: int = 150) -> str:
    b = fig_to_png_bytes(fig, dpi=dpi)
    if not b:
        return ""
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

def df_head_html(df: pd.DataFrame, n: int = 10) -> str:
    """DataFrame head를 HTML 테이블로 반환 (간단 리포트용)"""
    return df.head(n).to_html(classes="dataframe", index=False, escape=True)

# -------------------------
# Small convenience wrappers (combining eda + viz)
# -------------------------
def quick_numeric_overview_figs(df: pd.DataFrame, n_top: int = 4) -> List[Any]:
    """
    숫자형 컬럼 중 분산이 큰 것부터 n_top개 선택해 히스토그램과 박스플롯을 생성해서 리스트로 반환.
    """
    nums = df.select_dtypes(include=["number"])
    if nums.shape[1] == 0:
        return []
    var_order = nums.var().sort_values(ascending=False).index.tolist()[:n_top]
    figs = []
    for c in var_order:
        try:
            figs.append(hist_plotly(df, c))
            figs.append(box_plotly(df, c))
        except Exception:
            continue
    return figs
