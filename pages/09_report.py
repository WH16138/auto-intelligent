# pages/09_report.py
"""
Step 9 - Report

Generates a clean HTML/Markdown report summarizing the pipeline and offers direct downloads.
"""
import datetime
import json
import os
import zipfile
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.title("9 - Report")

# ------------------------- helpers -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_text(path: str, text: str):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _df_to_html(df: pd.DataFrame) -> str:
    try:
        return df.to_html(classes="table table-striped", index=False, escape=True)
    except Exception:
        return "<pre>표 생성 실패</pre>"


def _format_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def _active_df(session) -> Tuple[Optional[pd.DataFrame], bool]:
    if session.get("df_preprocessed") is not None:
        return session.get("df_preprocessed"), True
    return session.get("df"), False


# ------------------------- session data -------------------------
session = st.session_state
df, used_pre = _active_df(session)
target_col = session.get("target_col")
fp = session.get("fingerprint")
fe_meta = session.get("feature_engineering_meta")
baselines_df = session.get("baselines_df")
baseline_models = session.get("baseline_models")
best_model_name = session.get("best_model_name")
hpo_res = session.get("hpo_result")
validation_res = session.get("validation_result") or session.get("evaluation_result")

reports_dir = "reports"
artifacts_dir = "artifacts"
_ensure_dir(reports_dir)

now = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
default_basename = f"report_{now}"
report_basename = st.text_input("보고서 파일명(확장자 제외)", value=default_basename)
html_path = os.path.join(reports_dir, report_basename + ".html")
md_path = os.path.join(reports_dir, report_basename + ".md")
zip_path = os.path.join(reports_dir, report_basename + "_bundle.zip")

st.markdown("### 포함 항목 선택")
include_data = st.checkbox("데이터 개요 / Fingerprint", value=True)
include_fe = st.checkbox("특징공학 메타", value=True)
include_baseline = st.checkbox("베이스라인 결과", value=True)
include_hpo = st.checkbox("HPO 결과", value=True)
include_validation = st.checkbox("검증 결과", value=True)
include_artifacts = st.checkbox("artifacts 디렉터리 번들 포함", value=False)

st.markdown("---")

# ------------------------- builders -------------------------
CSS = """
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; padding: 24px; color: #1f2933; background: #f9fafb; }
h1, h2, h3 { color: #0f172a; }
section { margin-bottom: 32px; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
code, pre { background: #f1f5f9; padding: 8px; border-radius: 6px; display: block; white-space: pre-wrap; }
table { width: 100%; }
th, td { padding: 6px 10px; }
.meta { color: #4b5563; font-size: 0.95em; }
</style>
"""


def build_html() -> str:
    parts = ["<html><head><meta charset='utf-8'><title>Auto-Intelligent Report</title>", CSS, "</head><body>"]
    parts.append(f"<h1>Auto-Intelligent Report</h1>")
    parts.append(f"<p class='meta'>생성 시각 (UTC): {now}</p>")
    parts.append(f"<p class='meta'>데이터셋: {'전처리본' if used_pre else '원본'} | 타깃: {target_col or '미설정'}</p>")

    if include_data:
        parts.append("<section><h2>1. 데이터 개요</h2>")
        if df is not None:
            parts.append(f"<p>행/열: {df.shape[0]} × {df.shape[1]}</p>")
        if fp:
            parts.append("<h3>Fingerprint</h3>")
            parts.append(f"<pre>{_format_json(fp)}</pre>")
        else:
            parts.append("<p class='meta'>Fingerprint 없음</p>")
        parts.append("</section>")

    if include_fe:
        parts.append("<section><h2>2. 특징공학</h2>")
        if fe_meta:
            parts.append("<h3>생성 특징 메타</h3>")
            parts.append(f"<pre>{_format_json(fe_meta)}</pre>")
        else:
            parts.append("<p class='meta'>특징공학 메타 없음</p>")
        parts.append("</section>")

    if include_baseline:
        parts.append("<section><h2>3. 베이스라인 결과</h2>")
        if baselines_df is not None:
            parts.append(_df_to_html(baselines_df.round(4)))
        else:
            parts.append("<p class='meta'>베이스라인 미실행</p>")
        if best_model_name:
            parts.append(f"<p>추천 모델: <strong>{best_model_name}</strong></p>")
        if baseline_models:
            parts.append(f"<p>훈련된 베이스라인 수: {len(baseline_models)}</p>")
        parts.append("</section>")

    if include_hpo:
        parts.append("<section><h2>4. HPO 결과</h2>")
        if hpo_res:
            parts.append("<h3>best_params</h3>")
            parts.append(f"<pre>{_format_json(hpo_res.get('best_params', {}))}</pre>")
            if hpo_res.get("best_value") is not None:
                parts.append(f"<p>best_value: {hpo_res.get('best_value')}</p>")
            if hpo_res.get("study_summary"):
                parts.append("<h4>Study summary</h4>")
                parts.append(f"<pre>{_format_json(hpo_res.get('study_summary'))}</pre>")
        else:
            parts.append("<p class='meta'>HPO 미실행</p>")
        parts.append("</section>")

    if include_validation:
        parts.append("<section><h2>5. 검증 결과</h2>")
        if validation_res:
            parts.append(f"<pre>{_format_json(validation_res)}</pre>")
        else:
            parts.append("<p class='meta'>Validation 결과 없음</p>")
        parts.append("</section>")

    parts.append("<section><h2>6. Artifacts</h2>")
    if os.path.exists(artifacts_dir) and any(os.scandir(artifacts_dir)):
        parts.append(f"<p>경로: {artifacts_dir}</p><ul>")
        for root, _, files in os.walk(artifacts_dir):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), artifacts_dir)
                parts.append(f"<li>{rel}</li>")
        parts.append("</ul>")
    else:
        parts.append("<p class='meta'>artifacts 디렉터리가 비어 있습니다.</p>")
    parts.append("</section>")

    parts.append("</body></html>")
    return "\n".join(parts)


def build_md() -> str:
    lines = [
        "# Auto-Intelligent Report",
        f"- 생성 시각 (UTC): {now}",
        f"- 데이터셋: {'전처리본' if used_pre else '원본'}",
        f"- 타깃: {target_col or '미설정'}",
    ]
    if include_data:
        lines.append("\n## 1. 데이터 개요")
        if df is not None:
            lines.append(f"- 행/열: {df.shape[0]} × {df.shape[1]}")
        if fp:
            lines.append("### Fingerprint")
            lines.append("```json")
            lines.append(_format_json(fp))
            lines.append("```")
    if include_fe and fe_meta:
        lines.append("\n## 2. 특징공학")
        lines.append("```json")
        lines.append(_format_json(fe_meta))
        lines.append("```")
    if include_baseline and baselines_df is not None:
        lines.append("\n## 3. 베이스라인 결과")
        try:
            lines.append(baselines_df.round(4).to_markdown(index=False))
        except Exception:
            lines.append("_표 변환 실패_")
    if include_hpo and hpo_res:
        lines.append("\n## 4. HPO 결과")
        lines.append("```json")
        lines.append(_format_json(hpo_res.get("best_params", {})))
        lines.append("```")
    if include_validation and validation_res:
        lines.append("\n## 5. 검증 결과")
        lines.append("```json")
        lines.append(_format_json(validation_res))
        lines.append("```")
    return "\n".join(lines)


# ------------------------- actions -------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if st.button("HTML 보고서 생성"):
        try:
            html_content = build_html()
            _save_text(html_path, html_content)
            st.success(f"HTML 저장 완료: {html_path}")
            st.download_button("HTML 다운로드", data=html_content, file_name=os.path.basename(html_path), mime="text/html")
        except Exception as e:
            st.error(f"HTML 생성 실패: {e}")

with col2:
    if st.button("Markdown 보고서 생성"):
        try:
            md_content = build_md()
            _save_text(md_path, md_content)
            st.success(f"Markdown 저장 완료: {md_path}")
            st.download_button("Markdown 다운로드", data=md_content, file_name=os.path.basename(md_path), mime="text/markdown")
        except Exception as e:
            st.error(f"Markdown 생성 실패: {e}")

with col3:
    if st.button("ZIP 번들 생성"):
        try:
            files_to_zip = []
            if os.path.exists(html_path):
                files_to_zip.append(html_path)
            if os.path.exists(md_path):
                files_to_zip.append(md_path)
            if include_artifacts and os.path.exists(artifacts_dir):
                for root, _, files in os.walk(artifacts_dir):
                    for fn in files:
                        files_to_zip.append(os.path.join(root, fn))
            if not files_to_zip:
                st.warning("먼저 HTML/Markdown을 생성하거나 artifacts를 포함해 주세요.")
            else:
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in files_to_zip:
                        arcname = os.path.relpath(p, start=os.path.dirname(reports_dir))
                        zf.write(p, arcname=arcname)
                st.success(f"ZIP 저장 완료: {zip_path}")
                with open(zip_path, "rb") as f:
                    st.download_button("ZIP 다운로드", data=f.read(), file_name=os.path.basename(zip_path), mime="application/zip")
        except Exception as e:
            st.error(f"ZIP 생성 실패: {e}")

st.markdown("---")
st.info("보고서 생성 후 브라우저에서 바로 다운로드하거나 reports/ 폴더에서 확인할 수 있습니다.")
