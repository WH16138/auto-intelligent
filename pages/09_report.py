# pages/09_report.py
"""
09 - Report (리포트 생성 및 다운로드)

세션에 축적된 정보로 자동 리포트를 생성합니다.
- HTML 리포트 (포함: 데이터 지문, EDA 요약, 전처리 요약, 피처 엔지니어링, 모델 비교, HPO, Validation)
- Markdown 리포트
- 옵션: artifacts 폴더(모델/파이프라인 등)를 함께 ZIP으로 묶어 다운로드

의존:
- modules.eda, modules.io_utils 등(있으면 사용)
"""
import streamlit as st
import os
import json
import zipfile
import datetime
from typing import Any, Dict, Optional

st.set_page_config(layout="wide")
st.title("9 - Report (자동 리포트 생성)")

# ---------- Helpers ----------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_save_text(path: str, text: str):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _to_html_table_from_df(df):
    try:
        return df.to_html(classes="table table-striped", index=False, escape=True)
    except Exception:
        return "<pre>표 생성 실패</pre>"

def _format_kv_section(d: Dict[str, Any], title: str) -> str:
    parts = [f"<h3>{title}</h3><dl>"]
    for k, v in d.items():
        try:
            parts.append(f"<dt><strong>{str(k)}</strong></dt><dd><pre>{json.dumps(v, ensure_ascii=False, indent=2)}</pre></dd>")
        except Exception:
            parts.append(f"<dt><strong>{str(k)}</strong></dt><dd>{str(v)}</dd>")
    parts.append("</dl>")
    return "\n".join(parts)

# ---------- Gather session info ----------
session = st.session_state

df = session.get("df")
df_pre = session.get("df_preprocessed")
fp = session.get("fingerprint")
preproc = session.get("preprocessing_pipeline")
fe_meta = session.get("feature_engineering_meta")
baselines_df = session.get("baselines_df")
baseline_models = session.get("baseline_models")
best_model = session.get("best_model") or session.get("final_model") or None
best_model_name = session.get("best_model_name")
hpo_res = session.get("hpo_result")
validation_res = session.get("validation_result") or session.get("evaluation_result") or None

# optional modules
try:
    from modules import eda as eda_mod
except Exception:
    eda_mod = None

try:
    from modules import io_utils
except Exception:
    io_utils = None

# default paths
_reports_dir = "reports"
_artifacts_dir = "artifacts"
_ensure_dir(_reports_dir)

# ---------- UI: filename input ----------
now = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
default_basename = f"report_{now}"
report_basename = st.text_input("리포트 파일명(확장자 제외)", value=default_basename)
html_path = os.path.join(_reports_dir, report_basename + ".html")
md_path = os.path.join(_reports_dir, report_basename + ".md")
zip_path = os.path.join(_reports_dir, report_basename + "_with_artifacts.zip")

st.markdown("### 리포트에 포함될 항목 (체크)")
include_data = st.checkbox("데이터 요약 / EDA", value=True)
include_preprocessing = st.checkbox("전처리 요약", value=True)
include_feature_engineering = st.checkbox("특징공학 요약", value=True)
include_model_selection = st.checkbox("모델 비교 (베이스라인)", value=True)
include_hpo = st.checkbox("하이퍼파라미터 튜닝 결과 (HPO)", value=True)
include_validation = st.checkbox("Validation 결과", value=True)
include_artifacts_zip = st.checkbox("Artifacts 폴더를 함께 ZIP (큰 용량 주의)", value=False)

st.markdown("---")

# ---------- Build report content ----------
def build_report_html() -> str:
    parts = []
    parts.append("<html><head><meta charset='utf-8'><title>AutoML Report</title>")
    # simple bootstrap CSS for nicer table
    parts.append("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.6.0/css/bootstrap.min.css">""")
    parts.append("</head><body class='container'>")
    parts.append(f"<h1>AutoML Report</h1><p>생성시각(UTC): {now}</p>")

    # 1. Data / EDA
    if include_data:
        parts.append("<hr><h2>1. Data & EDA</h2>")
        if fp:
            parts.append(_format_kv_section(fp, "Fingerprint"))
        else:
            parts.append("<p><em>Fingerprint 없음</em></p>")

        if eda_mod is not None and df is not None:
            try:
                # embed generated EDA HTML fragment if available
                eda_html = eda_mod.create_eda_report_html(df, fingerprint=fp, target_col=session.get("target_col"))
                parts.append("<h3>Auto EDA (요약 및 시각화)</h3>")
                # eda_html is a full html document; extract body inner if possible (very simple)
                # to be safe, append as iframe
                iframe_path = os.path.join(_reports_dir, report_basename + "_eda.html")
                _safe_save_text(iframe_path, eda_html)
                parts.append(f"<p>자세한 EDA는 아래 임베디드 파일을 확인하세요:</p>")
                parts.append(f"<iframe src='{os.path.basename(iframe_path)}' style='width:100%;height:600px;border:1px solid #ddd'></iframe>")
            except Exception as e:
                parts.append(f"<p><em>EDA 생성 실패: {e}</em></p>")
        else:
            if df is None:
                parts.append("<p><em>세션에 데이터가 없습니다.</em></p>")
            else:
                # fallback simple stats
                try:
                    parts.append("<h3>데이터 요약 (기본)</h3>")
                    parts.append("<pre>" + str(df.head(5).to_html(index=False)) + "</pre>")
                except Exception:
                    parts.append("<p><em>간단 요약 생성 불가</em></p>")

    # 2. Preprocessing
    if include_preprocessing:
        parts.append("<hr><h2>2. Preprocessing</h2>")
        if preproc is not None:
            try:
                # prefer io_utils snapshot or preproc repr
                if io_utils and hasattr(io_utils, "snapshot_artifacts"):
                    snap = io_utils.snapshot_artifacts(model_obj=None, preprocessor_obj=preproc, params=None, base_dir=_artifacts_dir, prefix=f"report_preproc_{now}")
                    parts.append("<p>전처리 파이프라인이 artifacts에 스냅샷으로 저장되었습니다.</p>")
                    parts.append("<pre>" + str(snap) + "</pre>")
                else:
                    parts.append("<pre>" + str(preproc)[:2000] + "</pre>")
            except Exception as e:
                parts.append(f"<p><em>전처리 정보 저장/표시는 실패했습니다: {e}</em></p>")
        else:
            parts.append("<p><em>세션에 전처리 파이프라인이 없습니다.</em></p>")

    # 3. Feature engineering
    if include_feature_engineering:
        parts.append("<hr><h2>3. Feature Engineering</h2>")
        if fe_meta:
            try:
                parts.append("<h3>파생 피처 메타</h3>")
                parts.append("<pre>" + json.dumps(fe_meta, ensure_ascii=False, indent=2) + "</pre>")
            except Exception:
                parts.append("<pre>메타 출력 실패</pre>")
        else:
            parts.append("<p><em>파생 피처 메타 없음</em></p>")

    # 4. Model selection / baselines
    if include_model_selection:
        parts.append("<hr><h2>4. Model Selection (Baselines)</h2>")
        if baselines_df is not None:
            try:
                parts.append("<h3>Baseline Results</h3>")
                parts.append(_to_html_table_from_df(baselines_df))
            except Exception as e:
                parts.append(f"<p><em>베이스라인 결과 출력 실패: {e}</em></p>")
        else:
            parts.append("<p><em>베이스라인 결과가 없습니다.</em></p>")

        if best_model_name:
            parts.append(f"<p>추천 모델: <strong>{best_model_name}</strong></p>")
        if baseline_models:
            parts.append(f"<p>저장된 baseline 모델 수: {len(baseline_models)}</p>")

    # 5. HPO
    if include_hpo:
        parts.append("<hr><h2>5. Hyperparameter Tuning (HPO)</h2>")
        if hpo_res:
            try:
                parts.append("<h3>HPO 결과 (best_params)</h3>")
                parts.append("<pre>" + json.dumps(hpo_res.get("best_params", {}), ensure_ascii=False, indent=2) + "</pre>")
                parts.append("<h4>Study summary</h4>")
                parts.append("<pre>" + json.dumps(hpo_res.get("study_summary", {}), ensure_ascii=False, indent=2) + "</pre>")
            except Exception as e:
                parts.append(f"<p><em>HPO 정보 출력 실패: {e}</em></p>")
        else:
            parts.append("<p><em>HPO 결과 없음</em></p>")

    # 6. Validation (metrics)
    if include_validation:
        parts.append("<hr><h2>6. Validation</h2>")
        if validation_res:
            try:
                parts.append("<h3>Validation Metrics</h3>")
                parts.append("<pre>" + json.dumps(validation_res, ensure_ascii=False, indent=2) + "</pre>")
            except Exception:
                parts.append("<p><em>Validation 결과 포맷 출력 실패</em></p>")
        else:
            parts.append("<p><em>Validation 결과가 세션에 없습니다. Validation 페이지에서 평가를 먼저 실행하세요.</em></p>")

    # 7. Artifacts summary
    parts.append("<hr><h2>7. Artifacts</h2>")
    if os.path.exists(_artifacts_dir) and any(os.scandir(_artifacts_dir)):
        parts.append(f"<p>'{_artifacts_dir}' 폴더에 저장된 파일과 폴더를 포함할 수 있습니다.</p>")
        parts.append("<ul>")
        for entry in os.listdir(_artifacts_dir):
            parts.append(f"<li>{entry}</li>")
        parts.append("</ul>")
    else:
        parts.append("<p><em>Artifacts 폴더가 비어있거나 존재하지 않습니다.</em></p>")

    parts.append("</body></html>")
    return "\n".join(parts)

def build_report_md() -> str:
    lines = []
    lines.append(f"# AutoML Report")
    lines.append(f"생성시각(UTC): {now}")
    lines.append("\n## 1. Data & EDA")
    if fp:
        lines.append("### Fingerprint")
        lines.append("```json")
        lines.append(json.dumps(fp, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append("_Fingerprint 없음_")
    if fe_meta:
        lines.append("\n## Feature engineering")
        lines.append("```json")
        lines.append(json.dumps(fe_meta, ensure_ascii=False, indent=2))
        lines.append("```")
    if baselines_df is not None:
        lines.append("\n## Baseline results")
        try:
            lines.append(baselines_df.to_markdown(index=False))
        except Exception:
            lines.append("<표 변환 실패>")
    if hpo_res:
        lines.append("\n## HPO best params")
        lines.append("```json")
        lines.append(json.dumps(hpo_res.get("best_params", {}), ensure_ascii=False, indent=2))
        lines.append("```")
    if validation_res:
        lines.append("\n## Validation results")
        lines.append("```json")
        lines.append(json.dumps(validation_res, ensure_ascii=False, indent=2))
        lines.append("```")
    return "\n".join(lines)

# ---------- Buttons: Generate / Save / Download ----------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    if st.button("HTML 리포트 생성 및 저장"):
        try:
            html_content = build_report_html()
            # try using io_utils.save_report_html if available
            if io_utils and hasattr(io_utils, "save_report_html"):
                io_utils.save_report_html(html_content, html_path)
            else:
                _safe_save_text(html_path, html_content)
            st.success(f"HTML 리포트가 생성되어 저장되었습니다: {html_path}")
            # show small preview
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    preview = f.read()[:4000]
                st.code(preview, language="html")
            except Exception:
                pass
        except Exception as e:
            st.error(f"HTML 리포트 생성 실패: {e}")

with col2:
    if st.button("Markdown 리포트 생성 및 저장"):
        try:
            md_text = build_report_md()
            _safe_save_text(md_path, md_text)
            st.success(f"Markdown 리포트가 저장되었습니다: {md_path}")
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    st.code(f.read()[:4000], language="markdown")
            except Exception:
                pass
        except Exception as e:
            st.error(f"Markdown 리포트 생성 실패: {e}")

with col3:
    if st.button("아티팩트 포함 ZIP 생성"):
        try:
            # create zip including report and artifacts (if requested)
            files_to_zip = []
            if os.path.exists(html_path):
                files_to_zip.append(html_path)
            if os.path.exists(md_path):
                files_to_zip.append(md_path)
            # include artifacts directory
            if include_artifacts_zip and os.path.exists(_artifacts_dir):
                # walk artifacts and add
                for root, _, files in os.walk(_artifacts_dir):
                    for fn in files:
                        files_to_zip.append(os.path.join(root, fn))
            # create zip
            if not files_to_zip:
                st.warning("ZIP에 포함할 파일이 없습니다. 먼저 리포트를 생성하세요.")
            else:
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in files_to_zip:
                        arcname = os.path.relpath(p)
                        zf.write(p, arcname=arcname)
                st.success(f"ZIP 파일 생성 완료: {zip_path}")
                # provide download
                with open(zip_path, "rb") as f:
                    st.download_button("ZIP 다운로드", data=f.read(), file_name=os.path.basename(zip_path), mime="application/zip")
        except Exception as e:
            st.error(f"ZIP 생성 실패: {e}")

st.markdown("---")
st.write("리포트 생성이 완료되면 파일을 검토하여 보고서(PDF 변환 등)나 제출용 압축본을 준비하세요.")
