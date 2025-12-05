# Auto-Intelligent  
A Lightweight ML Model & Feature Recommendation Service

Auto-Intelligent is a lightweight web-based service designed to help beginners quickly understand their dataset, visualize key insights, and receive model/feature recommendations for machine learning tasks.  
This project is optimized for a 2-week university term project and focuses on simplicity, clarity, and educational value.

---

## 🚀 Project Overview

**Auto-Intelligent** provides:
- Simple CSV upload
- Automatic dataset summary
- Basic data visualizations
- Task selection (Classification / Regression)
- Recommended preprocessing steps
- Recommended machine learning models
- Suggested important features based on statistical heuristics
- Lightweight API + frontend interface

The system is built primarily in **Python**, using a minimal web layer to deliver a smooth and accessible UI.

---

## 🎯 Goals

- Lower the entry barrier for students learning machine learning  
- Provide clear explanations and simple model/feature suggestions  
- Demonstrate practical use of ML & visualization tools in a web application  
- Keep the implementation lightweight and feasible within a 2-week term project  

---

## 🛠️ Tech Stack

### Backend (Python)
- **FastAPI** — lightweight, async-friendly API server  
- **pandas** — dataset analysis  
- **scikit-learn** — preprocessing + baseline modeling  
- **matplotlib / seaborn** — quick visualization generation  
- **uvicorn** — backend server

### Frontend (Basic)
- **HTML/CSS/JavaScript**  
  - File upload  
  - Simple UI panels (select task, view summary, view recommendations)  
- (*Optional*) **Bootstrap** for quicker layout

---

## 📦 Features

### ✔️ 1. Dataset Upload
- Accepts CSV files  
- Automatically parses column types, missing values, and basic statistics  

### ✔️ 2. Data Visualization
Automatically generated:
- Histogram of numeric features  
- Correlation heatmap  
- Target distribution  
- Missing-value chart  

### ✔️ 3. Model Recommendation
Based on:
- Task type (classification / regression)  
- Dataset size  
- Feature types  
- Noise and imbalance detection  
- Simple heuristics  

Example output:
- “Classification task with mostly numeric data → Recommended: RandomForestClassifier”  
- “High-dimensional data → Lasso or Linear SVM recommended”  

### ✔️ 4. Feature Recommendation
Based on:
- Correlation (numeric targets)  
- Mutual information (categorical targets)  
- Variance / redundancy check  

Example output:
- “Top 5 useful features: radius_mean, texture_mean, …”

### ✔️ 5. Lightweight UI
- CSV upload button  
- Task type selector  
- Auto-generated result panels  
- Visualization previews  

---

## 🗂️ Project Structure (Suggested)

project-root/
│
├── modules/
│ ├── pipeline.py
│ ├── feature_engineering.py
│ ├── hpo.py
│ ├── explain.py
│ ├── eda.py
│ ├── ingestion.py
│ ├── io_utils.py
│ ├── model_search.py
│ ├── preprocessing.py
│ └── visualization.py
│
├── pages/
│ ├── 01_upload.py
│ ├── 02_overview.py
│ ├── 03_preprocessing.py
│ ├── 04_feature_engineering.py
│ ├── 05_modeling.py
│ ├── 06_model_selection.py
│ ├── 07_hpo.py
│ ├── 08_validation.py
│ └── 09_report.py
│
├── project_overview.md
└── README.md ← (this document)

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt
2. Run backend
bash
코드 복사
uvicorn backend.main:app --reload
3. Open frontend
Open frontend/index.html in a browser
(or serve it via any simple http server).

📘 Example Workflows
1) User uploads CSV
→ Backend computes summary + sends initial report

2) User selects task type (e.g., classification)
→ Backend returns recommended preprocessing + model list

3) User checks visualizations
→ Heatmap, distribution, missing values

4) User receives final recommendation bundle
→ Model candidates + top features + notes

📝 License
This project is licensed under the MIT License.

🙋 About This Project
This project was developed as a university term project to explore:

Practical machine learning workflows

Data visualization techniques

Lightweight web service integration

Automated insights and recommendations

Feel free to fork, expand, or improve the service.
---
