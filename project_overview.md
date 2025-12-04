# Project Overview

## 1. Purpose of the Project
This project aims to develop a modular and extensible system that leverages AI-assisted development through ChatGPT Codex. The repository is structured so that Codex can easily read, modify, and extend files, enabling rapid prototyping, refactoring, documentation, and automation.

The primary goal:  
**Provide Codex with a clear understanding of the project direction, objectives, and technical constraints so that it can autonomously assist development tasks.**

---

## 2. High-Level Objectives
1. Maintain a clean, modular repository structure for scalable development.  
2. Enable Codex to:
   - Read files and propose improvements
   - Generate or refactor modules
   - Create tests, documentation, and architecture plans
   - Produce Pull Requests with complete diffs
3. Support iterative development through automated feedback and version control.
4. Ensure long-term extensibility for future features or components.

---

## 3. Feature Scope (Current Phase)
Current development phase focuses on:
- Establishing core architecture and conventions
- Preparing the project for Codex-driven workflows
- Creating initial modules, pipelines, or utilities (to be expanded)
- Setting up reproducible environments and automated processes where needed

Future expansions may include:
- Additional modules
- Data pipelines
- Application-level integrations
- Testing suites
- CI/CD automation with GitHub Actions

---

## 4. Repository Structure (Initial Draft)

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
├── project_overview.md ← (this document)
└── README.md


This structure gives Codex a clear, consistent framework for exploring and modifying project components.

---

## 5. Technical Requirements
- **Python 3.10+**
- **Streamlit** for UI pages
- **scikit-learn / numpy / pandas** for ML components
- Clean, reproducible scripts with minimal external dependencies

Codex should ensure:
- Code remains modular and easy to extend
- Functions include docstrings
- No hidden runtime dependencies
- Compatibility across environments

---

## 6. Development Workflow with Codex
Codex is expected to support the following workflow:

1. **File Reading & Understanding**
   - Codex analyzes modules, pages, and architecture.

2. **Task Execution**
   - Implement new features
   - Refactor existing code
   - Fix bugs
   - Improve documentation
   - Generate unit tests

3. **Change Proposals**
   - Produce patch/diff outputs
   - Provide explanations for architectural decisions

4. **Pull Request Automation**
   - When requested, Codex prepares branches and PRs with summaries.

5. **Continuous Expansion**
   - Additional modules or components can be created by Codex as the project grows.

---

## 7. Future Direction
- Extend the project with more advanced ML pipelines
- Improve automation for preprocessing, tuning, and evaluation
- Add interactive dashboard capabilities
- Optimize performance for large datasets
- Add integration tests and CI/CD automation

Codex should treat this document as the authoritative project description and use it as a reference when generating or modifying any part of the repository.

---

## 8. Instructions for Codex
Whenever Codex works on this repository:
- Always refer back to this overview when deciding on architecture.
- Maintain consistency in naming conventions and file structure.
- Ask for clarification only when necessary.
- Prefer modular and reusable implementations.
- Provide diffs for all file modifications.
- Avoid breaking existing interfaces without explanation.

