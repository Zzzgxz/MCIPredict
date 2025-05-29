# 🧠 MCI Prediction and SHAP-based Explanation Dashboard

This project provides a web-based tool for predicting Mild Cognitive Impairment (MCI) risk using a Random Forest classifier trained on physical and laboratory indicators. The interface also presents an explanation for each prediction using SHAP (SHapley Additive exPlanations) force plots.

---

## 📌 Project Objectives

- Predict individual MCI risk using clinically interpretable features.
- Enable visual explanation of predictions via SHAP force plots.
- Provide a reproducible, deployable, and user-friendly web app.

---

## 🖥️ Web App Overview

The Streamlit interface allows users to:
- Input patient features such as education, grip strength, and blood indicators.
- Get a binary prediction of MCI risk (high/low).
- View a SHAP-based force plot explaining feature contributions.

---

## 🚀 How to Run the App

### Option 1: Run locally

1. Clone the repository:
    ```bash
    git clone https://github.com/yourname/mcipredict.git
    cd mcipredict
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the app:
    ```bash
    streamlit run 06webpage.py
    ```

Then open `http://localhost:8501` in your browser.

---

### Option 2: Run on Streamlit Cloud

- Push your code to a public GitHub repository.
- Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and deploy your repo.
- Set `06webpage.py` as the app entry point.

---

## ⚙️ Environment Setup

**Python version**: ≥ 3.8

**Required packages** (included in `requirements.txt`):

