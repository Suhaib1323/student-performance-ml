# 🎓 AI Student Performance Prediction System

An interactive Machine Learning dashboard that predicts final student grade (G3) using academic and behavioral features.

## 🚀 Project Overview

This project uses a supervised regression model trained on the UCI Student Performance Dataset to predict final exam scores.

The system includes:

- Data preprocessing & feature encoding
- Linear Regression & Random Forest comparison
- Model evaluation (R², MAE, RMSE)
- Feature importance visualization
- Interactive Streamlit dashboard
- What-If simulation analysis
- Real-time prediction gauge

---

## 🧠 Machine Learning Workflow

1. Data Cleaning
2. Feature Encoding
3. Train-Test Split
4. Model Training
5. Performance Evaluation
6. Model Saving (joblib)
7. Streamlit Deployment

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| R²     | 0.33  |
| MAE    | 2.96  |
| RMSE   | 3.70  |

---

## 🖥️ Dashboard Features

- Live Grade Prediction Gauge
- Feature Visualization
- What-If Study Time Simulation
- Real-time Prediction Updates
- Professional UI Design

---

## ⚙️ Tech Stack

- Python
- Pandas
- Scikit-Learn
- Plotly
- Streamlit
- Joblib

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
