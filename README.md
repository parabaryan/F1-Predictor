# 🏁 F1 Race Outcome Predictor
### Predicting Formula 1 race results with Machine Learning

Turn **24 years of Formula 1 history (2000–2024)** into race predictions.

This project uses historical **race results, qualifying sessions, and championship standings** to predict:

🏆 **Race Winners**  
🥈 **Podium Finishes (Top 3)**  
📈 **Championship Points**

Built with **scikit-learn + FastAPI**, and served through an interactive web interface.

---

## ✨ Demo

🎯 Select a **driver + circuit** → Generate predictions instantly.

Example:

```text
Driver: Max Verstappen
Circuit: Silverstone

🏆 Win Probability → High
🥈 Podium Probability → 92%
📈 Expected Points → 23
```

---

## 🚀 Features

✅ Predict race winners  
✅ Predict podium finishes  
✅ Estimate championship points  
✅ Train using 24 years of F1 data  
✅ Fast REST API with Swagger documentation  
✅ Interactive browser-based UI  
✅ Modular training pipeline for retraining on new seasons  

---

# 🏗 Architecture

```text
Jolpica API
     ↓
Data Collection
(fetch_data.py)
     ↓
Cleaning + Feature Engineering
     ↓
Model Training
(train.py)
     ↓
Saved Models (.joblib)
     ↓
FastAPI Backend
(api.py)
     ↓
Web Interface
(localhost:3000)
```

---

# 🧠 Machine Learning Pipeline

## Data Sources
- Race Results
- Qualifying Sessions
- Driver Standings
- Constructor Standings
- Circuit Information

Source:
Jolpica F1 API (Ergast replacement)

---

## Models Used

| Task | Model |
|------|------|
| 🏆 Race Winner | Random Forest Classifier |
| 🥈 Podium Prediction | Random Forest Classifier |
| 📈 Championship Points | Random Forest Regressor |

Libraries:
- scikit-learn
- pandas
- numpy
- joblib

---

# 📊 Model Performance

Performance measured on historical validation data.

| Prediction Task | Performance |
|----------------|------------|
| 🏆 Race Winner | **96% Accuracy** |
| 🥈 Podium Finish | **90% Accuracy** |
| 📈 Championship Points | **R² = 1.00** |

> ⚠️ Note: Results may indicate overfitting depending on evaluation methodology.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/f1-race-predictor.git

cd f1-race-predictor
```

Install dependencies:

```bash
pip install scikit-learn fastapi uvicorn pandas numpy requests rich joblib
```

---

# 📥 Step 1 — Collect Historical Data

```bash
python fetch_data.py
```

Downloads and prepares race data from **2000–2024**.

---

# 🧠 Step 2 — Train Models

```bash
python train.py
```

Outputs:

```text
models/
├── winner_model.joblib
├── podium_model.joblib
└── points_model.joblib
```

---

# 🌐 Step 3 — Launch Prediction API

```bash
uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Interactive Swagger API documentation.

---

# 🖥 Step 4 — Start Frontend

```bash
python -m http.server 3000
```

Open:

```text
http://localhost:3000
```

Select:
- Driver
- Circuit
- Generate prediction

---

# 📁 Project Structure

```text
f1-race-predictor/
│
├── fetch_data.py
├── train.py
├── api.py
├── models/
├── data/
├── ui/
├── requirements.txt
└── README.md
```

---

# 🔮 Future Improvements

- Add weather conditions
- Include tire strategies
- Driver head-to-head simulation
- Qualifying simulation
- XGBoost / LightGBM comparison
- Live race weekend predictions

---

# 🏎 Built For Formula 1 Fans + Machine Learning Enthusiasts
