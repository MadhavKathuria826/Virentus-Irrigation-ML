# 🌱 Virentus: Intelligent Irrigation Optimization System

A machine learning system for predicting **optimal irrigation water requirements** using environmental and agricultural data, with a focus on **efficiency, simulation, and real-world applicability**.

---

## 📌 Overview

Virentus is designed to optimize irrigation by replacing fixed watering schedules with **data-driven predictions**.

The system:

* Predicts required water (regression)
* Simulates irrigation strategies
* Quantifies water savings and efficiency

---

## 🧠 Key Results

* 🌿 **Best Model:** Random Forest Regressor
* 💧 **Water Savings:** ~1.43% vs naive fixed strategy
* 📊 Strong predictive performance (low error, stable trends)

---

## ⚙️ System Architecture

```id="m4h0re"
.
├── data_loading.py
├── preprocessing.py
├── feature_engineering.py
├── models.py
├── evaluation.py
├── simulation.py
├── utils.py
├── main.py
└── README.md
```

---

## 🔍 Methodology

### 1. Data Processing

* Environmental features:

  * temperature
  * humidity
  * rainfall
  * evapotranspiration
* Agricultural features:

  * soil type
  * crop type

---

### 2. Feature Engineering

* Interaction features (temperature × humidity)
* Normalized evapotranspiration
* Derived environmental relationships

---

### 3. Modeling

Trained:

* Linear Regression
* Random Forest Regressor

Target:

* **Water requirement (continuous value)**

---

### 4. Simulation

Compared:

* **Naive Strategy:** fixed water allocation
* **ML Strategy:** predicted water requirement

Evaluated:

* total water usage
* over/underwatering
* efficiency gains

---

## 📊 Results & Visualizations

### 📈 Predicted vs Actual Water Requirement

<img width="1280" height="960" alt="predicted_vs_actual" src="https://github.com/user-attachments/assets/be5388e7-2dc7-4aba-b355-5f458c7331f2" />


```markdown id="kq8p8q"
[Predicted vs Actual]
```

---

### 🌿 Feature Importance (Random Forest)

<img width="1600" height="1120" alt="feature_importance_random_forest" src="https://github.com/user-attachments/assets/3dcabf5a-a41f-4c95-98c1-b962a6abe78e" />


```markdown id="tq2m8z"
[Feature Importance]
```

---

### 💧 Water Usage Comparison (Naive vs ML)

<img width="1120" height="800" alt="water_usage_comparison" src="https://github.com/user-attachments/assets/bbbacb45-80d8-4dc4-aae7-f4b7cdbc63a0" />


```markdown id="93kqnl"
[Water Usage]
```

---

## 📈 Evaluation Metrics

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

Results are saved in:

```id="1t7j2l"
metrics_summary.csv
```

---

## 🚀 How to Run

### 1. Clone repository

```bash id="x56ayk"
git clone https://github.com/<MadhavKathuria826>/virentus-irrigation-ml.git
cd virentus-irrigation-ml
```

---

### 2. Install dependencies

```bash id="u3zjbi"
pip install numpy pandas scikit-learn matplotlib
```

---

### 3. Run pipeline

```bash id="l5t2aa"
python main.py
```

---

## 📦 Dataset

This project uses an irrigation dataset containing:

* temperature
* humidity
* rainfall
* soil type
* crop type
* evapotranspiration
* water requirement

👉 Add your dataset in the project directory before running.

---

## 💡 Key Insights

* ML-based irrigation reduces unnecessary water usage
* Environmental interactions significantly impact predictions
* Fixed schedules are inefficient compared to adaptive systems
* Even small improvements (~1–2%) scale significantly in agriculture

---

## 🎯 Future Improvements

* Time-series modeling (LSTM)
* Weather forecast integration
* IoT sensor deployment
* Real-time irrigation system

---

## 👨‍💻 Author

**Madhav Kathuria**
B.Tech CSE, South Asian University

---

## ⭐ If you found this useful

Star the repo ⭐ — it helps!
