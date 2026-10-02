# 🩺 Chronic Kidney Disease (CKD) Prediction System

🌐 **Live App:**
https://tayyibahshaik-detecting-ckd-ckdapp-dakgja.streamlit.app

A machine learning project that predicts whether a patient has Chronic Kidney Disease (CKD) using medical test data, with an interactive web application accessible from any browser.

---

## 📌 Project Overview

This project focuses on early detection of Chronic Kidney Disease using machine learning techniques. It analyzes patient data such as age, blood pressure, lab test results, and medical history to predict whether CKD is likely or not.

An ensemble learning approach was also explored to improve prediction reliability. The trained models are integrated into a Streamlit web application for easy user interaction.

---

## 🎯 Objectives

* Perform data cleaning and preprocessing
* Analyze medical data using EDA
* Build and compare multiple ML models
* Improve prediction reliability using ensemble methods
* Deploy the model as an interactive web application

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📊 Dataset

* The dataset is included as `datset.csv`

* It contains clinical attributes such as:

  * Age, Blood Pressure
  * Specific Gravity, Albumin, Sugar
  * Blood Glucose, Blood Urea, Serum Creatinine
  * Sodium, Potassium, Hemoglobin
  * Medical history (Hypertension, Diabetes, CAD, etc.)

* Target column: `classification` (`ckd` or `notckd`)

### 🔹 Important Note

The dataset includes an `id` column, which is **not used as a predictive feature**. Only clinical attributes are used for model training.

* Missing values are handled using **KNN Imputer**
* Categorical values are encoded numerically

---

## ⚙️ Process

1. Data cleaning & preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature selection (21 clinical features used for prediction)
4. Model building:

   * Logistic Regression
   * Random Forest
   * SVM
   * KNN
   * Gaussian Naive Bayes
   * Voting Classifier (LR + RF + Decision Tree)
5. Model evaluation (80/20 train-test split)
6. Deployment using Streamlit

---

## 📈 Results

* Built multiple classification models for CKD prediction
* Compared model performance using accuracy metrics
* Random Forest achieved the best performance
* Voting Classifier demonstrated ensemble robustness
* Results visualized using bar charts and comparisons

> Note: Accuracy may vary slightly due to random train-test split.

---

## 🚀 How to Use the App

1. Open the live app
2. Wait for the app to load (models train at startup)
3. Select a model from the sidebar

   * Random Forest is selected by default
4. Enter patient details:

   * Patient Info
   * Blood Tests
   * Urine Tests
   * Medical History
5. Click **Predict**

### Output includes:

* CKD prediction (Detected / Not Detected)
* Probability score
* Risk visualization
* Model comparison

---

## 💻 Run Locally

```bash
git clone https://github.com/TayyibahShaik/Detecting_CKD.git
cd Detecting_CKD

pip install -r requirements.txt

streamlit run app.py
```

App runs at: http://localhost:8501

---

## 📁 Repository Structure

```
Detecting_CKD/
├── app.py
├── datset.csv
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

---

## ☁️ Deployment

The application is deployed on **Streamlit Community Cloud**.
Any updates pushed to the repository automatically redeploy the app.

---

## 🛠️ Troubleshooting

* If dataset not found → ensure `datset.csv` exists
* First load may be slow → models train initially
* App sleeping → wake it up and wait

---

## 🔭 Future Improvements

* Hyperparameter tuning & cross-validation
* Model explainability (feature importance per prediction)
* Integration with real-time medical data
* Multilingual support

> ⚕️ **Disclaimer:** This project is for educational purposes only. It is **not a medical diagnosis tool**. Always consult a qualified healthcare professional.


---

## 👩‍💻 Author: **Tayyibah Shaik**
