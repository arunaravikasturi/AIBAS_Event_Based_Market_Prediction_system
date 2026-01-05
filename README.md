# 💼 Future Job Salary Prediction System (ANN & OLS)

This repository contains an AI-based system for **predicting future job salaries** in high-growth industries using **Artificial Neural Networks (ANN)** and **Ordinary Least Squares (OLS)** regression.

The project was developed as part of the course  
**“M. Grum: Advanced AI-based Application Systems (AIBAS)”**  
at the **University of Potsdam**.

---

## 📌 Project Overview

The objective of this project is to design, train, evaluate, and deploy a **salary prediction system** for future job postings (year 2025).  
The system predicts the **annual salary (USD)** of a job based on structured attributes such as industry, location, company size, remote option, and required skills.

The project follows the **AI-CPS architecture**, including:
- Data scraping and preparation
- ANN model training (TensorFlow)
- OLS baseline model (Statsmodels)
- Model comparison and diagnostics
- Docker-based deployment (learningBase, activationBase, knowledgeBase, codeBase)

---

## 🧠 Models Implemented

### 1️⃣ Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Task: Regression (salary prediction)
- Optimized with early stopping and validation monitoring

### 2️⃣ Ordinary Least Squares (OLS)
- Framework: Statsmodels
- Task: Same regression problem as ANN
- Used as an interpretable baseline model

---

## 📊 Dataset Description

**File:** `future_jobs_dataset.csv`  
**Type:** Synthetic dataset (educational use only)  
**Rows:** 10,000  
**Year Modeled:** 2025  

### Key Features:
- `job_id` – Unique job identifier  
- `job_title` – Job role title  
- `industry` – AI, Blockchain, Green Tech, Quantum Computing  
- `location` – Job location (city)  
- `salary_usd` – Annual salary (target variable)  
- `skills_required` – Required skills list  
- `remote_option` – Remote work availability  
- `company_size` – Small / Medium / Large  
- `posting_date` – Job posting date  

The dataset was **cleaned, normalized, and split** into:
- `training_data.csv` (80%)
- `test_data.csv` (20%)
- `activation_data.csv` (single unseen entry)

---


---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- Statsmodels
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Docker & Docker Compose
- Jupyter Notebook

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

git clone <your-repository-url>
cd <repository-folder>

### 2️⃣ Train Models

Run ANN notebook: ANN_Salary_Model.ipynb

Run OLS notebook: OLS_Salary_Model.ipynb

### 3️⃣ Activation (Inference)

Use activation_data.csv

Run ANN and OLS activation notebooks

Output: predicted salary in USD

### 4️⃣ Docker Deployment
docker-compose -f docker-compose-ann.yml up
docker-compose -f docker-compose-ols.yml up

### 📈 Evaluation Metrics

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

R² Score

The ANN consistently outperforms OLS in predictive accuracy, while OLS provides better interpretability.

### 👥 Project Team

Shashank S

Aruna Ravi Kasturi Rama Mohan Babu

University of Potsdam
Faculty of Business, Information Systems
Chair of AI-based Application Systems

### 📜 License

This project is licensed under the AGPL-3.0 License,
as required by the course guidelines.

See the LICENSE file for details.

### 📌 Acknowledgments

This project was developed as part of the course
“M. Grum: Advanced AI-based Application Systems”
at the University of Potsdam.


---

### If you want, I can next:
- ✅ Adjust this README **exactly to match your Docker images**
- ✅ Add a **“Course Requirements Mapping” section**
- ✅ Review it from an **examiner’s perspective**

### Just tell me 👍
