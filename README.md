📈 Event-Based Market Predictor

(ANN & OLS — AI-CPS Architecture)

This repository contains an AI-based system for predicting financial market prices using Artificial Neural Networks (ANN) and Ordinary Least Squares (OLS) regression.

The project was developed for the course
“M. Grum: Advanced AI-based Application Systems (AIBAS)”
at the University of Potsdam.

🧭 Project Overview

The objective of this project is to design, train, evaluate, and deploy an event-driven market prediction system.
The system predicts the closing market price based on economic signals, volatility indicators, geopolitical risk, sentiment analysis, and macro-economic events.

The project follows the AI-CPS architecture, including:

Data scraping and preparation

Algorithmic data cleaning, normalization & outlier removal

ANN model training (TensorFlow)

OLS baseline model (Statsmodels)

Model comparison and diagnostics

Docker-based deployment
(learningBase, activationBase, knowledgeBase, codeBase)

🧠 Models Implemented
1️⃣ Artificial Neural Network (ANN)

Framework: TensorFlow / Keras

Task: Market price regression

Optimized using:

Early stopping

Learning-rate scheduling

Validation monitoring

2️⃣ Ordinary Least Squares (OLS)

Framework: Statsmodels

Provides an interpretable linear baseline

Used for comparison and diagnostics

📊 Dataset Description

File: Market_Trend_External.csv
Type: Financial & macro-economic time-series dataset
Rows: 24,000+
Target: Close_Price

Key Features:
| Feature                    | Description                    |
| -------------------------- | ------------------------------ |
| `Date`                     | Market trading date            |
| `Open_Price`               | Opening price                  |
| `Close_Price`              | Closing price *(target)*       |
| `High_Price`               | Daily high                     |
| `Low_Price`                | Daily low                      |
| `Volume`                   | Trading volume                 |
| `Daily_Return_Pct`         | Daily return percentage        |
| `Volatility_Range`         | Market volatility indicator    |
| `VIX_Close`                | Market fear index              |
| `Economic_News_Flag`       | Major economic event indicator |
| `Sentiment_Score`          | Market sentiment score         |
| `Federal_Rate_Change_Flag` | Interest rate change event     |
| `GeoPolitical_Risk_Score`  | Geopolitical risk indicator    |
| `Currency_Index`           | Global currency index          |


🧪 Data Preparation Pipeline (Subgoal 2)

The dataset is processed as follows:

Data loading

Data cleaning

Algorithmic outlier removal

Algorithmic normalization

Train-test split

training_data.csv → 80%

test_data.csv → 20%

Activation dataset creation

activation_data.csv → single unseen row

Final dataset saved as:

joint_data_collection.csv

All generated datasets are stored in /data.

🛠️ Technologies Used

Python 3.x

TensorFlow / Keras

Statsmodels

Pandas, NumPy

Scikit-learn

Matplotlib

Docker & Docker Compose

Jupyter Notebook

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/shashank237/AIBAS_Event_Based_Market_Predictor.git
cd AIBAS_Event_Based_Market_Predictor

2️⃣ Train Models

Run:

ANN_Market_Model.ipynb

OLS_Market_Model.ipynb

3️⃣ Activation (Inference)

Use:

activation_data.csv

Run activation notebooks to generate predictions.

4️⃣ Docker Deployment (AI-CPS System)
cd scenarios/apply_ann
docker compose up --build

📈 Evaluation Results
Model	RMSE	R²
ANN	0.001067	0.999978
OLS	0.004719	0.999568

The ANN achieves near-perfect predictive performance, while OLS provides strong interpretability and diagnostic insight.

🧩 AI-CPS Architecture

The system implements a complete AI-Cyber-Physical System:

LearningBase → Model training

KnowledgeBase → Model storage & versioning

ActivationBase → Real-time inference

CodeBase → System orchestration

All components communicate via Docker containers.

👥 Project Team

Shashank Sanjay Kalaskar

Aruna Ravi Kasturi Rama Mohan Babu

University of Potsdam
Faculty of Business, Information Systems
Chair of AI-based Application Systems

📜 License

Licensed under AGPL-3.0, in compliance with course requirements.
See the LICENSE file for details.

🙏 Acknowledgments

This project was developed as part of the course
“Advanced AI-based Application Systems (AIBAS)”
at the University of Potsdam.