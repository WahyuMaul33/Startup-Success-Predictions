# Startup Success Predictor: End-to-End AI Deployment

This project is a high-performance, full-stack AI application designed to predict the survival and growth potential of startups. By utilizing a **Gradient Boosting Classifier (88.1% Accuracy)**, the system analyzes business metrics such as funding history, milestones, and geographical data to provide actionable success probabilities.

---

## Application Preview

### Dashboard
Users can input startup metrics via the sidebar or main dashboard. The UI is built to handle complex numerical inputs and provide instant feedback.

![Main Dashboard](visualization/dashboard.png)
*Figure 1: Streamlit Frontend showing the input parameters.*

### Real-Time Analysis
Once the "Generate Survival Analysis" button is clicked, the system communicates with the FastAPI backend to calculate success probability.

![Analysis Results](visualization/results.png)
*Figure 2: Prediction results showing Status, Survival Probability, and Model Metadata.*

---

## How the Prediction Engine Works

The system follows a 4-step process to ensure high-accuracy predictions:

1.  **Data Capture**: The Streamlit frontend collects 10 primary inputs from the user.
2.  **Feature Alignment**: The `UMKMModelWrapper` expands these 10 inputs into the **36-feature signature** required by the model, filling categorical defaults (like industry and location) automatically.
3.  **Inference**: The **Gradient Boosting** engine processes the aligned feature vector.
4.  **Probability Scoring**: The model returns both a binary classification (Success/At Risk) and a raw probability score, providing more nuance than a simple "Yes/No."

## System Architecture
The project is built with a decoupled architecture to simulate real-world production environments:
* **Core Engine**: A custom Python wrapper that standardizes input data and handles inference via `joblib`.
* **Backend API**: Built with **FastAPI**, providing a high-performance RESTful endpoint for predictions.
* **Interactive Dashboard**: A modern **Streamlit** UI that allows users to simulate startup scenarios in real-time.

---

## Model Performance
After benchmarking multiple architectures, the **Gradient Boosting Classifier** was selected for production due to its superior balance of precision and recall.

| Model | Accuracy |
| :--- | :--- |
| **Gradient Boosting (Deployed)** | **87.50%** |
| Bagging Classifier | 88.10% |
| Neural Network | 84.52% |
| AdaBoost | 83.93% |

---

## Tech Stack
* **Language**: Python 3.12
* **ML Frameworks**: Scikit-Learn, XGBoost
* **API Framework**: FastAPI + Uvicorn
* **Frontend**: Streamlit
* **DevOps**: Git, Pydantic (Data Validation)

---

## Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/WahyuMaul33/Startup-Success-Predictions.git](https://github.com/WahyuMaul33/Startup-Success-Predictions.git)
cd Startup-Success-Predictions
pip install -r requirements.txt
