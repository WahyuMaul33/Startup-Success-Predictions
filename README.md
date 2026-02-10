# Startup Success Predictor: End-to-End AI Deployment

This project is a high-performance, full-stack AI application designed to predict the survival and growth potential of startups. By utilizing a **Gradient Boosting Classifier (87.5% Accuracy)**, the system analyzes business metrics—such as funding history, milestones, and geographical data—to provide actionable success probabilities.

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
| Bagging Classifier | 87.50% |
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
