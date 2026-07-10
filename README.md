# Stock Price Movement Prediction Using Deep Learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://stock-price-movement-prediction-using-deep-learning.streamlit.app/)

[![Continuous Integration (CI)](https://github.com/askayusharma/Stock-Price-Movement-Prediction-Using-Deep-Learning/actions/workflows/python-ci.yml/badge.svg)](https://github.com/askayusharma/Stock-Price-Movement-Prediction-Using-Deep-Learning/actions/workflows/python-ci.yml)

## 📌 Project Overview
Predicting stock price movements is a challenging yet highly rewarding endeavor in financial modeling. This project leverages Deep Learning techniques to analyze historical financial data and predict future stock price movements. It includes a complete pipeline from data preprocessing to model training, culminating in an interactive dashboard for visualizing predictions and market trends.

## 🚀 Features
* **Deep Learning Architecture:** Utilizes advanced neural network models to capture complex patterns in time-series financial data.
* **Data Pipeline:** Automated cleaning and preprocessing of raw stock market data.
* **Interactive Dashboard:** A user-friendly interface (`dashboard.py`) to visualize historical data, moving averages, and model predictions in real-time.
* **Automated CI/CD:** Fully integrated GitHub Actions workflow for continuous integration, ensuring code quality and dependency management on every push.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Machine Learning:** [PyTorch / TensorFlow / Scikit-Learn] *(Edit based on your specific libraries)*
* **Data Processing:** Pandas, NumPy
* **Visualization:** [Streamlit / Dash / Matplotlib] *(Edit based on what dashboard.py uses)*
* **CI/CD:** GitHub Actions

## 📂 Repository Structure
```text
📦 Stock-Price-Movement-Prediction-Using-Deep-Learning
 ┣ 📂 .github/workflows   # GitHub Actions CI pipelines
 ┣ 📂 cleaned_data        # Processed and standardized datasets ready for training
 ┣ 📂 models              # Saved weights and architectures of trained deep learning models
 ┣ 📜 dashboard.py        # Main script to launch the interactive user interface
 ┣ 📜 requirements.txt    # Python dependencies and libraries
 ┗ 📜 README.md           # Project documentation
