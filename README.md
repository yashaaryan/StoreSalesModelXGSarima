# Sales Prediction App(SARIMA)

This is a simple web-based application for predicting sales using a SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model. The app is built with **FastAPI** on the backend, and the frontend is a basic HTML form that takes user input for various features and returns a predicted sales value.

## Features

- **Predict sales** using historical data and external factors like promotions, oil prices, day of the week, and payday.
- **Interactive web form** to input the necessary data and get the sales prediction in real-time.
- Uses a trained **SARIMAX model** for time series forecasting.

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, JavaScript (Vanilla)
- **Machine Learning**: SARIMAX model from `statsmodels` package
- **Deployment**: Can be deployed on Heroku or any server supporting FastAPI

## Requirements

- Python 3.7+
- FastAPI
- Pandas
- Statsmodels
- Uvicorn (for running the FastAPI server)

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yashaaryan/StoreSalesModelXGSarima.git

2. Create Viirtual environments
   ```bash
   python -m venv myenv
   source myenv/bin/activate
    # On Windows: myenv\Scripts\activate

4. Download the dataset from  https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
5. Replace the data file path in app.py according to your local file path
6. install dependencies
     ```bash
    pip install -r requirements.txt

8. Run fastapi server
   ```bash
   uvicorn app:app --reload
   
9. Open your browser and goto  http://127.0.0.1:8000/static/index.html or add /static/index.html to your localhost url.

