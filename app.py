from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pydantic import BaseModel


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

#File path 
train_df = pd.read_csv(r'C:\Users\Surface\Downloads\store-sales-time-series-forecasting\train.csv')
stores_df = pd.read_csv(r'c:\Users\Surface\Downloads\store-sales-time-series-forecasting\stores.csv')
oil_df = pd.read_csv(r'C:\Users\Surface\Downloads\store-sales-time-series-forecasting\oil.csv')
holidays_events_df = pd.read_csv(r'c:\Users\Surface\Downloads\store-sales-time-series-forecasting\holidays_events.csv')
transactions_df = pd.read_csv(r'C:\Users\Surface\Downloads\store-sales-time-series-forecasting\transactions.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])
holidays_events_df['date'] = pd.to_datetime(holidays_events_df['date'])
transactions_df['date'] = pd.to_datetime(transactions_df['date'])

train_df = train_df.merge(stores_df, on='store_nbr', how='left')
train_df = train_df.merge(oil_df, on='date', how='left')
train_df = train_df.merge(holidays_events_df, on='date', how='left')
train_df = train_df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

# Fill missing values and  feature engineering
train_df['dcoilwtico'] = train_df['dcoilwtico'].fillna(method='ffill')
train_df['day_of_week'] = train_df['date'].dt.dayofweek
train_df['lagged_sales'] = train_df.groupby(['store_nbr', 'family'])['sales'].shift(1)
train_df['payday'] = train_df['date'].apply(lambda x: 1 if x.day == 1 or x.day == 15 else 0)
train_df['is_weekend'] = train_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
train_df.fillna(0, inplace=True)

#removing noise
start_date = '2016-04-01'
end_date = '2016-05-31'
train_df = train_df[(train_df['date'] < start_date) | (train_df['date'] > end_date)]
train_df = train_df[train_df['store_nbr'] == 1]

train_df['date'] = pd.to_datetime(train_df['date'])  
train_df.index = pd.date_range(start='2013-01-01', periods=len(train_df), freq='D')

#train_df = train_df.head(1000)

target = 'sales'
exog_vars = ['onpromotion', 'dcoilwtico', 'day_of_week', 'payday', 'lagged_sales', 'is_weekend']

y = train_df[target]
X = train_df[exog_vars]

train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

#  SARIMAX model
p, d, q = 1, 0, 1
P, D, Q, S = 1, 1, 1, 12

sarimax_model = SARIMAX(y_train, exog=X_train, order=(p, d, q),
                        seasonal_order=(P, D, Q, S))
sarimax_results = sarimax_model.fit()

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html") as f:
        return f.read()

class SalesData(BaseModel):
    onpromotion: int
    dcoilwtico: float
    day_of_week: int
    payday: int
    lagged_sales: float
    is_weekend: int

@app.post("/predict/")
def predict_sales(data: SalesData):
    exog = pd.DataFrame({
        'onpromotion': [data.onpromotion],
        'dcoilwtico': [data.dcoilwtico],
        'day_of_week': [data.day_of_week],
        'payday': [data.payday],
        'lagged_sales': [data.lagged_sales],
        'is_weekend': [data.is_weekend]
    })

    # Predict sales using the SARIMAX model
    prediction = sarimax_results.predict(start=len(y_train), end=len(y_train), exog=exog)[0]

    return {"predicted_sales": prediction}