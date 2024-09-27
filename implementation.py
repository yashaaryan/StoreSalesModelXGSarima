import pandas as pd

# Load datasets 
train_df = pd.read_csv('/content/train.csv')
stores_df = pd.read_csv('/content/stores.csv')
oil_df = pd.read_csv('/content/oil.csv')
holidays_events_df = pd.read_csv('/content/holidays_events.csv')
transactions_df = pd.read_csv('/content/transactions.csv')

train_df['date'] = pd.to_datetime(train_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])
holidays_events_df['date'] = pd.to_datetime(holidays_events_df['date'])
transactions_df['date'] = pd.to_datetime(transactions_df['date'])

# Merge train data 
train_df = train_df.merge(stores_df, on='store_nbr', how='left')

train_df = train_df.merge(oil_df, on='date', how='left')

train_df = train_df.merge(holidays_events_df, on='date', how='left')

train_df = train_df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

# Fill missing values if needed 
train_df['dcoilwtico'] = train_df['dcoilwtico'].fillna(method='ffill')
train_df['type_y'] = train_df['type_y'].fillna('not-holiday')


# Feature engineering
# For example, extracting the day of the week from the date
train_df['day_of_week'] = train_df['date'].dt.dayofweek

train_df['lagged_sales'] = train_df.groupby(['store_nbr', 'family'])['sales'].shift(1)


# Finalizing the training DataFrame
train_df = train_df.drop(columns=['transactions'])
train_df = train_df.drop(columns=['transferred', 'description', 'locale', 'locale_name','city','state','type_x'], errors='ignore')

train_df['dcoilwtico'] = train_df['dcoilwtico'].ffill()
train_df['lagged_sales'] = train_df['lagged_sales'].ffill()
nan_counts = train_df.isna().sum()

start_date = '2016-04-01'
end_date = '2016-05-31'

# Remove the affected data from the training DataFrame
train_df = train_df[(train_df['date'] < start_date) | (train_df['date'] > end_date)]

def is_payday(date):
    # Check if the date is the 15th or the last day of the month
    if date.day == 15 or (date.day == 1 and date != date + pd.offsets.MonthEnd(0)):
        return 1  # payday
    else:
        return 0  # not payday

train_df['payday'] = train_df['date'].apply(is_payday)
train_df = train_df[train_df['store_nbr'] == 1]


train_df.fillna(0, inplace=True)
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['is_weekend'] = train_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Saturday and Sunday


print(train_df)

import seaborn as sns
import matplotlib.pyplot as plt

# Create a correlation matrix
corr_matrix = train_df[['sales', 'onpromotion', 'dcoilwtico', 'day_of_week', 'lagged_sales', 'payday', 'is_weekend']].corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

from statsmodels.tsa.stattools import adfuller

# Perform ADF test on the target variable
subset = train_df['sales'].iloc[:10000] 
adf_result = adfuller(subset)

print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')


train_df['sales_seasonal_diff'] = train_df['sales'].diff(12).dropna()

# Perform ADF test on a smaller subset of the seasonal differenced data
adf_result_seasonal = adfuller(train_df['sales_seasonal_diff'].dropna().iloc[:10000])  # Adjust the number of rows as needed
print(f'ADF Statistic (after seasonal differencing): {adf_result_seasonal[0]}')
print(f'p-value (after seasonal differencing): {adf_result_seasonal[1]}')

train_df['date'] = pd.to_datetime(train_df['date'])  # Convert to datetime if not already done
train_df.index = pd.date_range(start='2013-01-01', periods=len(train_df), freq='D')

print(train_df)

target = 'sales'
exog_vars = ['onpromotion', 'dcoilwtico', 'day_of_week', 'payday', 'lagged_sales', 'is_weekend']

y = train_df[target]
X = train_df[exog_vars]  # Use a list of exogenous columns, not a tuple

# Split data into training and testing sets
train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]
X_train, X_test = X[:train_size], X[train_size:]

print("Target and exogenous features are ready for model training.")

# Assuming your 'date' column is already in datetime format, select 3 years of data (adjust as needed)


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Drop NaN values after differencing
sales_diff_subset = train_df['sales_seasonal_diff'].dropna()

# Plot ACF and PACF for the subset of data
plt.figure(figsize=(12, 6))
plot_acf(sales_diff_subset, lags=40)  
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(sales_diff_subset, lags=40)  
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Configure SARIMAX parameters

from statsmodels.tsa.statespace.sarimax import SARIMAX

p, d, q = 1, 0, 1
P, D, Q, S = 1, 1, 1, 12  # Seasonal components with a season length of 12 months

# Initialize SARIMAX model with exogenous variables
sarimax_model = SARIMAX(y_train, exog=X_train, order=(p, d, q),
                        seasonal_order=(P, D, Q, S))

# Fit the model
sarimax_results = sarimax_model.fit()

print(sarimax_results.summary())

# Forecast sales on the test set using exogenous variables
predictions = sarimax_results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1,
                                  exog=X_test)

print(predictions)

import matplotlib.pyplot as plt
# Plot actual vs predicted sales
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Sales')
plt.plot(y_test.index, predictions, label='Predicted Sales', color='red')
plt.legend()
plt.show()

import numpy as np

# Ensure y_test and predictions are non-negative
y_test = np.maximum(y_test, 0)
predictions = np.maximum(predictions, 0)

# Calculate RMSLE
rmsle = np.sqrt(np.mean((np.log1p(predictions) - np.log1p(y_test)) ** 2))
print(f"Root Mean Squared Logarithmic Error (RMSLE): {rmsle}")
