import pandas as pd

# Load datasets (assuming paths are correct and data is already loaded)
train_df = pd.read_csv('/content/train.csv')
stores_df = pd.read_csv('/content/stores.csv')
oil_df = pd.read_csv('/content/oil.csv')
holidays_events_df = pd.read_csv('/content/holidays_events.csv')
transactions_df = pd.read_csv('/content/transactions.csv')

# Convert 'date' columns to datetime format
train_df['date'] = pd.to_datetime(train_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])
holidays_events_df['date'] = pd.to_datetime(holidays_events_df['date'])
transactions_df['date'] = pd.to_datetime(transactions_df['date'])

# Merge train data with store metadata
train_df = train_df.merge(stores_df, on='store_nbr', how='left')

# Merge train data with oil prices
train_df = train_df.merge(oil_df, on='date', how='left')

# Merge train data with holidays/events
train_df = train_df.merge(holidays_events_df, on='date', how='left')

# Merge train data with transactions
train_df = train_df.merge(transactions_df, on=['date', 'store_nbr'], how='left')

# Fill missing values if needed (for example, filling NaN oil prices with the previous day's price)
train_df['dcoilwtico'] = train_df['dcoilwtico'].fillna(method='ffill')
train_df['type_y'] = train_df['type_y'].fillna('not-holiday')


# Feature engineering: Create additional features if necessary
# For example, extracting the day of the week from the date
train_df['day_of_week'] = train_df['date'].dt.dayofweek

# You can also create lag features for sales (e.g., previous day sales)
# This requires grouping by store and family to create the lagged values
train_df['lagged_sales'] = train_df.groupby(['store_nbr', 'family'])['sales'].shift(1)


# Finalizing the training DataFrame
# Drop columns that are not necessary for the model
train_df = train_df.drop(columns=['transactions'])
train_df = train_df.drop(columns=['transferred', 'description', 'locale', 'locale_name','city','state','type_x'], errors='ignore')

train_df['dcoilwtico'] = train_df['dcoilwtico'].ffill()
train_df['lagged_sales'] = train_df['lagged_sales'].ffill()
nan_counts = train_df.isna().sum()

start_date = '2016-04-01'
end_date = '2016-05-31'

# Remove the affected data from the training DataFrame
train_df = train_df[(train_df['date'] < start_date) | (train_df['date'] > end_date)]

# Create a function to identify paydays
def is_payday(date):
    # Check if the date is the 15th or the last day of the month
    if date.day == 15 or (date.day == 1 and date != date + pd.offsets.MonthEnd(0)):
        return 1  # payday
    else:
        return 0  # not payday

# Apply the function to create a new column
train_df['payday'] = train_df['date'].apply(is_payday)
train_df = train_df[train_df['store_nbr'] == 1]


train_df.fillna(0, inplace=True)
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['is_weekend'] = train_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # Saturday and Sunday


# Check the final DataFrame structure
print(train_df)
