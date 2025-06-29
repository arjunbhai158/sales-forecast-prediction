# sales_forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('sales.csv')

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

# Sort by date
df = df.sort_values('Order Date')

# Aggregate daily sales
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# Feature engineering
daily_sales['day'] = daily_sales['Order Date'].dt.day
daily_sales['month'] = daily_sales['Order Date'].dt.month
daily_sales['year'] = daily_sales['Order Date'].dt.year
daily_sales['weekday'] = daily_sales['Order Date'].dt.weekday

# Prepare input and output
X = daily_sales[['day', 'month', 'year', 'weekday']]
y = daily_sales['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(daily_sales['Order Date'].iloc[y_test.index], y_test, label='Actual Sales')
plt.plot(daily_sales['Order Date'].iloc[y_test.index], y_pred, label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast - Actual vs Predicted')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the model
joblib.dump(model, 'sales_forecast_model.pkl')
print("Model saved as sales_forecast_model.pkl")
