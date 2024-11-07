# Predict the price of the Uber ride from a given pickup point to the agreed drop-off location.
# Perform following tasks:
# 1. Pre-process the dataset.
# 2. Identify outliers.
# 3. Check the correlation.
# 4. Implement linear regression and random forest regression models.
# 5. Evaluate the models and compare their respective scores like R2, RMSE, etc.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# 1. Pre-process the dataset.


# Load the dataset
df = pd.read_csv('uber_rides.csv')

# Drop rows with missing values (if any)
df = df.dropna()

# Convert pickup and drop-off times to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Extract date, hour, day of the week, etc.
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

# Calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                              df['dropoff_latitude'], df['dropoff_longitude'])

# Drop unnecessary columns (e.g., datetime columns)
df = df.drop(['pickup_datetime', 'pickup_latitude', 'pickup_longitude', 
              'dropoff_latitude', 'dropoff_longitude'], axis=1)


# 2. Identify outliers.


# Boxplot for price and distance to visualize outliers
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(df['fare_amount'])
plt.title("Fare Amount Boxplot")
plt.subplot(1, 2, 2)
sns.boxplot(df['distance_km'])
plt.title("Distance Boxplot")
plt.show()

# Removing outliers using Z-score
z = np.abs(stats.zscore(df[['fare_amount', 'distance_km']]))
df = df[(z < 3).all(axis=1)]

# 3. Check the correlation.

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 4. Implement linear regression and random forest regression models.

X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Random Forest Regression Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)



# 5. Evaluate the models and compare their respective scores like R2, RMSE, etc.

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"R² Score: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}\n")

# Evaluate Linear Regression
evaluate_model(y_test, y_pred_linear, "Linear Regression")

# Evaluate Random Forest Regression
evaluate_model(y_test, y_pred_rf, "Random Forest Regression")
