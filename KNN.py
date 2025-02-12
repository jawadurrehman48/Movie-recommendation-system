# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load and prepare data
data = pd.read_csv('tmdb_5000_movies.csv')
print(data.head())

# Select features and target
features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
target = 'revenue'
data = data[features + [target]].dropna()

# Split into X and y
X = data[features] 
y = data[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train KNN model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate errors
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Save predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
results.to_csv('tmdb_knn_predictions.csv', index=False)
print("Predictions saved to 'tmdb_knn_predictions.csv'")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs Predicted Revenue (KNN)')
plt.grid()
plt.show()
