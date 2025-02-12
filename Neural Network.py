import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


file_path = 'tmdb_5000_movies.csv'
data = pd.read_csv(file_path)


print(data.head())


features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
target = 'revenue'


data = data[features + [target]].dropna()


X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                     alpha=0.0001, batch_size='auto', learning_rate='adaptive',
                     max_iter=500, random_state=42)


model.fit(X_train, y_train)


predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")


output_path = 'tmdb_predictions.csv'
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
results.to_csv(output_path, index=False)
print(f"Predictions saved to '{output_path}'.")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(model.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.grid()
plt.show()
