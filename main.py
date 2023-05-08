# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create random data for predictive maintenance
n_samples = 1000
n_features = 10

# Create time series data
time_index = pd.date_range('2022-01-01', periods=n_samples, freq='H')
sensor_data = np.random.rand(n_samples, n_features)

# Create binary target indicating machine failure
target = np.zeros(n_samples, dtype=int)
failure_idx = np.random.randint(100, n_samples, size=5)
target[failure_idx] = 1

# Combine data into DataFrame
data = pd.DataFrame(sensor_data, index=time_index, columns=[f'Sensor {i+1}' for i in range(n_features)])
data['Target'] = target

# Split data into training and testing sets
X = data.drop('Target', axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
