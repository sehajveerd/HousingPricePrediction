import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Loads the California housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Exploratory Data Analysis (EDA)
print(data.info())
print(data.describe())

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Spliting the data
X = data.drop('PRICE', axis=1)
y = data['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluateing the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Cross-Validation for the houses
scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores: {scores}')

# This will save the model
joblib.dump(model, 'housing_price_model.pkl')

# Now this executes it
loaded_model = joblib.load('housing_price_model.pkl')
new_predictions = loaded_model.predict(X_test)
