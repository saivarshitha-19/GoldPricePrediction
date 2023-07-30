import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate random data for demonstration
np.random.seed(42)
num_samples = 100
gold_prices = np.random.randint(1500, 2000, num_samples)
feature1 = np.random.rand(num_samples) * 100
feature2 = np.random.rand(num_samples) * 50

# Create a DataFrame with the data
data = pd.DataFrame({'GoldPrice': gold_prices, 'Feature1': feature1, 'Feature2': feature2})

# Split the data into features (X) and target variable (y)
X = data.drop('GoldPrice', axis=1)
y = data['GoldPrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Plot actual vs. predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Gold Price')
plt.ylabel('Predicted Gold Price')
plt.title('Gold Price Prediction')
plt.show()
