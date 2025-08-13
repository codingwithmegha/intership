import pandas as pd
import numpy as np
'''import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error'''

# Load your dataset (replace 'your_dataset.csv' with your file)
df = pd.read_excel('train.xlsx',engine='openpyxl')

# Display the first few rows of the dataset
print(df())
'''
# Check for missing values
print(df.isnull().sum())

# Handle missing values (example: dropping rows with missing values)
df.dropna(inplace=True)

# Split the data into features (X) and target variable (y)
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the actual vs. predicted prices
plt.scatter(y_test, y_pred, color='gray')
plt.plot(y_test, y_test, color='red', linewidth=2)  # Line for perfect predictions
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
'''