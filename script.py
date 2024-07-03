#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the dataset
data = pd.read_csv('raza.csv')  # Replace with the actual file path

# Feature columns
features = ['Vibration', 'Temperature', 'Pressure', 'UsageHours']

# Target column
target = 'MaintenanceNeeded'

# Remove outliers using Z-score
z_scores = zscore(data[features])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train a Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions on the test set for each model
rf_predictions = rf_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)

print(f'Random Forest Model Accuracy: {rf_accuracy:.2f}')
print(f'Decision Tree Model Accuracy: {dt_accuracy:.2f}')
print(f'Gradient Boosting Model Accuracy: {gb_accuracy:.2f}')

# Input form for prediction
vibration_input = widgets.FloatText(value=0.5, description='Vibration:')
temperature_input = widgets.FloatText(value=40, description='Temperature:')
pressure_input = widgets.FloatText(value=100, description='Pressure:')
predict_button = widgets.Button(description='Predict Maintenance')
prediction_output = widgets.Output()

# Define the callback for the button click event
def on_button_click(b):
    with prediction_output:
        clear_output(wait=True)
        # Make a prediction using each trained model
        input_data = np.array([[vibration_input.value, temperature_input.value, pressure_input.value, 0]])

        # Ensure the input data is of the same shape as the training data
        input_data = input_data.reshape(1, -1)

        rf_prediction = rf_model.predict(input_data)[0]
        dt_prediction = dt_model.predict(input_data)[0]
        gb_prediction = gb_model.predict(input_data)[0]

        print(f'Random Forest Predicted MaintenanceNeeded: {rf_prediction}')
        print(f'Decision Tree Predicted MaintenanceNeeded: {dt_prediction}')
        print(f'Gradient Boosting Predicted MaintenanceNeeded: {gb_prediction}')


# Scatter plots for Vibration, Temperature, and Pressure against MaintenanceNeeded
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=data['Vibration'], y=data['MaintenanceNeeded'])
plt.title('Vibration vs MaintenanceNeeded')

plt.subplot(1, 3, 2)
sns.scatterplot(x=data['Temperature'], y=data['MaintenanceNeeded'])
plt.title('Temperature vs MaintenanceNeeded')

plt.subplot(1, 3, 3)
sns.scatterplot(x=data['Pressure'], y=data['MaintenanceNeeded'])
plt.title('Pressure vs MaintenanceNeeded')

plt.tight_layout()
plt.show()

# Display pair plots for visualization
sns.pairplot(data, hue='MaintenanceNeeded', palette='viridis', vars=['Vibration', 'Temperature', 'Pressure'])
plt.show()

# Attach the callback to the button
predict_button.on_click(on_button_click)

# Display widgets
display(widgets.HBox([vibration_input, temperature_input, pressure_input]))
display(predict_button)
display(prediction_output)


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the dataset
data = pd.read_csv('zaidii.csv')  # Replace with the actual file path

# Feature columns
features = ['Vibration', 'Temperature', 'Pressure']

# Target column
target = 'MaintenanceNeeded'

# Remove outliers using Z-score
z_scores = zscore(data[features])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Train a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Train a Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions on the test set for each model
rf_predictions = rf_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)

print(f'Random Forest Model Accuracy: {rf_accuracy:.2f}')
print(f'Decision Tree Model Accuracy: {dt_accuracy:.2f}')
print(f'Gradient Boosting Model Accuracy: {gb_accuracy:.2f}')

# Input form for prediction
vibration_input = widgets.FloatText(value=0.5, description='Vibration:')
temperature_input = widgets.FloatText(value=40, description='Temperature:')
pressure_input = widgets.FloatText(value=100, description='Pressure:')
predict_button = widgets.Button(description='Predict Maintenance')
prediction_output = widgets.Output()

# Define the callback for the button click event
def on_button_click(b):
    with prediction_output:
        clear_output(wait=True)
        # Make a prediction using each trained model
        input_data = np.array([[vibration_input.value, temperature_input.value, pressure_input.value]])

        # Ensure the input data is of the same shape as the training data
        input_data = input_data.reshape(1, -1)

        rf_prediction = rf_model.predict(input_data)[0]
        dt_prediction = dt_model.predict(input_data)[0]
        gb_prediction = gb_model.predict(input_data)[0]

        print(f'Random Forest Predicted MaintenanceNeeded: {rf_prediction}')
        print(f'Decision Tree Predicted MaintenanceNeeded: {dt_prediction}')
        print(f'Gradient Boosting Predicted MaintenanceNeeded: {gb_prediction}')

# Scatter plots for Vibration, Temperature, and Pressure against MaintenanceNeeded
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x=data['Vibration'], y=data['MaintenanceNeeded'])
plt.title('Vibration vs MaintenanceNeeded')

plt.subplot(1, 3, 2)
sns.scatterplot(x=data['Temperature'], y=data['MaintenanceNeeded'])
plt.title('Temperature vs MaintenanceNeeded')

plt.subplot(1, 3, 3)
sns.scatterplot(x=data['Pressure'], y=data['MaintenanceNeeded'])
plt.title('Pressure vs MaintenanceNeeded')

plt.tight_layout()
plt.show()

# Display pair plots for visualization
sns.pairplot(data, hue='MaintenanceNeeded', palette='viridis', vars=features)
plt.show()

# Attach the callback to the button
predict_button.on_click(on_button_click)

# Display widgets
display(widgets.HBox([vibration_input, temperature_input, pressure_input]))
display(predict_button)
display(prediction_output)


# In[ ]:




