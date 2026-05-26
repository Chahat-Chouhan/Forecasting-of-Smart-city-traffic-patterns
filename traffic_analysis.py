"""
Project: Smart City Traffic Pattern Forecasting
### Organization: UniConverge Technologies Private Limited (UCT) & Upskill Campus
### Role: Data Science & Machine Learning Intern

**Objective:** To analyze and predict traffic patterns across 4 major junctions in the city. 
This notebook handles Step 1: Initializing the project environment, importing required tools, 
loading the dataset, and performing an initial structure inspection.
"""

import pandas as pd
import holidays
import matplotlib.pyplot as plt # Library for creating the chart canvas
import seaborn as sns # Library for making the chart look modern and beautiful
from sklearn.model_selection import train_test_split # Function to split data into training and testing sets
from sklearn.ensemble import RandomForestRegressor  # Import the AI Model
from sklearn.metrics import mean_absolute_error, r2_score  # Import evaluation metrics

print("STARTING TRAFFIC PATTERNS FORECASTING...")

# =====================================================================================================
#                                              DATA LOADING
# =====================================================================================================

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print("--- DATA LOADED COMPLETED ---")

print(f"Total rows loaded: {len(train_data)}\n")

# Check column names, missing values, and data types
print(train_data.info())

# Check the exact number of missing values in each column
missing_counts = train_data.isnull().sum()
print("\n--- MISSING VALUES PER COLUMN---")
print(missing_counts, "\n")

# ==========================================
#  FEATURE ENGINEERING (TRAIN DATA)
# ==========================================

# Setup holiday calendar (Change 'IN' to your target country code if needed)
local_holidays = holidays.country_holidays('IN') # This will automatically include all public holidays for the specified country

# Convert text timestamps to official datetime objects
train_data['DateTime'] = pd.to_datetime(train_data['DateTime'])

# Extract time features for the machine learning model
train_data['Hour'] = train_data['DateTime'].dt.hour # Captures hourly rush periods (0-23)
train_data['Day'] = train_data['DateTime'].dt.day # Captures daily patterns (1-31) - Identifies monthly trends (1-31)
train_data['Month'] = train_data['DateTime'].dt.month # Captures seasonal patterns (1-12)
train_data['Year'] = train_data['DateTime'].dt.year # Captures yearly trends (e.g., 2020, 2021) -Tracks long-term traffic growth

# Distinguishes weekdays from weekends (0=Mon, 6=Sun)
train_data['DayOfWeek'] = train_data['DateTime'].dt.dayofweek # Captures weekly patterns (0-6) - Identifies weekday vs weekend traffic differences

# <-- NEW HOLIDAY FEATURE -->
# Extracts the date and checks if it exists in our holiday calendar. Returns 1 for Yes, 0 for No.
train_data['Is_Holiday'] = train_data['DateTime'].dt.date.apply(lambda d: 1 if d in local_holidays else 0)


# ==========================================
# FEATURE ENGINEERING (TEST DATA)
# ==========================================
test_data['DateTime'] = pd.to_datetime(test_data['DateTime'])
test_data['Hour'] = test_data['DateTime'].dt.hour
test_data['Day'] = test_data['DateTime'].dt.day
test_data['Month'] = test_data['DateTime'].dt.month
test_data['Year'] = test_data['DateTime'].dt.year
test_data['DayOfWeek'] = test_data['DateTime'].dt.dayofweek

test_data['Is_Holiday'] = test_data['DateTime'].dt.date.apply(lambda d: 1 if d in local_holidays else 0)

# Verify the feature engineering columns were added correctly to the DataFrame
print("\n--- TRAFFIC DATA PROCESS COMPLETED ---")
print(train_data.head())

# =============================================================================================
#                           DATA VISUALIZATION: HOURLY TRAFFIC TRENDS
# =============================================================================================

# Group the data by 'Hour' and calculate the mathematical average (mean) of vehicles for each hour
hourly_traffic = train_data.groupby('Hour')['Vehicles'].mean().reset_index()

# Create a blank chart canvas of size 10 inches wide by 6 inches tall
plt.figure(figsize=(10, 6))

# Draw a smooth line plot tracking vehicles across the hours
sns.lineplot(data=hourly_traffic, x='Hour', y='Vehicles', marker='o', color='royalblue', linewidth=2.5)

# Add clear, descriptive text labels to the graph
plt.title('Average Traffic Volume by Hour of the Day (Rush Hours Analysis)', fontsize=14, fontweight='bold')

plt.xlabel('Hour of the Daay (0-23)', fontsize=14)
plt.ylabel('Average Number of Vehicles', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7) # Add a subtle background grid for easier reading

# Save the finished chart as a high-quality image file in your folder
plt.savefig('hourly_traffic_trends.png', dpi=300)
plt.close() # Close the plot to free up memory

print("\n--- VISUALIZATION COMPLETED ---")
print("Success! Graph saved as 'hourly_traffic_trends.png' in the project folder.")

# =============================================================================================
#                                MACHINE LEARNING DATA PREPARATION
# =============================================================================================

# 1. Isolate the inputs (Features) from the answer key (Target)
feature_columns = ['Junction','Hour','Day','Month','Year','DayOfWeek','Is_Holiday'] # These are the clues the model will use to learn patterns
X = train_data[feature_columns]        # Clues for the model
y = train_data['Vehicles'] # The answer the model needs to learn to predict

# Diagnostic Check: Verify that NO datetime columns are leaking into X
print("\n VERIFYING MODEL INPUT DATA TYPES:")
print(X.dtypes)

# 2. Split into Training sets (80%) and Testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- MACHINE LEARNING DATA PREPARATION COMPLETED ---")

# 3. Print confirmation dimensions
print("--- DATA SPLITTING COMPLETED ---")
print(f"X_train (Training clues): {X_train.shape}")
print(f"y_train (Training Answers): {y_train.shape}")
print(f"X_test  (Testing Clues):  {X_test.shape}")
print(f"y_test  (Testing Answers):  {y_test.shape}")

# =====================================================================================================
#                                MODEL INITIALIZATION & TRAINING
# =====================================================================================================

print("\nTraining the AI model... (This might take a few seconds)")

# Initialize the Random Forest model
# n_estimators=50 means we are using 50 individual decision trees to vote on the answer
print("Training the compressed AI model... (Should take about 5 seconds)")
model = RandomForestRegressor(n_estimators=100,min_samples_split=5,random_state=42)

# Train the model by feeding it the training clues and the correct answers
model.fit(X_train, y_train)

print("SUCCESS: The Machine Learning model has finished training!")

# =======================================================================================================
#                                       MODEL EVALUATION
# =======================================================================================================

print("\nTesting the AI model on unseen data...")

# Let the model generate traffic predictions for the test set
y_pred = model.predict(X_test)

# Calculate metrics by comparing real values (y_test) vs predicted values (y_pred)
Mean_Absolute_Error = mean_absolute_error(y_test, y_pred) # Average error in vehicle count predictions
R2_Score = r2_score(y_test, y_pred) # Percentage of variance in traffic explained by the model

print("\n MODEL PERFORMANCE METRICS:")
print(f"Mean Absolute Error (MAE): {Mean_Absolute_Error:.2f} vehicles")
print(f"R-squared (R2) Score: {R2_Score:.4f} ({R2_Score*100:.1f}%)")


print("\n--- MODEL EVALUATION COMPLETED ---")


# ==========================================
# FUTURE TRAFFIC FORECASTING
# ==========================================

print("\n Generating future forecasts from the test sheet...")
X_future = test_data[feature_columns]   
future_predictions = model.predict(X_future)

# Append predictions back to the test sheet (rounded to whole cars)
test_data['Vehicles'] = future_predictions.round().astype(int)

# Drop any temporary feature columns to keep the file tidy
final_output = test_data[['DateTime', 'Junction', 'Vehicles','ID']]

# Save the final results to a new CSV file
final_output.to_csv('final_traffic_forecast_predictions.csv', index=False)

print("\n--- FUTURE TRAFFIC FORECASTING COMPLETED ---")
print("Success! Future traffic predictions saved as 'final_traffic_forecast_predictions.csv' in the project folder.")

# ==========================================
# SAVE THE TRAINED AI MODEL
# ==========================================

import joblib # Library for saving and loading machine learning models
joblib.dump(model, 'traffic_model.pkl' , compress=3) # Save the model with moderate compression to reduce file size
print("SUCCESS: Model successfully saved as 'traffic_model.pkl'!")




























