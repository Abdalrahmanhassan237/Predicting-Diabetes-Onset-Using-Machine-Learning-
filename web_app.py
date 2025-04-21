import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

# Load the dataset
diabetes_df = pd.read_csv('Diabetes Dataset/diabetes.csv')

# Split the dataset into features (X) and target (y)
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=6)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search for hyperparameter tuning
rfc = RandomForestClassifier(n_estimators= 100 , max_depth= 4)
rfc.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test_selected)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)



# Define the web app
def main():
    st.title("Diabetes Prediction")
    st.write("Enter the following details to predict diabetes:")

    # Collect user input
    pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Plasma glucose concentration", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood pressure (mm Hg)", min_value=0, max_value=120, value=60)
    skin_thickness = st.number_input("Skin thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin level (mu U/ml)", min_value=0, max_value=800, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=20.0)
    diabetes_pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age (years)", min_value=0, max_value=100, value=30)

    # Scale and select features
    input_features = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
    input_features_scaled = scaler.transform(input_features)
    input_features_selected = selector.transform(input_features_scaled)

    # Make a prediction
    prediction = rfc.predict(input_features_selected)

    # Display the prediction
    if prediction[0] == 0:
        st.write("The person is not diabetic.")
    else:
        st.write("The person is diabetic.")

    # Display the accuracy score
    st.write("Model Accuracy:", accuracy)

   

if __name__ == '__main__':
    main()