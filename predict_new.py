import pickle
import pandas as pd

# Load model
with open("model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define test cases — matching the trained model features
test_cases = pd.DataFrame([
    {
        'age': 45,
        'workclass': 'Private',
        'educational-num': 13,  # <-- FIXED COLUMN NAME
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'gender': 'Male',
        'capital-gain': 5000,
        'capital-loss': 0,
        'hours-per-week': 60,
        'native-country': 'United-States'
    },
    {
        'age': 23,
        'workclass': 'Private',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Sales',
        'gender': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 30,
        'native-country': 'United-States'
    },
    {
        'age': 36,
        'workclass': 'State-gov',
        'education-num': 11,
        'marital-status': 'Divorced',
        'occupation': 'Tech-support',
        'gender': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'India'
    },
    {
        'age': 50,
        'workclass': 'Self-emp-not-inc',
        'education-num': 14,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'gender': 'Male',
        'capital-gain': 10000,
        'capital-loss': 0,
        'hours-per-week': 55,
        'native-country': 'United-States'
    },
    {
        'age': 19,
        'workclass': 'Private',
        'education-num': 7,
        'marital-status': 'Never-married',
        'occupation': 'Handlers-cleaners',
        'gender': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 20,
        'native-country': 'United-States'
    }
])

# Make predictions
predictions = model.predict(test_cases)

# Display results
for i, pred in enumerate(predictions):
    print(f"🧪 Test Case {i}: Predicted Income Class: {'>50K' if pred == 1 else '<=50K'}")
