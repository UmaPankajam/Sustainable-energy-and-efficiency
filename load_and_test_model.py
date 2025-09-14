# load_and_test_model.py

import sklearn
import joblib

# -------------------------------
# Step 1: Patch the missing internal class
# -------------------------------
from sklearn.compose._column_transformer import ColumnTransformer

class _RemainderColsList(list):
    pass

setattr(sklearn.compose._column_transformer, "_RemainderColsList", _RemainderColsList)

# -------------------------------
# Step 2: Load your old model
# -------------------------------
model_file = "xgb_model_pipeline_pickle.pkl"  # Replace with your actual model file name
try:
    model = joblib.load(model_file)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# -------------------------------
# Step 3: Test the model with sample input
# -------------------------------
# Replace this with your actual input features
X_new = [[5.0, 3.2, 1.2]]  

try:
    predictions = model.predict(X_new)
    print("Predictions:", predictions)
except Exception as e:
    print("❌ Error during prediction:", e)
