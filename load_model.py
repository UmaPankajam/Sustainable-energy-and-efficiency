import sklearn
import joblib

# Patch the missing internal class
from sklearn.compose._column_transformer import ColumnTransformer

class _RemainderColsList(list):
    pass

setattr(sklearn.compose._column_transformer, "_RemainderColsList", _RemainderColsList)

# Load your old model
model = joblib.load("your_model.pkl")

# Test if it works
print("Model loaded successfully!")
