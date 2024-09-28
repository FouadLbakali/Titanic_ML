import pandas as pd
from src.preprocess import preprocess_data
from src.model import train_model, predict_model, save_submission

# Load the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Data preprocessing
X = preprocess_data(train).drop(columns=['Survived'])
y = train['Survived']
X_test = preprocess_data(test)

# Model training
model = train_model(X, y)

# Prediction on the test data
predictions = predict_model(model, X_test)

# Save predictions for submission
save_submission(predictions, test['PassengerId'])
