# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model 
from ml.model import compute_model_metrics
from sklearn import tree
import pandas as pd
import os
import pickle

# Add the necessary imports for the starter code.

# Add code to load in the data.
path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = parent_path + "/data/"
data_file = "census.csv"


# Remove whitespaces
data = pd.read_csv(data_path+data_file)
for column in data.columns:
    data = data.rename(columns={column : column.strip()})


print(data.head())

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
   test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)



# Train model.
model = train_model(X_train, y_train)

# Save Model, encoder + lb
pickle.dump(model, open('../model/model.pkl', 'wb'))
pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))
pickle.dump(lb, open('../model/lb.pkl', 'wb'))

# Calculate Model-Metrics
preds = model.predict(X_test)
precision, recall, fbeta = compute_model_metrics(y_test,preds)
print(precision)
print(recall)
print(fbeta)

# Calculate Model-Metrics Slicing
for feature in cat_features:
    # Value per feature
    for cls in data[feature].unique():
        data_sclice = data[data[feature] == cls]
        X_slice, y_sclice, encoder, lb = process_data(
        data_sclice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )        
        preds_sclice = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_sclice,preds_sclice)
        print(f"feature: {feature} value: {cls} precision: {precision}")
      



