import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("heart.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Split
X = df.drop("output", axis=1)
y = df["output"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=26
)

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔷 Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# 🔷 SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# 🔷 Predictions
lr_pred = lr_model.predict(X_test)
svm_pred = svm_model.predict(X_test)

# 🔷 Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
svm_acc = accuracy_score(y_test, svm_pred)
print("Logistic Regression Accuracy:", lr_acc)
print("SVM Accuracy:", svm_acc)

import json
with open('metrics.json', 'w') as f:
    json.dump({'lr_acc': lr_acc, 'svm_acc': svm_acc}, f)

# Save models
pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(svm_model, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Models saved successfully!")