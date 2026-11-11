# train_model.py trains the model on the training data,
# checks it on the test data, 
# and saves the model as sign_model.joblib.
# A Random Forest (many small decision trees that vote),


import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = np.load("prepared_data.npz", allow_pickle=True) # allow_pickle=True is used to allow loading of object arrays, which is necessary for loading the labels (y_train and y_test) that are strings
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

# the model learns from the training set only
model = RandomForestClassifier(n_estimators=200, random_state=42) # n_estimators=200 means the model will create 200 decision trees, random_state=42 is used to ensure reproducibility of results
model.fit(X_train, y_train) # fit() is used to train the model on the training data (X_train and y_train)

# the exam: predictions on hands it has never seen
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print()
print(classification_report(y_test, pred))
print("Confusion matrix (rows = real letter, columns = predicted):")
print(model.classes_)
print(confusion_matrix(y_test, pred, labels=model.classes_))

joblib.dump(model, "sign_model.joblib")
print("Model saved as sign_model.joblib")
