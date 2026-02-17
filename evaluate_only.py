from tensorflow.keras.models import load_model
from src.data_loader import load_data
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# Load test data
_, _, test_data = load_data("data/raw")

# Load saved model
model = load_model("models/best_model.h5")

# Predict
print("Predicting on test data...")
predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred,
      target_names=list(test_data.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()
import os
os._exit(0)

