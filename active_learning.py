import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.data_loader import load_data
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
K = 50                  # top uncertain samples per iteration
EPOCHS = 2              # fine-tuning epochs per iteration
LEARNING_RATE = 1e-5    # small learning rate
ITERATIONS = 5          # active learning iterations
FREEZE_LAYERS = True
BATCH_SIZE_FIT = 2
BATCH_SIZE_PREDICT = 2
MAX_LOAD = 2000         # CPU-friendly: max images to preload

# ----------------------------
# Load data generator
# ----------------------------
_, _, test_gen = load_data("data/raw")

# ----------------------------
# Preload images into memory (subset for CPU-friendly runs)
# ----------------------------
print("Loading images into memory (subset)...")
test_images, test_labels = [], []
loaded = 0

for x_batch, y_batch in test_gen:
    for i in range(len(x_batch)):
        if loaded >= MAX_LOAD:
            break
        test_images.append(x_batch[i])
        test_labels.append(y_batch[i])
        loaded += 1
    if loaded >= MAX_LOAD:
        break

test_images = np.array(test_images)
test_labels = np.array(test_labels)
num_classes = test_labels.shape[1] if len(test_labels.shape) > 1 else np.max(test_labels)+1

# Ensure one-hot labels
if len(test_labels.shape) == 1 or test_labels.shape[1] != num_classes:
    test_labels = to_categorical(test_labels, num_classes=num_classes)

print(f"Loaded {len(test_images)} images into memory.")

# ----------------------------
# Load model
# ----------------------------
model = load_model("models/best_model.h5")

if FREEZE_LAYERS:
    for layer in model.layers[:-2]:
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# Active Learning Loop
# ----------------------------
accuracy_list, f1_list = [], []

for iteration in range(1, ITERATIONS+1):
    print(f"\n===== Active Learning Iteration {iteration} =====")

    # Step 1: Predict uncertainty
    preds = model.predict(test_images, batch_size=BATCH_SIZE_PREDICT, verbose=0)
    uncertainty = 1 - np.max(preds, axis=1)

    # Step 2: Select top-K uncertain samples
    top_indices = np.argsort(uncertainty)[-K:]
    x_uncertain = test_images[top_indices]
    y_uncertain = test_labels[top_indices]
    print(f"Selected {len(x_uncertain)} uncertain samples for fine-tuning.")

    # Step 3: Fine-tune
    model.fit(x_uncertain, y_uncertain, epochs=EPOCHS, batch_size=BATCH_SIZE_FIT, verbose=1)

    # Step 4: Evaluate
    preds_all = model.predict(test_images, batch_size=BATCH_SIZE_PREDICT, verbose=0)
    y_true = np.argmax(test_labels, axis=1)
    y_pred = np.argmax(preds_all, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy_list.append(acc)
    f1_list.append(f1)

    print(f"Iteration {iteration} Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print(classification_report(y_true, y_pred))

# ----------------------------
# Save final model
# ----------------------------
model.save("models/active_learning_subset_model.h5")
print("\nActive Learning Completed and Model Saved!")

# ----------------------------
# Plot metrics
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(range(1, ITERATIONS+1), accuracy_list, marker='o', label='Accuracy')
plt.plot(range(1, ITERATIONS+1), f1_list, marker='s', label='Macro F1-score')
plt.title('Active Learning Performance')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.show()
