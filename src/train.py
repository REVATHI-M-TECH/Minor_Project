import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.data_loader import load_data
from src.model import build_model

def train():

    base_path = "data/raw"

    train_data, val_data, test_data = load_data(base_path)

    num_classes = len(train_data.class_indices)

    model = build_model(num_classes)

    os.makedirs("models", exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            "models/best_model.h5",
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    history = model.fit(
        train_data,
        epochs=5,   # keep small for CPU
        validation_data=val_data,
        callbacks=callbacks
    )

    model.save("models/final_model.h5")

    return model, history, test_data
