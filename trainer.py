import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from classifier import build_model
from config import *


def train_kfold(data, labels, n_splits=5):

    """
    Train the model using Stratified K-Fold cross-validation.

    Args:
      data       : np.array, shape (N, freq_bins, time_bins, channels)
      labels     : 1D array of ints (0/1), shape (N,)
      n_splits   : number of folds (default 5)

    Returns:
      fold_models   : list of trained keras Models, one per fold
      fold_histories: list of History objects, one per fold
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    fold_models = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # build & compile a fresh model for each fold
        model = build_model(input_shape=data.shape[1:])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True
        )

        # fit on this fold
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=2
        )

        # save the best model for this fold
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        model_path = f"models/fold{fold}-{timestamp}-RSSIModel.keras"
        model.save(model_path)
        print(f"Saved fold {fold} model to {model_path}")

        fold_models.append(model)
        fold_histories.append(history)

    return fold_models, fold_histories
