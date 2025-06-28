from keras import layers, models


def build_model(input_shape):
    """
    Builds a CNN model for classification of spectrogram frames.

    This model consists of multiple convolutional blocks followed by dense layers,
    designed to process spectrogram-like 2D input data for binary classification.

    Args:
        input_shape (tuple): Shape of the input data in the format
            (freq_bins, time_bins, channels).

    Returns:
        keras.Model: A compiled Keras CNN model ready for training.
    """

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        # layers.SpatialDropout2D(0.2),  # spatial dropout to prevent overfitting
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        # layers.SpatialDropout2D(0.2),  # spatial dropout to prevent overfitting

        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 1)),  # downsample freq only

        # Block 2
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        # layers.SpatialDropout2D(0.2),  # spatial dropout to prevent overfitting

        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.SpatialDropout2D(0.2),  # spatial dropout to prevent overfitting
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 1)),

        # Block 3
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # adaptive spatial compression

        # Dense Head
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='sigmoid')
    ])
    # binary classification
    return model
