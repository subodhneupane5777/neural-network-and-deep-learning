import numpy as np
from tensorflow import keras


def data_normalization(raw_data, train_start, train_end):
    # Calculate mean and std of each feature based on training data only
    train_data = raw_data[train_start:train_end, :]
    means = np.mean(train_data, axis=0)
    stds = np.std(train_data, axis=0)

    # Normalize all data using training stats
    normalized_data = (raw_data - means) / stds

    return normalized_data


def make_inputs_and_targets(data, months, size, sampling):

    window_size = 336

    # Number of features in each time step
    n_features = data.shape[1]

    # Prepare arrays
    inputs = np.zeros((size, window_size, n_features))
    targets = np.zeros(size, dtype=int)

    # Total available time steps considering the window size
    max_start_idx = len(data) - window_size * sampling

    # Randomly select starting points for each input window
    start_indices = np.random.randint(0, max_start_idx, size=size)

    for i, start_idx in enumerate(start_indices):
        # Create input window with sampling
        for j in range(window_size):
            time_idx = start_idx + j * sampling
            inputs[i, j, :] = data[time_idx, :]

        # Set target as month at the midpoint of the window
        midpoint_idx = start_idx + (window_size * sampling) // 2
        targets[i] = months[midpoint_idx]

    return inputs, targets


def build_and_train_dense(train_inputs, train_targets, val_inputs, val_targets, filename):
    # Get input shape
    input_shape = train_inputs.shape[1:]

    # Build model
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="tanh"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(12, activation="softmax")  # 12 months (0-11)
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Define callbacks to save best model
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=filename,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_inputs, train_targets,
        epochs=10,
        validation_data=(val_inputs, val_targets),
        callbacks=callbacks
    )

    return history


def test_model(filename, test_inputs, test_targets):
    # Load the model
    model = keras.models.load_model(filename)

    # Evaluate the model
    _, test_acc = model.evaluate(test_inputs, test_targets, verbose=0)

    return test_acc


def confusion_matrix(filename, test_inputs, test_targets):

    # Load the model
    model = keras.models.load_model(filename)

    # Get predictions
    predictions = model.predict(test_inputs)
    predicted_classes = np.argmax(predictions, axis=1)

    # Create confusion matrix
    conf_matrix = np.zeros((12, 12), dtype=int)
    for true_class, pred_class in zip(test_targets, predicted_classes):
        conf_matrix[true_class, pred_class] += 1

    return conf_matrix