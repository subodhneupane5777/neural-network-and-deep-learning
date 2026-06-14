import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs):
    # Normalize the inputs
    cifar_tr_inputs = cifar_tr_inputs.astype("float32") / 255

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(cifar_tr_inputs, cifar_tr_labels, batch_size=batch_size, epochs=epochs, verbose=1)


def load_and_refine(filename, training_inputs, training_labels, batch_size, epochs):
    # Load the pre-trained model
    pretrained_model = keras.models.load_model(filename)

    # Create a new model with the pre-trained layers (excluding the last layer)
    new_model = keras.Sequential(pretrained_model.layers[:-1])

    # Freeze the weights of the hidden layers
    for layer in new_model.layers:
        layer.trainable = False

    # Add a new output layer
    new_model.add(layers.Dense(10, activation='softmax'))

    # Compile the new model
    new_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Reshape the inputs to match the expected input shape
    training_inputs = np.expand_dims(training_inputs, axis=-1)
    training_inputs = np.repeat(training_inputs, 3, axis=-1)

    # Train the new model
    new_model.fit(training_inputs, training_labels, batch_size=batch_size, epochs=epochs, verbose=1)

    return new_model


def evaluate_my_model(model, test_inputs, test_labels):
    # Reshape the inputs to match the expected input shape
    test_inputs = np.expand_dims(test_inputs, axis=-1)
    test_inputs = np.repeat(test_inputs, 3, axis=-1)

    # Evaluate the model
    _, accuracy = model.evaluate(test_inputs, test_labels, verbose=0)

    return accuracy




