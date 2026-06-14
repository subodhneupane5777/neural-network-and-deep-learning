from tensorflow import keras
import ssl


def load_mnist():
    # Configure SSL context to avoid certificate verification issues
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load MNIST dataset using Keras
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Normalize the input values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Reshape the input data: from (samples, 28, 28) to (samples, 784)
    train_images = train_images.reshape(train_images.shape[0], 28 * 28)
    test_images = test_images.reshape(test_images.shape[0], 28 * 28)


    return train_images, train_labels, test_images, test_labels


def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):
    # Get input dimensions and number of classes
    input_dim = training_inputs.shape[1]
    num_classes = 10  # MNIST has 10 classes (digits 0-9)

    # Create model
    model = keras.Sequential()

    # Add input layer
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))

    # Add hidden layers if there are any (layers > 2)
    if layers > 2:
        for i in range(layers - 2):
            if i < len(units_per_layer) and i < len(hidden_activations):
                model.add(keras.layers.Dense(
                    units=units_per_layer[i],
                    activation=hidden_activations[i]
                ))

    # Add output layer with softmax activation
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile model with Adam optimizer and sparse Categorical Cross Entropy loss
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        training_inputs,
        training_labels,
        epochs=epochs,
        verbose=1
    )

    return model