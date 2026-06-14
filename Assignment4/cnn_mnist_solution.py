from tensorflow import keras
from tensorflow.keras import layers


def load_mnist():

    # Load MNIST dataset from Keras
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Reshape and normalize training images
    training_inputs = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Reshape and normalize test images
    test_inputs = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to one-hot encoding (column vector)
    num_classes = 10
    training_labels = keras.utils.to_categorical(train_labels, num_classes).reshape(-1, num_classes)
    test_labels = keras.utils.to_categorical(test_labels, num_classes).reshape(-1, num_classes)

    return training_inputs, training_labels, test_inputs, test_labels


def create_and_train_model(training_inputs, training_labels, blocks,
                           filter_size, filter_number, region_size,
                           epochs, cnn_activation):

    # Create sequential model
    model = keras.Sequential()

    # Input layer is implicit in Keras

    # Add the specified number of convolutional blocks
    for _ in range(blocks):
        # Add convolutional layer
        model.add(layers.Conv2D(
            filters=filter_number,
            kernel_size=(filter_size, filter_size),
            activation=cnn_activation,
            padding='same',
            input_shape=training_inputs.shape[1:] if model.layers == [] else None
        ))

        # Add max pooling layer
        model.add(layers.MaxPooling2D(pool_size=(region_size, region_size)))

    # Flatten the output for the fully connected layer
    model.add(layers.Flatten())

    # Output layer (fully connected with softmax activation)
    model.add(layers.Dense(training_labels.shape[1], activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        training_inputs,
        training_labels,
        epochs=epochs,
        batch_size=64,
        verbose=1,
        validation_split=0.1
    )

    return model