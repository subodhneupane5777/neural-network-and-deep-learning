import numpy as np
from tensorflow import keras
import random


def create_and_train_model(training_inputs, training_labels, layers, units_per_layer, epochs, hidden_activations):


    # Get input dimensions and number of classes
    input_dim = training_inputs.shape[1]
    num_classes = np.max(training_labels) + 1

    # Create model
    model = keras.Sequential()

    # Add input layer
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))

    # Add hidden layers if there are any (layers > 2)
    if layers > 2:
        for i in range(layers - 2):
            model.add(keras.layers.Dense(
                units=units_per_layer[i],
                activation=hidden_activations[i]
            ))

    # Add output layer with softmax activation
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile model with Adam optimizer and Categorical Cross Entropy loss
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


def test_model(model, test_inputs, test_labels, ints_to_labels):


    # Get model predictions (probabilities for each class)
    predictions = model.predict(test_inputs)

    total_accuracy = 0.0
    num_test_samples = test_inputs.shape[0]

    for test_index in range(num_test_samples):
        # Get the true label
        true_label_int = test_labels[test_index, 0]
        true_label = ints_to_labels[int(true_label_int)]

        # Get predicted probabilities for current test sample
        pred_probs = predictions[test_index]

        # Find the maximum probability value(s)
        max_prob = np.max(pred_probs)

        # Find indices of classes that tied for the maximum probability
        max_indices = np.where(pred_probs == max_prob)[0]

        # Handle ties by randomly selecting one of the tied classes
        if len(max_indices) > 1:  # If there's a tie
            # Check if the true class is among the tied classes
            if true_label_int in max_indices:
                accuracy = 1.0 / len(max_indices)
            else:
                accuracy = 0.0

            # Randomly select one of the tied classes as the prediction
            predicted_index = random.choice(max_indices)
        else:  # No tie
            predicted_index = max_indices[0]
            accuracy = 1.0 if predicted_index == true_label_int else 0.0

        # Get the predicted class label
        predicted_class = ints_to_labels[predicted_index]

        # Print results for this test sample
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (test_index, predicted_class, true_label, accuracy))

        # Add to total accuracy
        total_accuracy += accuracy

    # Calculate overall test accuracy
    test_accuracy = total_accuracy / num_test_samples

    return test_accuracy