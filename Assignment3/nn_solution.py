# Subodh Neupane
# 1001995258


import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_data(tr_data, test_data):
    # Find single maximum absolute value across all dimensions of training data
    max_abs_val = np.max(np.abs(tr_data))

    # Normalize both training and test data using this single value
    normalized_tr_data = tr_data / max_abs_val
    normalized_test_data = test_data / max_abs_val

    return normalized_tr_data, normalized_test_data


def initialize_weights(num_layers, units_per_layer, input_dim, num_classes):
    weights = []
    biases = []

    # Define dimensions for each layer
    layer_dims = [input_dim] + units_per_layer + [num_classes]

    # Initialize weights and biases for each layer
    for i in range(1, num_layers):
        # Initialize weights with small random values between -0.05 and 0.05
        w = np.random.uniform(-0.05, 0.05, (layer_dims[i], layer_dims[i - 1]))
        b = np.random.uniform(-0.05, 0.05, (layer_dims[i], 1))
        weights.append(w)
        biases.append(b)

    return weights, biases


def forward_pass(x, weights, biases):
    z_values = [x]  # Store all layer outputs
    a_values = []  # Store all layer inputs

    # Forward propagate through each layer
    current_input = x
    for w, b in zip(weights, biases):
        a = np.dot(w, current_input) + b
        z = sigmoid(a)
        a_values.append(a)
        z_values.append(z)
        current_input = z

    return z_values, a_values


def backward_pass(z_values, a_values, weights, biases, target, learning_rate):
    num_layers = len(weights) + 1
    deltas = [None] * (num_layers - 1)

    # Compute delta for output layer
    delta_output = z_values[-1] * (1 - z_values[-1]) * (z_values[-1] - target)
    deltas[-1] = delta_output

    # Compute deltas for hidden layers
    for l in range(len(deltas) - 2, -1, -1):
        delta = np.dot(weights[l + 1].T, deltas[l + 1]) * z_values[l + 1] * (1 - z_values[l + 1])
        deltas[l] = delta

    # Update weights and biases
    new_weights = []
    new_biases = []

    for l in range(len(weights)):
        w_update = -learning_rate * np.dot(deltas[l], z_values[l].T)
        b_update = -learning_rate * deltas[l]

        new_weights.append(weights[l] + w_update)
        new_biases.append(biases[l] + b_update)

    return new_weights, new_biases


def nn_train_and_test(tr_data, tr_labels, test_data, test_labels,
                      labels_to_ints, ints_to_labels, parameters):
    # Convert data to numpy arrays and normalize using single maximum value
    tr_data = np.array(tr_data)
    test_data = np.array(test_data)
    tr_data, test_data = normalize_data(tr_data, test_data)

    # Get number of classes
    num_classes = len(labels_to_ints)

    # Convert labels to one-hot encoding
    tr_labels_onehot = np.zeros((len(tr_labels), num_classes))
    for i in range(len(tr_labels)):
        tr_labels_onehot[i, tr_labels[i, 0]] = 1

    # Transpose data for matrix multiplication
    tr_data = tr_data.T
    test_data = test_data.T
    tr_labels_onehot = tr_labels_onehot.T

    # Initialize network parameters
    input_dim = tr_data.shape[0]
    weights, biases = initialize_weights(parameters.num_layers, parameters.units_per_layer, input_dim, num_classes)

    # Training
    learning_rate = 1.0

    for epoch in range(parameters.training_rounds):

        current_learning_rate = learning_rate * (0.98 ** epoch)

        # Process each training example
        for i in range(tr_data.shape[1]):
            x = tr_data[:, i:i + 1]
            target = tr_labels_onehot[:, i:i + 1]

            # Forward pass
            z_values, a_values = forward_pass(x, weights, biases)

            # Backward pass and update weights
            weights, biases = backward_pass(z_values, a_values, weights, biases,target, current_learning_rate)

    # Testing
    total_accuracy = 0
    total = len(test_labels)

    for i in range(len(test_data.T)):
        x = test_data[:, i:i + 1]
        true_label = ints_to_labels[test_labels[i, 0]]

        # Forward pass
        z_values, _ = forward_pass(x, weights, biases)
        predicted_class = np.argmax(z_values[-1])
        predicted_label = ints_to_labels[predicted_class]

        accuracy = 1.0 if predicted_label == true_label else 0.0
        total_accuracy += accuracy

        # Print prediction details with specified format
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' %
              (i + 1, str(predicted_label), str(true_label), accuracy))

        # Print final accuracy with specified format
    classification_accuracy = total_accuracy / total
    print('classification accuracy=%6.4f\n' % (classification_accuracy))