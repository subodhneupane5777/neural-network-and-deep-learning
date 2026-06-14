# Subodh Neupane
# 1001995258


import numpy as np


def perceptron_train_and_test(tr_data, tr_labels, test_data, test_labels, training_rounds):
    #maximum absolute value in training data
    max_abs = np.max(np.abs(tr_data))

    # Normalize training and test data
    tr_data_normalized = tr_data / max_abs
    test_data_normalized = test_data / max_abs

    # Initialize weights and bias randomly between -0.05 and 0.05
    num_features = tr_data.shape[1]
    weights = np.zeros(num_features)

    weights = np.random.uniform(-0.05, 0.05, num_features)
    bias = np.random.uniform(-0.05, 0.05)

    # Training
    for round in range(training_rounds):
        # Calculate learning rate: η = 0.98^(r-1)
        learning_rate = 0.98 ** round

        # Process each training example
        for i in range(len(tr_data_normalized)):
            x = tr_data_normalized[i]

            # Calculate net input
            net_input = bias
            for j in range(num_features):
                net_input += weights[j] * x[j]

            # Apply sigmoid activation
            output = 1.0 / (1.0 + np.exp(-net_input))

            # Calculate error and delta
            error = tr_labels[i, 0] - output
            delta = error * output * (1 - output)

            # Update weights and bias
            for j in range(num_features):
                weights[j] += learning_rate * delta * x[j]
            bias += learning_rate * delta

    # Testing
    total_accuracy = 0

    for i in range(len(test_data_normalized)):
        x = test_data_normalized[i]

        # Calculate net input and output
        net_input = bias
        for j in range(num_features):
            net_input += weights[j] * x[j]

        output = 1.0 / (1.0 + np.exp(-net_input))

        # Convert output to predicted class (0 if < 0.5, 1 if >= 0.5)
        predicted_class = 1 if output >= 0.5 else 0
        true_class = test_labels[i, 0]

        # Calculate accuracy
        accuracy = 1.0 if predicted_class == true_class else 0.0
        total_accuracy += accuracy

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' %
              (i, str(predicted_class), str(true_class), accuracy))

    classification_accuracy = total_accuracy / len(test_data)
    print('classification accuracy=%6.4f\n' % (classification_accuracy))