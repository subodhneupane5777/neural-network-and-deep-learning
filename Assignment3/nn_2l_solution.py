# Subodh Neupane
# 1001995258



import numpy as np

def nn_2l_train_and_test(tr_data, tr_labels, test_data, test_labels,
                         labels_to_ints, ints_to_labels, training_rounds):

    #maximum absolute value in training data
    max_abs = np.max(np.abs(tr_data))

    # normalize training and test data
    tr_data_normalized = tr_data / max_abs
    test_data_normalized = test_data / max_abs

    num_features = tr_data.shape[1]
    num_classes = len(labels_to_ints)

    # Initialize weights and biases
    weights = np.random.uniform(-0.05, 0.05, (num_features, num_classes))
    biases = np.random.uniform(-0.05, 0.05, num_classes)

    # Convert labels to one-hot encoding
    tr_labels_onehot = np.zeros((len(tr_labels), num_classes))
    for i in range(len(tr_labels)):
        label = int(tr_labels[i, 0])  # Convert to int
        class_idx = label  # Use label directly as it aligns with labels_to_ints values
        tr_labels_onehot[i, class_idx] = 1

    # Training
    for round in range(training_rounds):
        #learning rate
        learning_rate = 0.98 ** round

        # process each training
        for i in range(len(tr_data_normalized)):
            x = tr_data_normalized[i]

            # Calculate net input for each output unit
            net_inputs = biases + np.dot(x, weights)

            # Apply sigmoid activation
            outputs = 1.0 / (1.0 + np.exp(-net_inputs))

            # Calculate error for each output unit
            deltas = (tr_labels_onehot[i] - outputs) * outputs * (1 - outputs)

            # Update weights and bias for each output unit
            weights += learning_rate * np.outer(x, deltas)
            biases += learning_rate * deltas

    # Testing
    total_accuracy = 0

    for i in range(len(test_data_normalized)):
        x = test_data_normalized[i]

        # Calculate outputs
        net_inputs = biases + np.dot(x, weights)
        outputs = 1.0 / (1.0 + np.exp(-net_inputs))

        # Find highest output
        max_output = np.max(outputs)
        tied_classes = np.where(np.abs(outputs - max_output) < 1e-10)[0]

        # Calculate accuracy
        label = int(test_labels[i, 0])  # Convert to int
        true_class = label

        if true_class in tied_classes:
            accuracy = 1.0 / len(tied_classes)
        else:
            accuracy = 0.0

        # Convert predicted class back to original label
        predicted_class_label = ints_to_labels[tied_classes[0]]
        true_class_label = ints_to_labels[true_class]

        total_accuracy += accuracy

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (i, str(predicted_class_label), str(true_class_label), accuracy))

    classification_accuracy = total_accuracy / len(test_data)
    print('classification accuracy=%6.4f' % (classification_accuracy))
