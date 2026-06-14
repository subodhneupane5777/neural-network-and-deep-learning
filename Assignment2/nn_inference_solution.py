#1001995258
#Subodh Neupane
import numpy as np

def nn_inference(layers, units, biases, weights, activation, input_vector):
    # Initialize lists to store a and z values
    a_values = [None] * (layers + 1)
    z_values = [None] * (layers + 1)

    # Input layer
    a_values[1] = None
    z_values[1] = input_vector

    # Forward propagation through layers 2 to L
    for i in range(2, layers + 1):
        # Compute a = (W.T * z_prev) + b
        a_values[i] = np.dot(weights[i], z_values[i - 1]) + biases[i]

        # Apply activation function
        if activation == "step":
            z_values[i] = (a_values[i] >= 0).astype(float)  # Step function
        elif activation == "sigmoid":
            z_values[i] = 1 / (1 + np.exp(-a_values[i]))  # Sigmoid function
        else:
            raise ValueError("Invalid activation function. Use 'step' or 'sigmoid'.")

    return a_values, z_values
