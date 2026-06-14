#Subodh neupane
#1001995258
import numpy as np

def perceptron_inference(b,w,activation,input_vector):
    # Compute weighted sum of inputs plus bias
    a = np.dot(w.T, input_vector)[0, 0] + b

    # Apply activation function
    if activation == "step":
        z = 1.0 if a >= 0 else 0.0
    elif activation == "sigmoid":
        z = 1 / (1 + np.exp(-a))
    else:
        raise ValueError("Invalid activation function. Use 'step' or 'sigmoid'.")

    return a, z