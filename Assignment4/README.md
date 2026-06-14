# Assignment 5: Deep Learning & Convolutional Neural Networks

This repository contains implementations for training and evaluating both Dense (Fully Connected) Neural Networks and Convolutional Neural Networks (CNNs) using TensorFlow and Keras. The project explores image classification on the MNIST dataset as well as classification on synthetic/UCI text datasets.

---

## My Work

### Implemented Solution Modules
* **`cnn_mnist_solution.py`**: Contains the code to load and preprocess the 4D spatial MNIST dataset, and constructs a dynamic CNN architecture using sequential convolutional and max-pooling blocks.
* **`dense_mnist_solution.py`**: Handles loading, flattening (2D to 1D), and normalizing the MNIST dataset for a fully connected multi-layer network.
* **`nn_keras_solution.py`**: Implements a highly configurable Dense Neural Network wrapper capable of dynamically stacking hidden layers and custom activations, alongside a specialized test evaluation loop that handles classification tie-breaking logic.

### Provided Starter & Utility Files
* **`cnn_mnist_base.py`**, **`dense_mnist_base.py`**, **`nn_keras_base.py`**: High-level execution scripts that define hyperparameters (layers, filters, epochs, activations), invoke the solution functions, and output final test accuracies.
* **`uci_load.py`**: A utility script used to parse text-formatted data files, automatically generating internal index mappings for string or integer target labels.

---

## Topics Covered

* **Deep Learning Frameworks**: Building models using the TensorFlow Keras `Sequential` API.
* **Convolutional Neural Networks (CNNs)**: Implementing spatial feature extraction pipelines using 2D convolutions (`Conv2D`), zero-padding setups, and spatial reduction (`MaxPooling2D`).
* **Dense Neural Networks (DNNs)**: Stacking fully connected (`Dense`) layers configured with multi-tiered hidden nodes and varying activation functions such as `ReLU`, `tanh`, and `sigmoid`.
* **Data Pipelines & Scaling**: Normalizing pixel elements and arbitrary numeric features down to a floating-point scale between $0.0$ and $1.0$ for optimized gradient descent.
* **Categorical Loss Strategies**: Working with different target formats using both `categorical_crossentropy` (requiring one-hot encoded vectors) and `sparse_categorical_crossentropy` (working directly with integer labels).
* **Custom Prediction Logic**: Formulating manual evaluation matrices to securely track index predictions and handle probability ties using random selection routines.

---
