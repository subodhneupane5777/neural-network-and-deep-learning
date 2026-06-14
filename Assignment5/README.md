# Assignment 5: Deep Learning — Transfer Learning & Time-Series Classification

This repository contains implementation tasks focusing on Deep Learning concepts using TensorFlow and Keras. It covers two main sub-projects:
1. **Transfer Learning**: Reusing features from a spatial CNN model trained on CIFAR-10 to classify a downsampled MNIST handwritten digit dataset.
2. **Time-Series Analysis**: Processing tabular weather logs (Jena Climate dataset) to extract temporal sliding windows and train deep architectures that predict seasonal calendar months.

---

## My Work

### Implemented Solution Modules
* **`cifar2mnist_solution.py`**: 
  * `train_model`: Normalizes inputs and compiles/trains a base Convolutional Neural Network (CNN) on CIFAR-10.
  * `load_and_refine`: Drops the terminal classification layer of the pre-trained CIFAR model, freezes its feature extraction layers, modifies the grayscale 1-channel MNIST inputs to match the 3-channel RGB requirement, and attaches a new dense head for target refinement.
  * `evaluate_my_model`: Preprocesses validation matrices and evaluates overall classification metrics on the test split.
* **`month_solution.py`**:
  * `data_normalization`: Calculates and applies isolated $z$-score feature normalization computed exclusively over training bounds.
  * `make_inputs_and_targets`: Samples sequential, down-sampled history windows across climatic parameters, mapping structural targets to the calendar month index found at the window's midpoint.
  * `build_and_train_dense`: Sets up a dense flattening architecture utilizing `tanh`, `relu`, and dropout regularizations along with best-epoch validation checkpoint metrics.
  * `test_model` & `confusion_matrix`: Manages inference tracking, calculation metrics, and builds $12 \times 12$ cross-class error-frequency distribution grids.

### Provided Starter & Utility Files
* **`cifar2mnist_base.py`**: Oversees programmatic model configuration, small-sample class extraction (100 samples per category), and validation loops for transfer evaluation.
* **`month_base.py`**: Orchestrates string-line file conversions, dataset time boundaries split splits (50% Train / 25% Val / 25% Test), execution pipelines, and matrix formatting prints.
* **`cifar10_e20_b128_6.keras`**: A pre-saved Keras layout file storing architectural weights after initial training sequences over baseline targets.

---

## Topics Covered

* **Transfer Learning & Weight Freezing**: Slicing dense heads from pretrained layers, locking functional structural filters (`layer.trainable = False`), and optimizing weights exclusively within appended output blocks.
* **Channel Dimension Adapting**: Converting single-channel grayscale elements into synthetic three-channel pseudo-RGB instances via tensor extensions (`expand_dims` and `repeat`) to satisfy pre-configured network constraints.
* **Time-Series Windowing**: Transforming tabular data matrices into 3D structural tensors `(samples, window_size, features)` with customized index stride rates.
* **Data Leakage Mitigation**: Ensuring statistical means and standard deviations are strictly evaluated using the isolated training slice before scaling validation/test partitions.
* **Performance Checkpointing**: Employing automated callback instances (`ModelCheckpoint`) to continuously capture and store optimal weight structures on evaluation improvements.
* **Confusion Matrices**: Generating frequency grids to visually diagnose classification behavior and misclassifications across all target indices.

---
